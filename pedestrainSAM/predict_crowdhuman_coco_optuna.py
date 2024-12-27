import argparse
import os
import shutil
import json
from datetime import datetime

import torch
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

import optuna  # 导入 Optuna
import optuna.visualization as vis  # 导入可视化模块

from sam2.build_sam import build_sam2_for_self_train
from .trainer_ddp import PedestrainSAM2
from .automask_from_sam1 import PedestrainSamAutomaticMaskGenerator
from loguru import logger
from PIL import Image
import yaml

class CrowdHumanDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json_path, images_dir, transform=None, width_resize=1920.0):
        self.images_dir = images_dir
        self.transform = transform
        with open(coco_json_path, "r") as f:
            coco = json.load(f)
        self.width_resize = width_resize
        self.images_info = coco["images"]
        self.image_id_to_info = {image["id"]: image for image in self.images_info}

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image_filename = image_info["file_name"]
        image_id = image_info["id"]
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        # Resize image, keeping aspect ratio and setting long edge to width_resize
        max_dim = max(image.size)
        scale = self.width_resize / max_dim
        new_size = tuple(int(dim * scale) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)

        image = np.array(image)
        return image, image_path, image_id, original_size

def evaluate_ap(config, stability_score_thresh, pred_iou_thresh, person_probability_thresh):
    # Set device index,这里总是0，因为用了cuda——visible——devices
    device_index = 0
    device = torch.device('cuda', device_index)

    # Build the SAM2 model
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model, _, _ = build_sam2_for_self_train(
        model_config_path_for_build_sam2,
        sam2_checkpoint,
        device=device,
        apply_postprocessing=True,
    )

    # Initialize the PedestrainSAM2 model
    pedestrian_sam2 = PedestrainSAM2(
        model=sam2_model,
        config=config,
        device_index=device_index,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    )
    pedestrian_sam2.load_classifier(config["model"]["pretrain_classfier_path"])
    pedestrian_sam2.sam2_model.eval()
    pedestrian_sam2.classifier.eval()
    # Paths to CrowdHuman dataset
    img_dir = "/data2/zly/mot_data/crowdhuman/Images"  # Update with the actual path
    # img_dir = "/home/cseadmin/zly/data/crowdhuman/Images"
    # annotation_file_name = "/data2/zly/mot_data/crowdhuman/crowdhuman_train_split.json
    # annotation_file_name = "/data2/zly/mot_data/crowdhuman/crowdhuman_val_split.json"
    # annotation_file_name = "/home/cseadmin/zly/data/crowdhuman/crowdhuman_val_split.json"
    annotation_file_name = "/data2/zly/mot_data/crowdhuman/crowdhuman_train_sample_10.json"

    replace_date_str = f"sam2_pred_images_{datetime.now().astimezone().strftime('%Y-%m-%d-%H')}"
    output_base_dir = img_dir.replace("Images", replace_date_str)
    if os.path.exists(os.path.join(output_base_dir, "code")):
        shutil.rmtree(os.path.join(output_base_dir, "code"))
    shutil.copytree("./pedestrainSAM", os.path.join(output_base_dir, "code", "pedestrainSAM"))  # 保存一些代码
    shutil.copytree("./sam2", os.path.join(output_base_dir, "code", "sam2"))
    shutil.copytree("./train_config", os.path.join(output_base_dir, "code", "train_config"))
    # shutil.copy(config["model"]["model_config"], os.path.join(config["train"]["save_path"], os.path.basename(config["model"]["model_config"])))

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Initialize the CrowdHuman dataset
    test_dataset = CrowdHumanDataset(
        coco_json_path=annotation_file_name,
        images_dir=img_dir,
        transform=None  # set_image会处理
    )

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize the mask generator with hyperparameters
    mask_generator = PedestrainSamAutomaticMaskGenerator(
        model=pedestrian_sam2,
        points_per_batch=64 * 64 // 16,  # Adjust based on your GPU memory
        points_per_side=64,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        person_probability_thresh=person_probability_thresh,
        vis=True,  # 关闭可视化以加快速度
        vis_detailed_process_prabability=0.0,
        replaced_immediate_path="Images",
        vis_immediate_folder_to_replace_images=replace_date_str,
        vis_resize_width=1280,
        loguru_path=f"./logs/crowdhuman_test/{datetime.now().strftime('%Y-%m-%d')}.log",
    )

    coco_predictions = []
    coco_images = []
    annotation_id = 1  # Unique ID for each annotation

    # Iterate over the test dataset
    for idx, (image, image_abs_path, image_id, original_size) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = image.squeeze(0)  # Remove batch dimension
        image_np = np.array(image)
        image_abs_path = image_abs_path[0]

        # Retrieve original image size
        orig_width, orig_height = original_size[0].item(), original_size[1].item()

        # Add image information to coco_images with original size
        coco_images.append({
            'id': image_id.item(),
            'file_name': os.path.basename(image_abs_path),
            'width': orig_width,
            'height': orig_height
        })

        # Compute the scale factor used during resizing
        scale = test_dataset.width_resize / max(orig_width, orig_height)

        # Generate masks for the image
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = mask_generator.generate(image_np, image_abs_path)

        image_id_val = image_id.item()
        for mask in masks:
            bbox_resized = mask['bbox']  # [x, y, w, h] in resized image
            score = mask['predicted_iou']  # Use predicted IoU as the confidence score

            # Rescale bbox to original image size
            bbox_original = [coord / scale for coord in bbox_resized]  # [x, y, w, h]

            # Ensure bbox is within original image boundaries
            x, y, w, h = bbox_original
            if x < 0 or y < 0 or (x + w) > orig_width or (y + h) > orig_height:
                logger.warning(
                    f"Bounding box {bbox_resized} for image {os.path.basename(image_abs_path)} "
                    f"rescaled to {bbox_original} exceeds image boundaries ({orig_width}, {orig_height}). "
                    f"Clipping the bbox to fit within the image."
                )
            bbox_original = [round(coord, 2) for coord in [x, y, w, h]]

            # Calculate area
            area = round(w * h, 2)

            # Prepare annotation dictionary without 'segmentation'
            pred = {
                'id': annotation_id,
                'image_id': image_id_val,
                'category_id': 1,  # Assuming 'person' class ID is 1
                'bbox': bbox_original,
                'score': round(score, 4),
                'area': area,
                'iscrowd': 0,
            }
            coco_predictions.append(pred)
            annotation_id += 1

        # If no masks are generated, add a placeholder annotation
        if len(masks) == 0:
            placeholder_bbox = [0, 0, 1, 1]
            pred = {
                'id': annotation_id,
                'image_id': image_id_val,
                'category_id': 1,
                'bbox': placeholder_bbox,
                'score': 0.0,
                'area': 1,
                'iscrowd': 0,
            }
            coco_predictions.append(pred)
            annotation_id += 1

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(test_loader)} images")

    # Define category information
    coco_categories = [
        {
            'id': 1,
            'name': 'person',
            'supercategory': 'person'
        }
    ]

    # Define info section (optional)
    coco_info = {
        'description': 'CrowdHuman Predictions',
        'version': '1.0',
        'year': datetime.now().year,
        'contributor': 'The CrowdHuman prediction from the SAM2 automask',
        'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Combine all parts
    coco_output = {
        'info': coco_info,
        'images': coco_images,
        'annotations': coco_predictions,
        'categories': coco_categories
    }

    coco_result_save_json_path = os.path.join(output_base_dir, "predict_crowdhuman_coco_format.json")
    with open(coco_result_save_json_path, 'w') as f:
        json.dump(coco_output, f)

    print(f"COCO format predictions saved to {coco_result_save_json_path}")

    # Load ground truth annotations
    coco_gt = COCO(annotation_file_name)
    coco_dt = COCO(coco_result_save_json_path)

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract the AP from COCOeval
    ap = coco_eval.stats[0]  # AP at IoU=0.50:0.95

    return {
        "AP": ap,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CrowdHuman SAM2 model with Optuna.")
    # 不设置就必须设置成None
    parser.add_argument(
        "--config_path",
        type=str,
        default= "./train_config/train_large.yaml",
        help="Path to the config file"
    )

    # Define ranges for stability_score_thresh
    parser.add_argument(
        "--stability_score_thresh_range",
        type=float,
        nargs=2,
        metavar=('LOW', 'HIGH'),
        # default=[0.5, 0.7],
        default= None,
        help="Set lower and upper bounds for stability_score_thresh"
    )

    # Define ranges for pred_iou_thresh
    parser.add_argument(
        "--pred_iou_thresh_range",
        type=float,
        nargs=2,
        metavar=('LOW', 'HIGH'),
        # default=[0.5, 0.7],
        default= None,
        help="Set lower and upper bounds for pred_iou_thresh"
    )

    # Define ranges for person_probability_thresh
    parser.add_argument(
        "--person_probability_thresh_range",
        type=float,
        nargs=2,
        metavar=('LOW', 'HIGH'),
        # default=[0.7, 0.85],
        default= None,
        help="Set lower and upper bounds for person_probability_thresh"
    )

    # Number of Optuna trials
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Number of Optuna trials"
    )

    return parser.parse_args()

def load_config(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override stability_score_thresh_range if provided
    if args.stability_score_thresh_range is not None:
        low, high = args.stability_score_thresh_range
        if low > high:
            raise ValueError("stability_score_thresh_range: LOW cannot be greater than HIGH")
        config["test"]["stability_score_thresh_range"] = [low, high]

    # Override pred_iou_thresh_range if provided
    if args.pred_iou_thresh_range is not None:
        low, high = args.pred_iou_thresh_range
        if low > high:
            raise ValueError("pred_iou_thresh_range: LOW cannot be greater than HIGH")
        config["test"]["pred_iou_thresh_range"] = [low, high]

    # Override person_probability_thresh_range if provided
    if args.person_probability_thresh_range is not None:
        low, high = args.person_probability_thresh_range
        if low > high:
            raise ValueError("person_probability_thresh_range: LOW cannot be greater than HIGH")
        config["test"]["person_probability_thresh_range"] = [low, high]

    return config

# Define Optuna objective function to optimize hyperparameters
def objective(trial, config):
    stability_score_thresh_range = config["test"].get("stability_score_thresh_range", [0.5, 0.55])
    stability_score_thresh_interval = config["test"].get("stability_score_thresh_interval", 0.05)

    pred_iou_thresh_range = config["test"].get("pred_iou_thresh_range", [0.5, 0.7])
    pred_iou_thresh_interval = config["test"].get("pred_iou_thresh_interval", 0.05)

    person_probability_thresh_range = config["test"].get("person_probability_thresh_range", [0.7, 0.85])
    person_probability_thresh_interval = config["test"].get("person_probability_thresh_interval", 0.05)

    stability_score_thresh = trial.suggest_float(
        "stability_score_thresh",
        stability_score_thresh_range[0],
        stability_score_thresh_range[1],
        step=stability_score_thresh_interval
    )

    pred_iou_thresh = trial.suggest_float(
        "pred_iou_thresh",
        pred_iou_thresh_range[0],
        pred_iou_thresh_range[1],
        step=pred_iou_thresh_interval
    )

    person_probability_thresh = trial.suggest_float(
        "person_probability_thresh",
        person_probability_thresh_range[0],
        person_probability_thresh_range[1],
        step=person_probability_thresh_interval
    )

    metrics = evaluate_ap(
        config,
        stability_score_thresh=stability_score_thresh,
        pred_iou_thresh=pred_iou_thresh,
        person_probability_thresh=person_probability_thresh
    )
    
    # Set additional metrics as user attributes
    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, metric_value)

    # The objective is to maximize AP
    return metrics['AP']


# Function to optimize hyperparameters and display visualizations
def optimize_hyperparameters(n_trials=50, config=None):
    study = optuna.create_study(
        study_name="crowdhuman_optimization_with_sam2.1",
        direction="maximize",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

    best_trial = study.best_trial
    print("Best trial parameters:", best_trial.params)
    print("Best trial AP:", best_trial.value)

    # 创建日志目录
    os.makedirs("./logs/optuna", exist_ok=True)

    optimization_history = vis.plot_optimization_history(study)
    optimization_history.write_image("./logs/optuna/optimization_history.png")

    param_importances = vis.plot_param_importances(study)
    param_importances.write_image("./logs/optuna/param_importances.png")

    parallel_coordinate = vis.plot_parallel_coordinate(study)
    parallel_coordinate.write_image("./logs/optuna/parallel_coordinate.png")

    slice_plot = vis.plot_slice(study)
    slice_plot.write_image("./logs/optuna/slice_plot.png")

    return best_trial

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    best_trial = optimize_hyperparameters(n_trials=args.n_trials, config=config)
