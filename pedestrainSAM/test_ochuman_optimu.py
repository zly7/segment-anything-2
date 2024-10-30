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

import optuna  # Import Optuna
import optuna.visualization as vis  # Import visualization module

from sam2.build_sam import build_sam2_for_self_train
from .trainer_ddp import PedestrainSAM2
from dataset_zly.dayasets_ochuman_test import OCHumanSegmentationTest  # Update the import path
from .automask_from_sam1 import PedestrainSamAutomaticMaskGenerator

# Modify `evaluate_ap` function to evaluate the model performance and return the AP score.
def evaluate_ap(config, stability_score_thresh, pred_iou_thresh, person_probability_thresh):
    # Set device index
    device_index = 0
    device = torch.device('cuda', device_index)

    # Build the SAM2 model
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model, missing_keys, unexpected_keys = build_sam2_for_self_train(
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

    # Paths to your OCHuman dataset
    ann_file = "/data2/zly/CrowdSeg/OCHuman/ochuman_coco_format_test_range_0.00_1.00_full_labelled.json"  # Update with the actual path
    # ann_file = "/data2/zly/CrowdSeg/OCHuman/coco_annotation_first_10.json"
    # ann_file = "/data2/zly/CrowdSeg/OCHuman/ochuman_coco_format_test_range_0.00_1.00.json"
    img_dir = "/data2/zly/CrowdSeg/OCHuman/images"  # Update with the actual path

    current_datetime = datetime.now()
    current_date_str = current_datetime.date().strftime("%Y-%m-%d")
    which_to_replace_img_dir = "pred_images_" + current_date_str + "hq_finetune_all_test_no_filter"
    output_base_dir = img_dir.replace("images", which_to_replace_img_dir)

    if os.path.exists(output_base_dir):
        shutil.rmtree(os.path.join(output_base_dir, "code"))
    if not os.path.exists(os.path.join(output_base_dir, "code")):
        os.makedirs(os.path.join(output_base_dir, "code"))
    shutil.copytree("./pedestrainSAM", os.path.join(output_base_dir, "code", "pedestrainSAM"))
    shutil.copytree("./sam2", os.path.join(output_base_dir, "code", "sam2"))
    shutil.copytree("./train_config", os.path.join(output_base_dir, "code", "train_config"))

    # Initialize the test dataset
    test_dataset = OCHumanSegmentationTest(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=None
    )

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize the mask generator with hyperparameters
    mask_generator = PedestrainSamAutomaticMaskGenerator(
        model=pedestrian_sam2,
        points_per_batch=16 * 16,
        points_per_side=16,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        person_probability_thresh=person_probability_thresh,
        vis=False,
        vis_immediate_folder=which_to_replace_img_dir,
        loguru_path=f"./logs/ochuman_test/{current_date_str}.log",
        use_hq=config["test"]["use_hq"],
    )

    # Load the ground truth annotations
    coco_gt = COCO(ann_file)

    # Prepare to collect predictions in COCO format
    coco_predictions = []

    # Iterate over the test dataset
    for idx, (image, img_path, rle_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = image.squeeze(0)  # Remove batch dimension
        img_path = img_path[0]  # Get the image path string
        image_np = np.array(image)

        # Generate masks for the image
        masks = mask_generator.generate(image_np, img_path)

        # Prepare predictions in COCO format
        image_id = test_dataset.img_ids[idx]
        assert image_id in coco_gt.getImgIds(), f"Image ID {image_id} not found in annotations!"
        for mask in masks:
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            score = mask['predicted_iou']

            if isinstance(segmentation, dict) and 'counts' in segmentation:
                rle = segmentation
            else:
                rle = mask_utils.encode(np.asfortranarray(segmentation.astype(np.uint8)))

            pred = {
                'image_id': image_id,
                'category_id': 1,
                'segmentation': {
                    'size': rle['size'],
                    'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                },
                'bbox': bbox,
                'score': score,
            }
            coco_predictions.append(pred)
        if len(masks) == 0:
            height, width = image_np.shape[:2]
            placeholder_mask = np.zeros((height, width), dtype=np.uint8)
            rle = mask_utils.encode(np.asfortranarray(placeholder_mask))
            placeholder_bbox = [0, 0, 1, 1]
            placeholder_score = 0.0
            pred = {
                'image_id': image_id,
                'category_id': 1,
                'segmentation': {
                    'size': rle['size'],
                    'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                },
                'bbox': placeholder_bbox,
                'score': placeholder_score,
            }
            coco_predictions.append(pred)

    # Save predictions to a JSON file
    coco_result_save_json_path = os.path.join(output_base_dir, "result.json")
    with open(coco_result_save_json_path, 'w') as f:
        json.dump(coco_predictions, f)

    # Load results and evaluate
    coco_dt = coco_gt.loadRes(coco_result_save_json_path)

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract the AP from COCOeval
    ap = coco_eval.stats[0]

    shutil.rmtree(output_base_dir, ignore_errors=True)

    return ap

# Define Optuna objective function to optimize hyperparameters
def objective(trial):
    # Load the configuration
    config_path = "./train_config/train_large.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    current_stability_score_thresh = config["test"].get("stability_score_thresh", 0.6)
    current_pred_iou_thresh = config["test"].get("pred_iou_thresh", 0.6)
    current_person_probability_thresh = config["test"].get("person_probability_thresh", 0.7)

    stability_score_thresh = trial.suggest_float(
        "stability_score_thresh",
        max(0.0, current_stability_score_thresh - 0.2),
        min(1.0, current_stability_score_thresh + 0.2),
        step=0.05
    )

    pred_iou_thresh = trial.suggest_float(
        "pred_iou_thresh",
        max(0.0, current_pred_iou_thresh - 0.2),
        min(1.0, current_pred_iou_thresh + 0.2),
        step=0.05
    )

    person_probability_thresh = trial.suggest_float(
        "person_probability_thresh",
        max(0.0, current_person_probability_thresh - 0.2),
        min(1.0, current_person_probability_thresh + 0.2),
        step=0.05
    )

    ap = evaluate_ap(
        config,
        stability_score_thresh=stability_score_thresh,
        pred_iou_thresh=pred_iou_thresh,
        person_probability_thresh=person_probability_thresh
    )

    trial.set_user_attr("AP", ap)
    return ap

# Function to optimize hyperparameters and display visualizations
def optimize_hyperparameters(n_trials=50):
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    print("Best trial parameters:", best_trial.params)
    print("Best trial AP:", best_trial.value)

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
    best_trial = optimize_hyperparameters(n_trials=50)
