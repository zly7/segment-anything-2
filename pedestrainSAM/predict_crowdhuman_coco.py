import os
import shutil
import torch
import json
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
import yaml

from sam2.build_sam import build_sam2_for_self_train
from .trainer_ddp import PedestrainSAM2
from .automask_from_sam1 import PedestrainSamAutomaticMaskGenerator

from PIL import Image
from loguru import logger
class CrowdHumanDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json_path, images_dir, transform=None, width_resize = 1920.0):
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
        # Resize image, keeping aspect ratio and setting long edge to 1920
        max_dim = max(image.size)
        scale = self.width_resize / max_dim
        new_size = tuple(int(dim * scale) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)

        image = np.array(image)
        return image, image_path, image_id, original_size


def main():
    # Load the configuration
    config_path = "./train_config/train_large.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set device index
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

    # Paths to CrowdHuman dataset
    img_dir = "/data2/zly/mot_data/crowdhuman/Images"  # Update with the actual path
    # annotation_file_name = "/data2/zly/mot_data/crowdhuman/crowdhuman_train_split.json
    annotation_file_name = "/data2/zly/mot_data/crowdhuman/crowdhuman_val_split.json"
    # annotation_file_name = "/data2/zly/mot_data/crowdhuman/crowdhuman_train_sample_10.json"

    replace_date_str = "sam2_pred_images_" + datetime.now().date().strftime("%Y-%m-%d-%H")
    output_base_dir = img_dir.replace("Images", replace_date_str)
    if os.path.exists(os.path.join(output_base_dir, "code")):
        shutil.rmtree(os.path.join(output_base_dir, "code"))
    shutil.copytree("./pedestrainSAM", os.path.join(output_base_dir,"code", "pedestrainSAM")) # 保存一些代码
    shutil.copytree("./sam2", os.path.join(output_base_dir,"code", "sam2"))
    shutil.copytree("./train_config", os.path.join(output_base_dir,"code", "train_config"))
    shutil.copy(config_path, os.path.join(config["train"]["save_path"], os.path.basename(config_path)))
        
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

    # Initialize the mask generator
    mask_generator = PedestrainSamAutomaticMaskGenerator(
        model=pedestrian_sam2,
        points_per_batch=32 * 32 // 2,  # Adjust based on your GPU memory
        points_per_side=32,
        pred_iou_thresh=config["test"]["pred_iou_thresh"],
        stability_score_thresh=config["test"]["stability_score_thresh"],
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        person_probability_thresh=config["test"]["person_probability_thresh"],
        vis = True,
        vis_detailed_process_prabability=0.0,
        replaced_immediate_path="Images",
        vis_immediate_folder_to_replace_images= replace_date_str,
        vis_resize_width= 1280,
        loguru_path=f"./logs/crowdhuman_test/{datetime.now().date().strftime('%Y-%m-%d')}.log",
        use_hq=config["test"]["use_hq"],
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
        scale = test_dataset.width_resize / orig_width

        # Generate masks for the image
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = mask_generator.generate(image_np, image_abs_path)

        image_id_val = image_id[0].item()
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
        'contributor': 'The crowdhuman prediction from the SAM2 automask',
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

if __name__ == "__main__":
    main()
