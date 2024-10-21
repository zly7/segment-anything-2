from datetime import datetime
import os
import torch
from tqdm import tqdm
import yaml
import numpy as np
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sam2.build_sam import build_sam2_for_self_train
from .trainer_ddp import PedestrainSAM2
from dataset_mots.dayasets_ochuman_test import OCHumanSegmentationTest  # Update the import path
from .automask_from_sam1 import PedestrainSamAutomaticMaskGenerator
from pycocotools import mask as masks_utils
def main():
    # Load the configuration
    # config_path = "../train_config/train_large.yaml"
    config_path = "./train_config/train_large.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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
    img_dir = "/data2/zly/CrowdSeg/OCHuman/images"  # Update with the actual path
    current_datetime = datetime.now()
    current_date_str = current_datetime.date().strftime("%Y-%m-%d")
    which_to_replace_img_dir = "pred_images_" + current_date_str
    output_base_dir = img_dir.replace("images", which_to_replace_img_dir)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    # Initialize the test dataset
    test_dataset = OCHumanSegmentationTest(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=None  # You can add any transformations if needed
    )

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=4,  # Adjust based on your system
        pin_memory=True,
    )
    current_datetime = datetime.now()
    current_date_str = current_datetime.date().strftime("%Y-%m-%d")
    # Initialize the mask generator
    mask_generator = PedestrainSamAutomaticMaskGenerator(
        model=pedestrian_sam2,
        points_per_batch=32*32,  # Adjust based on your GPU memory
        points_per_side=32, # 这里是perside
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires OpenCV for post-processing
        person_probability_thresh=0.7,
        vis=True,  # Set to True if you want to save visualizations
        vis_immediate_folder=which_to_replace_img_dir,
        loguru_path=f"./logs/ochuman_test/{current_date_str}.log"
    )

    # Load the ground truth annotations
    coco_gt = COCO(ann_file)

    # Prepare to collect predictions in COCO format
    coco_predictions = []

    # Iterate over the test dataset
    for idx, (image, img_path, rle_masks) in tqdm(enumerate(test_loader)):
        image = image.squeeze(0)  # Remove batch dimension
        img_path = img_path[0]  # Get the image path string
        image_np = np.array(image)

        # Generate masks for the image
        masks = mask_generator.generate(image_np,img_path)

        # Prepare predictions in COCO format
        image_id = test_dataset.img_ids[idx]
        assert image_id in coco_gt.getImgIds(), f"Image ID {image_id} not found in annotations!"
        for mask in masks:
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            score = mask['predicted_iou']  # Use predicted IoU as the confidence score

            # Convert segmentation to COCO RLE format
            from pycocotools import mask as mask_utils
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                rle = segmentation
            else:
                # Assuming segmentation is a binary mask
                rle = mask_utils.encode(np.asfortranarray(segmentation.astype(np.uint8)))

            # Prepare the prediction dictionary
            # 准备预测字典
            pred = {
                'image_id': image_id,
                'category_id': 1,  # 假设 'person' 类的 ID 为 1
                'segmentation': {
                    'size': rle['size'],
                    'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                },
                'bbox': bbox,
                'score': score,
            }
            coco_predictions.append(pred)
        if len(masks) == 0:
            height, width = image_np.shape[:2]  # Assuming image_np is the image array in (H, W, C) format
            # Generate a temporary placeholder binary mask with the same dimensions as the image
            placeholder_mask = np.zeros((height, width), dtype=np.uint8) 
            rle = masks_utils.encode(np.asfortranarray(placeholder_mask))
            placeholder_bbox = [0, 0, 1, 1]  # Small bbox for the placeholder mask
            placeholder_score = 0.0  # Set the score low to indicate it's a placeholder
            pred = {
                'image_id': image_id,
                'category_id': 1,  # Assuming 'person' class ID is 1
                'segmentation': {
                    'size': rle['size'],
                    'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                },
                'bbox': placeholder_bbox,
                'score': placeholder_score,
            }
            coco_predictions.append(pred)
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(test_loader)} images")

    # Save predictions to a JSON file
    import json
    coco_result_save_json_path = os.path.join(output_base_dir, "result.json")
    with open(coco_result_save_json_path, 'w') as f:
        json.dump(coco_predictions, f)

    coco_dt = coco_gt.loadRes(coco_result_save_json_path)

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()
