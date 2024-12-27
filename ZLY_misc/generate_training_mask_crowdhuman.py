import argparse
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from typing import List, Dict
import cv2
from tqdm import tqdm
import yaml
from .trainer_ddp import PedestrainSAM2, build_sam2_for_self_train
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from loguru import logger
from pycocotools import mask as maskUtils
from sam2.utils.amg import (
    MaskData,
    calculate_stability_score,
)
import random

class CocoDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None,start_idx=0,end_idx=-1):
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images_dir = images_dir
        self.transform = transform

        # Build image id to filename mapping
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.image_id_to_size = {img['id']: (img['width'], img['height']) for img in self.coco_data['images']}

        # Build list of bbox entries: (image_id, image_filename, bbox, ann_id)
        self.bbox_entries = []
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            image_filename = self.image_id_to_filename[image_id]
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            ann_id = ann['id']
            self.bbox_entries.append({
                'image_id': image_id,
                'image_filename': image_filename,
                'bbox': bbox,
                'ann_id': ann_id
            })
        logger.info(f"Loaded {len(self.bbox_entries)} bbox entries")
        self.bbox_entries = self.bbox_entries[start_idx:end_idx]

    def __len__(self):
        return len(self.bbox_entries)

    def __getitem__(self, idx):
        entry = self.bbox_entries[idx]
        image_id = entry['image_id']
        image_filename = entry['image_filename']
        bbox = entry['bbox']  # [x, y, width, height]
        ann_id = entry['ann_id']

        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Convert bbox to [x1, y1, x2, y2],这里经常出现的情况是bbox超出了图片的边界,但是只超出了一点点
        x, y, w, h = bbox
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        if y2 > image.size[1] and y2 < image.size[1] + 5:
            y2 = image.size[1]
        if x2 > image.size[0] and x2 < image.size[0] + 5:
            x2 = image.size[0]
        if x1 < 0 and x1 > -3:
            x1 = 0
        if y1 < 0 and y1 > -3:
            y1 = 0
        if y2 > image.size[1] or x2 > image.size[0] or x1 < 0 or y1 < 0:
            logger.warning(f"bbox out of image boundary: {image_filename}, {bbox}, image_id {image_id}, ann_id {ann_id} image size: {image.size}")
            if y2 > image.size[1]:
                logger.warning(f"y2 ({y2}) is out of image boundary (height: {image.size[1]}), bbox: {bbox}, image_id: {image_id}")
                return None
            if x2 > image.size[0]:
                logger.warning(f"x2 ({x2}) is out of image boundary (width: {image.size[0]}), bbox: {bbox}, image_id: {image_id}")
                return None
            if x1 < 0:
                logger.warning(f"x1 ({x1}) is out of image boundary (must be >= 0), bbox: {bbox}, image_id: {image_id}")
                return None
            if y1 < 0:
                logger.warning(f"y1 ({y1}) is out of image boundary (must be >= 0), bbox: {bbox}, image_id: {image_id}")
                return None
            return None

        # Calculate cropping window according to the rules
        image_width, image_height = image.size

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_height > bbox_width:
            # Taller bbox
            crop_height = bbox_height / 0.5  # bbox occupies 50% of the cropped image height
            crop_width = crop_height  # Keep the cropped image square

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            crop_y1 = center_y - crop_height * 0.5
            crop_y2 = center_y + crop_height * 0.5

            crop_x1 = center_x - crop_width * 0.5
            crop_x2 = center_x + crop_width * 0.5
        else:
            # Wider bbox
            crop_width = bbox_width / 0.5  # bbox occupies 50% of the cropped image width
            crop_height = crop_width  # Keep the cropped image square

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            crop_x1 = center_x - crop_width * 0.5
            crop_x2 = center_x + crop_width * 0.5

            crop_y1 = center_y - crop_height * 0.5
            crop_y2 = center_y + crop_height * 0.5

        # Handle crop out of bounds
        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - image_width)
        pad_bottom = max(0, crop_y2 - image_height)

        # Adjust crop coordinates to be within image boundaries
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(image_width, crop_x2)
        crop_y2 = min(image_height, crop_y2)

        crop_width_adj = crop_x2 - crop_x1
        crop_height_adj = crop_y2 - crop_y1

        # Crop the image
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        cropped_image = image.crop(crop_box)

        # Pad the image if necessary
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_width = crop_width_adj + pad_left + pad_right
            padded_height = crop_height_adj + pad_top + pad_bottom
            new_image = Image.new('RGB', (int(padded_width), int(padded_height)), (0, 0, 0))
            new_image.paste(cropped_image, (int(pad_left), int(pad_top)))
            cropped_image = new_image

        # Resize the cropped image to 1024x1024
        cropped_image = cropped_image.resize((1024, 1024), Image.BILINEAR)

        # Adjust the bbox coordinates accordingly
        scale_x = 1024 / (crop_width_adj + pad_left + pad_right)
        scale_y = 1024 / (crop_height_adj + pad_top + pad_bottom)

        adjusted_x1 = (x1 - crop_x1 + pad_left) * scale_x
        adjusted_y1 = (y1 - crop_y1 + pad_top) * scale_y
        adjusted_x2 = (x2 - crop_x1 + pad_left) * scale_x
        adjusted_y2 = (y2 - crop_y1 + pad_top) * scale_y

        adjusted_bbox = [adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2]

        if self.transform:
            cropped_image = self.transform(cropped_image)

        # Original bbox in [x1, y1, x2, y2] in original image coordinates
        origin_bbox = [x1, y1, x2, y2]

        # Save additional data needed for inverse transformations
        meta = {
            'crop_x1': crop_x1,
            'crop_y1': crop_y1,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
            'crop_width_adj': crop_width_adj,
            'crop_height_adj': crop_height_adj,
            'image_width': image_width,
            'image_height': image_height,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'ann_id': ann_id,
        }
        if scale_x > 50 or scale_y > 50: # 原图不到20个像素
            logger.warning(f"scale_x or scale_y is too large: {scale_x}, {scale_y}, image_filename: {image_filename}, ann_id: {ann_id}, bbox: {bbox}")
            return None

        return image_id, image_filename, origin_bbox, cropped_image, adjusted_bbox, meta

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    image_ids = []
    image_filenames = []
    origin_bboxes = []
    cropped_images = []
    adjusted_bboxes = []
    metas = []

    for item in batch:
        image_id, image_filename, origin_bbox, cropped_image, adjusted_bbox, meta = item
        image_ids.append(image_id)
        image_filenames.append(image_filename)
        origin_bboxes.append(origin_bbox)
        cropped_images.append(cropped_image)
        adjusted_bboxes.append(adjusted_bbox)
        metas.append(meta)

    return image_ids, image_filenames, origin_bboxes, cropped_images, adjusted_bboxes, metas

def visualize_prediction_single(image, mask, bbox, iou_prediction):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw bbox
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    # Draw mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image = mask_image.resize(image.size, resample=Image.NEAREST)
    mask_image.putalpha(128)  # Semi-transparent
    image.paste(mask_image, (0, 0), mask_image)

    # Draw IoU score
    draw.text((x1, y1), f'IoU: {iou_prediction:.2f}', fill='yellow', font=font)

    return image

def mask_to_coco_segmentation(mask):
    # mask: numpy array of shape (H, W), binary mask
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')  # counts need to be decoded to string
    return rle

def main():
    parser = argparse.ArgumentParser(description="Example of argparse usage")
    parser.add_argument('--start_index', type=int, required=False,default=0)
    parser.add_argument('--end_index', type=int, required=False,default=-1)
    args = parser.parse_args()
    start_idx = args.start_index
    end_idx = args.end_index
    images_dir = '/data2/zly/mot_data/crowdhuman/Images'  # Path to images
    # annotations_file = '/data2/zly/mot_data/crowdhuman/crowdhuman_train_sample_10.json'
    # annotations_file = "/data2/zly/mot_data/crowdhuman/crowdhuman_train_split.json"
    annotations_file = "/data2/zly/mot_data/crowdhuman/crowdhuman_val_split.json"
    save_path = '/data2/zly/mot_data/crowdhuman/SAM2_visualizations_val_crop_image'
    batch_size = 32

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = CocoDataset(images_dir, annotations_file, transform=None,start_idx=start_idx,end_idx=end_idx)  # set_image accepts raw PIL Image
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize variables for COCO JSON
    images_coco = []
    annotations_coco = []
    existing_image_ids = set()

    sam2_checkpoint = "/data3/zly/multi_ob/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor_model = SAM2ImagePredictor(sam2_model)
    vis_probabilities = 0.05
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image_ids, image_filenames, origin_bboxes, cropped_images, adjusted_bboxes, metas in tqdm(dataloader):
            # Convert images to numpy arrays
            image_ndarrays = [np.array(img) for img in cropped_images]

            predictor_model.set_image_batch(image_ndarrays)

            box_batch = [np.array(bbox, dtype=np.float32).reshape(1, 4) for bbox in adjusted_bboxes]  # Each is (1, 4)

            # Predict masks
            masks_first, iou_predictions, all_low_res_masks = predictor_model.predict_batch(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=box_batch,
                mask_input_batch=None,
                multimask_output=False,
                return_logits=False,
                normalize_coords=True
            )
            logits_masks, iou_predictions, all_low_res_masks = predictor_model.predict_batch(
                point_coords_batch=None,
                point_labels_batch=None,
                box_batch=box_batch,
                mask_input_batch=all_low_res_masks,
                multimask_output=False,
                return_logits=True,
                normalize_coords=True
            )

            # Process each item in the batch
            for idx in range(len(image_ids)):
                image_id = image_ids[idx]
                image_filename = image_filenames[idx]
                origin_bbox = origin_bboxes[idx]  # [x1, y1, x2, y2] in original image coordinates
                cropped_image = image_ndarrays[idx]  # numpy array
                adjusted_bbox = adjusted_bboxes[idx]  # [x1, y1, x2, y2] in 1024x1024 image coordinates
                logit_mask = logits_masks[idx][0]  # masks[idx] is an array; take the first element
                mask = (logit_mask > 0.0).astype(np.uint8)  # Convert to binary mask
                logit_mask_tensor = torch.tensor(logit_mask, dtype=torch.float32)
                iou_prediction = iou_predictions[idx][0]  # similarly, take the first element
                meta = metas[idx]

                # Get original image size
                image_width = meta['image_width']
                image_height = meta['image_height']

                # Transform the mask back to original image coordinates
                full_mask = np.zeros((image_height, image_width), dtype=np.uint8)

                # Compute the size of the mask in original image coordinates
                mask_width = int(meta['crop_width_adj'] + meta['pad_left'] + meta['pad_right'])
                mask_height = int(meta['crop_height_adj'] + meta['pad_top'] + meta['pad_bottom'])

                # Resize the mask to the size of the cropped area in the original image
                resized_mask = cv2.resize(mask.astype(np.uint8), (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)

                # Compute the position where to place the mask in the original image
                x0 = int(meta['crop_x1'] - meta['pad_left'])
                y0 = int(meta['crop_y1'] - meta['pad_top'])

                # Handle negative starting indices,这里是在选择原图的坐标
                x_start = max(0, x0)
                y_start = max(0, y0)

                x_end = min(image_width, x0 + mask_width)
                y_end = min(image_height, y0 + mask_height)

                # Compute corresponding indices in the resized mask
                mask_x_start = x_start - x0
                mask_y_start = y_start - y0
                mask_x_end = mask_x_start + (x_end - x_start)
                mask_y_end = mask_y_start + (y_end - y_start)

                # Place the resized mask into the full mask
                full_mask[y_start:y_end, x_start:x_end] = resized_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

                # Adjust the bbox back to original image coordinates
                inv_scale_x = (meta['crop_width_adj'] + meta['pad_left'] + meta['pad_right']) / 1024
                inv_scale_y = (meta['crop_height_adj'] + meta['pad_top'] + meta['pad_bottom']) / 1024

                x1_orig = adjusted_bbox[0] * inv_scale_x + (meta['crop_x1'] - meta['pad_left'])
                y1_orig = adjusted_bbox[1] * inv_scale_y + (meta['crop_y1'] - meta['pad_top'])
                x2_orig = adjusted_bbox[2] * inv_scale_x + (meta['crop_x1'] - meta['pad_left'])
                y2_orig = adjusted_bbox[3] * inv_scale_y + (meta['crop_y1'] - meta['pad_top'])
                # Calculate area
                area = np.sum(full_mask > 0)

                # Convert the mask to RLE
                segmentation = mask_to_coco_segmentation(full_mask)

                # Convert bbox to COCO format [x, y, width, height]
                bbox_width = x2_orig - x1_orig
                bbox_height = y2_orig - y1_orig
                stability_score = calculate_stability_score(logit_mask_tensor, mask_threshold = 0.0,threshold_offset=1.0)

                # Prepare the annotation
                annotations_coco.append({
                    'id': meta['ann_id'],  # Convert to Python int
                    'image_id': int(image_id),  # Convert to Python int
                    'category_id': 1,  # Assuming 'person' category id is 1, and it's already int
                    'bbox': [float(x1_orig), float(y1_orig), float(bbox_width), float(bbox_height)],  # Convert to float if necessary
                    'area': int(area),  # Convert to Python int
                    'segmentation': segmentation,  # This should already be fine as RLE is encoded as strings
                    'iscrowd': 0,  # Already int
                    'confidence': float(iou_prediction),
                    'stability_score': float(stability_score)
                })

                if random.random() < vis_probabilities:
                    # Save visualization if needed
                    save_dir = os.path.join(save_path, image_filename)
                    os.makedirs(save_dir, exist_ok=True)
                    vis_image = visualize_prediction_single(
                        image=cropped_image,
                        mask=mask,
                        bbox=adjusted_bbox,
                        iou_prediction=iou_prediction
                    )

                    vis_image_filename = f"{image_filename}_{image_id}_{meta['ann_id']}_vis.png"
                    vis_image_filepath = os.path.join(save_dir, vis_image_filename)
                    vis_image.save(vis_image_filepath)

                # Add the image to images_coco if not already added
                if image_id not in existing_image_ids:
                    images_coco.append({
                        'id': image_id,
                        'file_name': image_filename,
                        'width': image_width,
                        'height': image_height
                    })
                    existing_image_ids.add(image_id)

    # Save COCO JSON
    coco_output = {
        'images': images_coco,
        'annotations': annotations_coco,
        'categories': [{'id': 1, 'name': 'person'}]
    }

    with open(os.path.join(save_path, 'annotations.json'), 'w') as f:
        json.dump(coco_output, f)

if __name__ == '__main__':
    logger.add("logs/generate_training_mask_crowdhuman/log_file.log", rotation="50 MB", encoding="utf-8", level="INFO")
    main()
