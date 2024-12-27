import argparse
import json
import os
import random
import shutil
import time
from anyio import current_time
import numpy as np
import torchvision
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, ChainedScheduler
from tensorboardX import SummaryWriter
from loguru import logger
from sam2.build_sam import build_sam2, build_sam2_for_self_train
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from typing import Optional
from dataset_zly import MOTSDataset,COCOPersonDataset, CombinedDataset
from PIL import Image, ImageDraw, ImageFont
from training.loss_fns import sigmoid_focal_loss, dice_loss, iou_loss
from typing import Union
from torchvision import transforms as torch_transforms  # 导入 torchvision 的 transforms 模块
from PIL import Image
import torchvision.transforms as transforms
from trainer_ddp import PedestrainSAM2,get_dataset
from loguru import logger
from sam2.sam2_image_predictor import SAM2ImagePredictor
class GenerateMaskDataset:
    def __init__(self, dataset: CombinedDataset, pedestrainSAM: PedestrainSAM2, batch_size: int = 16, num_workers: int = 4, predictor: Optional[SAM2ImagePredictor] = None):
        self.father_dataset = dataset
        self.pedestrainSAM = pedestrainSAM
        self.device = self.pedestrainSAM.device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predictor = predictor

        # Define transformations if needed
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def create_dataloader(self):
        """
        Creates a DataLoader for positive or negative samples.

        Args:
            positive (bool): If True, loads positive samples. Otherwise, negative samples.
            num_negative_samples (int, optional): Number of negative samples to load. Required if positive is False.

        Returns:
            DataLoader: Configured DataLoader.
        """

        dataloader = DataLoader(
            self.father_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return dataloader

    def generate_and_save_positive_samples(self, output_dir, json_annotations, starting_image_id=0):
        """
        Generates and saves all positive samples using batching.

        Args:
            output_dir (str): Directory where positive samples will be saved.
            json_annotations (dict): Dictionary to store COCO annotations.
            starting_image_id (int): Starting ID for images to avoid ID conflicts.

        Returns:
            int: The next available image ID after processing.
        """
        os.makedirs(output_dir, exist_ok=True)
        dataloader = self.create_dataloader()
        image_id = starting_image_id
        for batch in tqdm(dataloader, desc="Processing positive samples"):
            # Assume that each batch is a dictionary with 'image' and 'mask' keys
            images = batch['image'].to(self.device)  # Shape: (B, 3, H, W)
            masks = batch['mask'].to(self.device)    # Shape: (B, H, W)
            masks = masks.unsqueeze(1)               # Shape: (B, 1, H, W)

            extracted_images = self.pedestrainSAM._extract_and_resize_images(
                images, masks, [True] * images.size(0), [True] * images.size(0), whether_vis=False, whether_tolerate_no_mask=False
            )  # List of PIL Images or Tensors
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            for img_tensor in extracted_images:
                img_tensor = img_tensor * std + mean
                img_tensor = img_tensor.clamp(0, 1).cpu()

                # Convert to PIL Image
                img_pil = transforms.ToPILImage()(img_tensor)

                # Save the image
                filename = f"positive_{image_id}.jpg"
                filepath = os.path.join(output_dir, filename)
                img_pil.save(filepath)

                # Update JSON annotations
                json_annotations['images'].append({
                    'id': image_id,
                    'file_name': filename,
                    'width': img_pil.width,
                    'height': img_pil.height
                })

                json_annotations['annotations'].append({
                    'id': image_id,
                    'image_id': image_id,
                    'category_id': 1  # Assuming category_id 1 is 'person'
                })

                image_id += 1

        print(f"Saved {image_id - starting_image_id} positive samples to {output_dir}")
        return image_id

    def generate_and_save_negative_samples(self, output_dir, json_annotations, starting_image_id=0):
        """
        Generates and saves negative samples using SAM with batching.

        Args:
            output_dir (str): Directory where negative samples will be saved.
            json_annotations (dict): Dictionary to store COCO annotations.
            num_samples (int): Number of negative samples to generate.
            starting_image_id (int): Starting ID for images to avoid ID conflicts.

        Returns:
            int: The next available image ID after processing.
        """
        os.makedirs(output_dir, exist_ok=True)
        dataloader = self.create_dataloader()
        image_id = starting_image_id

        for batch in tqdm(dataloader, desc="Processing negative samples"):
            images = batch['image'].to(self.device)         # Shape: (B, 3, H, W)
            click_points = batch['click_point'].to(self.device)  # Shape: (B, 2)
            point_labels = batch['point_label'].to(self.device)  # Shape: (B,)
            person_label = batch["is_person"].to(self.device)  # Shape: (B,)
            not_person_indices = torch.where(person_label == 0)[0]  # Indices of images with person labels
            if len(not_person_indices) != len(images):
                logger.warning(f"Batch contains {len(not_person_indices)} images with person labels out of {len(images)}")
            images = images[not_person_indices]
            click_points = click_points[not_person_indices]
            point_labels = point_labels[not_person_indices]
            
            # Generate masks using SAM
            with torch.no_grad():
                with torch.amp.autocast('cuda',dtype=torch.float16): # 这里想加入original SAM也太难修改了，所以直接在后面我的API改成true
                    prd_masks, person_logits, iou_predictions = self.pedestrainSAM.forward(
                        images, point_coords=click_points, point_labels=point_labels
                    )  # prd_masks shape: (B, num_masks, H, W)

            # Assuming you take the first mask for each image
            masks = prd_masks > 0.0 
            # Process the batch with SAM
            extracted_images = self.pedestrainSAM._extract_and_resize_images(
                images, masks, [False] * images.size(0), [False] * images.size(0), whether_vis=False
            )   
            # De-normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            for img_tensor in extracted_images:
                img_tensor = img_tensor * std + mean
                img_tensor = img_tensor.clamp(0, 1).cpu()

                # Convert to PIL Image
                img_pil = transforms.ToPILImage()(img_tensor)

                # Save the image
                filename = f"negative_{image_id}.jpg"
                filepath = os.path.join(output_dir, filename)
                img_pil.save(filepath)

                # Update JSON annotations
                json_annotations['images'].append({
                    'id': image_id,
                    'file_name': filename,
                    'width': img_pil.width,
                    'height': img_pil.height
                })

                json_annotations['annotations'].append({
                    'id': image_id,
                    'image_id': image_id,
                    'category_id': 0  # Assuming category_id 0 is 'non-person' or 'background'
                })

                image_id += 1

        print(f"Saved {image_id - starting_image_id} negative samples to {output_dir}")
        return image_id

    def create_coco_annotations(self):
        """
        Initializes the COCO annotations structure.

        Returns:
            dict: The initialized annotations dictionary.
        """
        annotations = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'background'},
                {'id': 1, 'name': 'person'}
            ]
        }
        return annotations


def main():
    logger_file_path = "/data3/zly/segpersonclass/generate_classification_dataset.log"
    logger.add(logger_file_path, rotation="100 MB", backtrace=True, diagnose=True)
    config_path = "./train_config/train_large.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize device
    device_index = config.get("device_index", 1)
    device = torch.device('cuda', device_index) if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Build dataset
    train_dataset = get_dataset(config, dataset_type="train",whether_augument=False)

    # Build model
    model_config_path = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model, missing_keys, unexpected_keys = build_sam2_for_self_train(
        model_config_path,
        sam2_checkpoint,
        device=device,
        apply_postprocessing=True,
    )
    pedestrain_sam_model = PedestrainSAM2(
        model=sam2_model,
        config=config,
        device_index=device.index if device.type == 'cuda' else -1
    )
    # predictor = SAM2ImagePredictor(sam2_model) # 之后思考了下还是从这个基础去想比较对
    pedestrain_sam_model.load_classifier(sam2_checkpoint)

    # Initialize mask generator with appropriate batch size and number of workers
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 32)
    generator = GenerateMaskDataset(train_dataset, pedestrain_sam_model, batch_size=batch_size, num_workers=num_workers)


    # use_positive_or_negative = "positive"
    use_positive_or_negative = "negative"

    # Define output directories and annotation file paths
    output_dir = "/data3/zly/segpersonclass/images"
    # output_dir = "/data3/zly/segpersonclass/test_for_negative"
    logger.warning(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    annotations_file = "/data3/zly/segpersonclass/dataset_annotations.json"

    if use_positive_or_negative == "positive":
        # Create COCO annotations dictionary
        json_annotations = generator.create_coco_annotations()
        starting_image_id = 0

        next_image_id = generator.generate_and_save_positive_samples(
            output_dir=output_dir,
            json_annotations=json_annotations,
            starting_image_id=starting_image_id
        )
        # Save JSON annotations to file
        with open(annotations_file, 'w') as f:
            json.dump(json_annotations, f)

        print(f"COCO annotations saved to {annotations_file}")
        print(f"Samples saved to {output_dir}")
    else:
        # Load existing annotations to append negative samples
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                json_annotations = json.load(f)
            # Determine the next image ID
            existing_ids = [img['id'] for img in json_annotations['images']]
            next_image_id = max(existing_ids) + 1 if existing_ids else 0
            logger.success("use the positve coco file as the base!!!")
        else:
            # If annotations file does not exist, initialize it
            json_annotations = generator.create_coco_annotations()
            next_image_id = 0

        num_positive_samples = len(train_dataset)
        generator.generate_and_save_negative_samples(
            output_dir=output_dir,
            json_annotations=json_annotations,
            starting_image_id=next_image_id
        )
        # Save updated JSON annotations to file
        with open(annotations_file, 'w') as f:
            json.dump(json_annotations, f)

        print(f"COCO annotations updated and saved to {annotations_file}")
        print(f"Negative samples saved to {output_dir}")


if __name__ == "__main__":
    main()


