import math
import os
import random
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pycocotools.mask as rletools
import cv2
from loguru import logger
from sam2.utils.transforms import SAM2Transforms
from scipy.ndimage import binary_erosion, binary_dilation
from .transform_zly import (
    ComposeAPI,
    RandomHorizontalFlip,
    RandomResizeAPI,
    Pad,
    ToTensorAPI,
    NormalizeAPI,
    RandomGrayscale,
    ColorJitter,
    RandomAffine
)
class COCOPersonDataset(Dataset):
    def __init__(self, annotation_file, images_dir, use_SAM2_transform=True, augment=False,
                 img_output_size=1024, max_length_for_validate=-1, enable_negative_sample=True):
        """
        Initializes the COCO Person Dataset.

        Args:
            annotation_file (str): Path to the COCO JSON annotations file.
            images_dir (str): Directory containing the images.
            use_SAM2_transform (bool): Whether to use SAM2 transformations.
            augment (bool): Whether to apply data augmentation.
            img_output_size (int): Desired output image size (both height and width).
            max_length_for_validate (int): Maximum number of samples for validation. -1 means no limit.
            enable_negative_sample (bool): Whether to include negative samples.
        """
        self.annotation_file = annotation_file
        self.images_dir = images_dir
        self.use_SAM2_transform = use_SAM2_transform
        self.augment = augment
        self.img_output_size = img_output_size
        self.max_length_for_validate = max_length_for_validate
        self.enable_negative_sample = enable_negative_sample

        self.sam2_transform = SAM2Transforms(resolution=img_output_size, mask_threshold=0)
        self.data = []
        self.all_images = []  # For negative samples
        self.objects_per_image = {}  # For negative samples

        self.load_data()
        if self.augment:
            self.transform_pipeline = ComposeAPI([
                RandomHorizontalFlip(p=0.5, consistent_transform=True),
                RandomAffine(p=0.5, degrees=25, shear=20, image_interpolation='bilinear', consistent_transform=True),
                RandomResizeAPI(sizes=[img_output_size], square=True, consistent_transform=True),
                ColorJitter(brightness=0.1, contrast=0.03, saturation=0.03, hue=None, consistent_transform=True),
                RandomGrayscale(p=0.05, consistent_transform=True),
                ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=None, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_pipeline = ComposeAPI([
                RandomResizeAPI(sizes=[img_output_size], square=True, consistent_transform=True),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def load_data(self):
        """
        Loads and processes the COCO annotations to prepare the dataset.
        """
        with open(self.annotation_file, 'r') as f:
            coco = json.load(f)

        images_info = {image['id']: image for image in coco['images']}
        annotations = coco['annotations']
        categories = {category['id']: category['name'] for category in coco['categories']}

        # Assuming 'person' category is labeled as 'person' in categories
        person_category_ids = [cat_id for cat_id, name in categories.items() if name.lower() == 'person']
        if not person_category_ids:
            raise ValueError("No 'person' category found in the annotation file.")

        # Mapping from image_id to its annotations
        img_id_to_annots = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_id_to_annots:
                img_id_to_annots[img_id] = []
            if ann['category_id'] in person_category_ids:
                img_id_to_annots[img_id].append(ann)

        # Process each image
        for img_id, image in images_info.items():
            img_filename = image['file_name']
            img_path = os.path.join(self.images_dir, img_filename)
            if not os.path.exists(img_path):
                logger.warning(f"Image {img_path} does not exist. Skipping.")
                continue

            if img_id in img_id_to_annots:
                # Positive samples
                for ann in img_id_to_annots[img_id]:
                    self.data.append((img_path, ann))
                # For negative sampling
                self.objects_per_image[img_path] = img_id_to_annots[img_id]
                self.all_images.append(img_path)
            else:
                # Images without person annotations can be used as negative samples
                self.all_images.append(img_path)

        logger.success(f"Successfully loaded {len(self.data)} positive samples from {len(images_info)} images.")

        if self.max_length_for_validate is not None and self.max_length_for_validate > 0:
            random.shuffle(self.data)
            self.data = self.data[:self.max_length_for_validate]

    def __len__(self):
        """
        Returns the total number of samples, considering whether negative samples are enabled.
        """
        if self.enable_negative_sample:
            # The dataset length is the number of positive samples
            return len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing image, mask, image_path, click_point, etc.
        """
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            if self.enable_negative_sample and (int(time.time() * 1000) % 2) == 0:
                # Negative sample
                sample = self.get_negative_sample(idx)
            else:
                # Positive sample
                sample = self.get_positive_sample(idx)
            h, w = self.img_output_size, self.img_output_size
            click_point = sample['click_point'].squeeze(0).numpy()  # (2,)
            x, y = click_point
            if 0 <= x < w and 0 <= y < h:
                return sample
            else:
                if attempt > 3:
                    logger.warning(f"Attempt {attempt}: click_point ({x}, {y}) out of image bounds ({w}, {h}). Resampling.")
        logger.error(f"Failed to obtain a valid sample after {max_retries} attempts for index {idx}.")
        return sample

    def get_positive_sample(self, idx):
        """
        Retrieves a positive sample (containing a person).

        Args:
            idx (int): Index of the positive sample.

        Returns:
            dict: A dictionary containing image, mask, image_path, click_point, etc.
        """
        img_path, ann = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)  # H, W, C

        # Decode the binary mask
        binary_mask = self.decode_mask(ann)
        eroded_mask = binary_erosion(binary_mask, structure=np.ones((3, 3)))
        ys, xs = np.where(eroded_mask)
        if len(ys) == 0:
            logger.warning(f"No pixels found in mask for index {idx} in image {img_path}. the ann is {ann}")
            return self.get_negative_sample(idx)
        random_idx = random.randint(0, len(ys) - 1)
        click_point = (xs[random_idx], ys[random_idx])

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': binary_mask,
            'click_point': np.array(click_point).reshape(1, 2),
            'is_person': torch.tensor([1.0]),  # Positive sample
            'track_id': int(ann["id"]),  # Use -1 if track_id not present
            'point_label': torch.tensor([1.0]), 
        }

        sample = self.process_sample(sample)
        return sample

    def get_negative_sample(self, idx):
        """
        Retrieves a negative sample (no person present).

        Args:
            idx (int): Index of the negative sample.

        Returns:
            dict: A dictionary containing image, mask, image_path, click_point, etc.
        """
        img_path = self.all_images[idx % len(self.all_images)]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)  # H, W, C

        # Get all masks for persons to exclude
        masks_to_exclude = []
        if img_path in self.objects_per_image:
            for ann in self.objects_per_image[img_path]:
                mask = self.decode_mask(ann)
                masks_to_exclude.append(mask)

        # Combine masks to exclude
        if masks_to_exclude:
            combined_exclude_mask = np.any(masks_to_exclude, axis=0)
        else:
            combined_exclude_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        # Identify non-person areas
        combined_exclude_mask_dilation = binary_dilation(combined_exclude_mask, structure=np.ones((3, 3)))
        non_person_indices = np.where(~combined_exclude_mask_dilation)
        if non_person_indices[0].size == 0:
            logger.warning(f"No non-person pixels found in image {img_path}. Using no dilation")
            non_person_indices = np.where(~combined_exclude_mask)
            if non_person_indices[0].size == 0:
                return self.get_positive_sample(idx)
        
        random_idx = random.randint(0, len(non_person_indices[0]) - 1)
        click_point = (math.floor(non_person_indices[1][random_idx]), math.floor(non_person_indices[0][random_idx]))

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),  # Background mask
            'click_point': np.array(click_point).reshape(1, 2),
            'is_person': torch.tensor([0.0]),  # Negative sample
            'track_id': -1,
            'point_label': torch.tensor([1.0]), 
        }

        sample = self.process_sample(sample)
        return sample

    def decode_mask(self, ann):
        """
        Decodes the mask from an annotation.

        Args:
            ann (dict): Annotation dictionary from COCO.

        Returns:
            np.ndarray: Binary mask of shape (H, W), where H and W are image dimensions.
        """
        if 'segmentation' not in ann:
            raise KeyError("Annotation does not contain 'segmentation' field.")

        if isinstance(ann['segmentation'], list):
            # Polygon format
            rles = rletools.frPyObjects(ann['segmentation'], ann['height'], ann['width'])
            binary_mask = rletools.decode(rles)
        elif isinstance(ann['segmentation'], dict):
            # RLE format
            binary_mask = rletools.decode(ann['segmentation'])
        else:
            raise ValueError("Unknown segmentation format.")

        if binary_mask.ndim == 3:
            # If multiple objects are encoded, take the first channel
            binary_mask = binary_mask[:, :, 0]
        return binary_mask.astype(bool)
    
    def process_sample(self, sample):
        sample = self.transform_pipeline(sample)
        return sample
