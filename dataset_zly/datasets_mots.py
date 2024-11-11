import math
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import PIL.Image as Image
import pycocotools.mask as rletools
import sys
import cv2
from .mots_io import load_txt
from loguru import logger
from sam2.utils.transforms import SAM2Transforms
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
from scipy.ndimage import binary_erosion, binary_dilation
class MOTSDataset(Dataset):
    def __init__(self, root_path, use_SAM2_transform=True, augment=False, img_output_size=1024, max_length_for_validate=-1, enable_negative_sample=True):
        self.root_path = root_path
        self._pre_data = {}
        self.data = []
        self.all_images = []  # For negative samples
        self.objects_per_image = {}  # For negative samples
        self.no_person_mask_map = {}
        self.augment = augment
        self.img_output_size = img_output_size
        self.max_length_for_validate = max_length_for_validate
        self.whether_use_sam2_transform = use_SAM2_transform
        self.sam2_transform = SAM2Transforms(resolution=img_output_size, mask_threshold=0)
        self.enable_negative_sample = enable_negative_sample
        self.load_data()
        if self.augment:
            self.transform_pipeline = ComposeAPI([
                RandomHorizontalFlip(p=0.5, consistent_transform=True),
                RandomAffine(degrees=25, shear=20, image_interpolation='bilinear', consistent_transform=True),
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
        '''
        This function loads the dataset list.
        '''
        for seq in os.listdir(self.root_path):
            seq_path = os.path.join(self.root_path, seq)
            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            img_folder_path = os.path.join(seq_path, 'img1')
            self._pre_data[gt_path] = load_txt(gt_path)
            for frame, segmented_objects in self._pre_data[gt_path].items():
                img_path = os.path.join(img_folder_path, f"{frame:06d}.jpg")
                if not os.path.exists(img_path):
                    continue  # Skip if image does not exist
                # Collect all images and objects for negative samples
                if img_path not in self.objects_per_image:
                    self.objects_per_image[img_path] = []
                    self.all_images.append(img_path)
                for obj in segmented_objects:
                    if obj.class_id == 2 or obj.class_id == 10:  # 反正只需要把人的id筛选出来就可以了
                        self.objects_per_image[img_path].extend(segmented_objects)
                # Only include class_id == 2 for positive samples
                for obj in segmented_objects:
                    if obj.class_id == 2:
                        self.data.append((img_path, obj))
            logger.success(f"Successfully loaded {seq_path}")
        if self.max_length_for_validate != -1:
            random.shuffle(self.data)
            self.data = self.data[:self.max_length_for_validate]

    def __len__(self):
        # Return the length based on whether negative samples are enabled
        if self.enable_negative_sample:
            # return max(len(self.data), len(self.all_images)), 这里理论上肯定是self.data大
            return len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            if self.enable_negative_sample and (int(time.time() * 1000) % 2) == 0:
                sample =  self.get_negative_sample(idx)
            else:
                sample =  self.get_positive_sample(idx)
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
        img_path, obj = self.data[idx]
        image = Image.open(img_path)
        image = np.array(image)  # H,W,C

        # Decode the binary mask
        binary_mask = rletools.decode(obj.mask)
        eroded_mask = binary_erosion(binary_mask, structure=np.ones((3, 3)))
        ys, xs = np.where(eroded_mask)
        if len(ys) == 0:
            # If no non-person area, fall back to a negative sample
            return self.get_negative_sample(idx)
        random_idx = random.randint(0, len(ys) - 1)
        click_point = (xs[random_idx], ys[random_idx])

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': binary_mask,
            'click_point': np.array(click_point).reshape(1, 2),
            'is_person': torch.tensor([1]).float(),  # Positive sample
            'track_id':obj.track_id,
            'point_label': torch.tensor([1.0]), 
        }
        sample = self.process_sample(sample)
        return sample
    
    def get_negative_sample(self, idx):
        # Negative sample
        img_path = self.all_images[idx % len(self.all_images)]
        image = Image.open(img_path)
        image = np.array(image)  # H,W,C

        # Get all masks for class_id == 2 and class_id == 10 (persons to exclude)
        masks_to_exclude = []
        objects = self.objects_per_image[img_path]
        for obj in objects:
            if obj.class_id == 2 or obj.class_id == 10:
                mask = rletools.decode(obj.mask)
                masks_to_exclude.append(mask)
                
        if masks_to_exclude:
            combined_exclude_mask = np.any(masks_to_exclude, axis=0)
        else:
            combined_exclude_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        # Create the non-person mask (areas not labeled as person)
        non_person_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        combined_exclude_mask_dilation = binary_dilation(combined_exclude_mask, structure=np.ones((3, 3)))
        ys, xs = np.where(~combined_exclude_mask_dilation)
        if len(ys) == 0:
            non_person_indices = np.where(~combined_exclude_mask)
            if non_person_indices[0].size == 0:
                return self.get_positive_sample(idx)
        random_idx = random.randint(0, len(ys) - 1)
        click_point = (math.floor(xs[random_idx]), math.floor(ys[random_idx]))

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': non_person_mask.astype(np.uint8),
            'click_point': np.array(click_point).reshape(1, 2),
            'is_person': torch.tensor([0]).float(),  # Negative sample
            'track_id':-1,
            'point_label': torch.tensor([1.0]), 
        }
        sample = self.process_sample(sample)
        return sample
    
    def process_sample(self, sample):
        sample = self.transform_pipeline(sample)
        return sample

