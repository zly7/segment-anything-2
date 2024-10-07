import os
import random
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
            return max(len(self.data), len(self.all_images))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.enable_negative_sample and random.random() < 0.5:
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

            # Combine masks to exclude
            if masks_to_exclude:
                combined_exclude_mask = np.any(masks_to_exclude, axis=0)
            else:
                combined_exclude_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

            # Create the non-person mask (areas not labeled as person)
            non_person_mask = ~combined_exclude_mask

            ys, xs = np.where(non_person_mask)
            if len(ys) == 0:
                # If no non-person area, fall back to a positive sample
                return self.get_positive_sample(idx)
            random_idx = random.randint(0, len(ys) - 1)
            click_point = (xs[random_idx], ys[random_idx])

            sample = {
                'image_path': img_path,
                'image': image,
                'mask': non_person_mask.astype(np.uint8),
                'click_point': click_point,
                'is_person': torch.tensor([0]).float()  # Negative sample
            }
        else:
            # Positive sample
            return self.get_positive_sample(idx)

        sample = self.process_sample(sample)
        return sample

    def get_positive_sample(self, idx):
        img_path, obj = self.data[idx % len(self.data)]
        image = Image.open(img_path)
        image = np.array(image)  # H,W,C

        # Decode the binary mask
        binary_mask = rletools.decode(obj.mask)

        ys, xs = np.where(binary_mask)
        if len(ys) == 0:
            raise Exception(f"No pixels found in mask for idx {idx}")
        random_idx = random.randint(0, len(ys) - 1)
        click_point = (xs[random_idx], ys[random_idx])

        sample = {
            'image_path': img_path,
            'image': image,
            'mask': binary_mask,
            'click_point': click_point,
            'is_person': torch.tensor([1]).float()  # Positive sample
        }
        sample = self.process_sample(sample)
        return sample

    def process_sample(self, sample):
        if self.augment:
            sample = self.transform(sample)
        else:
            # Resize to output size
            if self.whether_use_sam2_transform:
                sample['image'] = self.sam2_transform(sample['image'])
            else:
                sample['image'] = cv2.resize(sample['image'], (self.img_output_size, self.img_output_size), interpolation=cv2.INTER_LINEAR)
                sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1).float()  # (C, H, W)
            sample['mask'] = cv2.resize(sample['mask'].astype(np.uint8), (self.img_output_size, self.img_output_size), interpolation=cv2.INTER_NEAREST)
            # Adjust click_point accordingly
            h_orig, w_orig = sample['mask'].shape[:2]
            scale_x = self.img_output_size / w_orig
            scale_y = self.img_output_size / h_orig
            sample['click_point'] = (sample['click_point'][0] * scale_x, sample['click_point'][1] * scale_y)

        # Convert to tensors
        sample['mask'] = torch.from_numpy(sample['mask']).unsqueeze(0).float()  # (1, H, W)
        sample['click_point'] = torch.tensor(sample['click_point']).unsqueeze(0).float()  # (1,2)
        sample['point_label'] = torch.tensor([1]).float()  # (1,)
        return sample

    def transform(self, sample):
        # [Your existing augmentation code can be used here]
        # For brevity, the augmentation code is omitted, but you can include it as in your original script.
        pass
