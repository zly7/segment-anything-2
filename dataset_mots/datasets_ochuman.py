import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
from pycocotools.coco import COCO
import torchvision.transforms as T
from sam2.utils.transforms import SAM2Transforms
class OCHumanDataset(Dataset):
    def __init__(self, root_path, annotation_file, transform_use_sam2=True, img_output_size=1024):
        """
        Args:
            root_path (str): Directory with all the images.
            annotation_file (str): Path to the COCO-style JSON annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_output_size (int): Desired output image size (resized to square).
        """
        self.root_path = root_path
        self.annotation_file = annotation_file
        self.transform_use_sam2 = transform_use_sam2 
        self.img_output_size = img_output_size
        self.sam2_transform = SAM2Transforms(resolution=img_output_size, mask_threshold=0)

        # Initialize COCO api for instance annotations
        self.coco = COCO(self.annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.load_classes()

    def load_classes(self):
        # Load category mapping (e.g., person)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for cat in cats:
            self.coco_labels[len(self.classes)] = cat['id']
            self.coco_labels_inverse[cat['id']] = len(self.classes)
            self.classes[cat['name']] = len(self.classes)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image and annotations
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root_path, image_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Get annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        categories = []
        for ann in anns:
            # Check if the annotation is for the person class
            if self.coco_labels_inverse[ann['category_id']] == self.classes['person']:
                # annToMask returns a binary mask with shape (height, width)
                mask = self.coco.annToMask(ann)
                masks.append(mask)
                categories.append(ann['category_id'])

        # If no masks are found, create an empty array
        if len(masks) == 0:
            masks = np.zeros((0, image.height, image.width), dtype=np.uint8)
        else:
            # Stack masks along the first dimension (number of instances)
            masks = np.stack(masks, axis=0)

        sample = {
            'image_path': img_path,
            'image': image,
            'masks': masks,  # Shape: (N_instances, H, W)
            'categories': categories  # List of category IDs
        }

        if self.transform_use_sam2:
            sample['image']  = self.sam2_transform(sample['image'])
            resized_masks = []
            for mask in masks:
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                        (self.img_output_size, self.img_output_size), 
                                        interpolation=cv2.INTER_NEAREST)
                resized_masks.append(resized_mask)
            sample['masks'] =  np.stack(resized_masks, axis=0)
        else:
            sample = self.default_transform(sample)

        return sample

    def default_transform(self, sample):
        # Default transformations: Resize image and masks to desired output size
        image = sample['image']
        masks = sample['masks']

        # Resize image
        resize = T.Resize((self.img_output_size, self.img_output_size))
        image = resize(image)
        image = T.ToTensor()(image)

        # Resize masks
        masks = torch.from_numpy(masks).float()  # Convert masks to tensor
        masks = masks.unsqueeze(1)  # Add channel dimension: (N, 1, H, W)
        masks = T.Resize((self.img_output_size, self.img_output_size))(masks)
        masks = masks.squeeze(1)  # Remove channel dimension: (N, H, W)

        sample['image'] = image
        sample['masks'] = masks

        return sample

    def transform_sample(self, sample):
        # Apply custom transformations to image and masks
        image = sample['image']
        masks = sample['masks']

        # Convert image and masks to tensors
        image = T.ToTensor()(image)
        masks = torch.from_numpy(masks).float()  # Shape: (N, H, W)

        # Apply transformations (e.g., normalization, resizing)
        # Ensure that the same transformation is applied to both image and masks
        # Example using Compose:
        composed_transforms = T.Compose([
            T.Resize((self.img_output_size, self.img_output_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        image = composed_transforms(image)

        # Resize masks
        masks = masks.unsqueeze(1)  # Shape: (N, 1, H, W)
        masks = T.Resize((self.img_output_size, self.img_output_size))(masks)
        masks = masks.squeeze(1)  # Shape: (N, H, W)

        sample['image'] = image
        sample['masks'] = masks

        return sample
