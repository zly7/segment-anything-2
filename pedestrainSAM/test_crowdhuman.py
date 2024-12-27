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

'''
这个函数是纯纯的画图函数，所以没啥用
'''
# Define the dataset class
class CocoDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None):
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images_dir = images_dir
        self.transform = transform

        # Build image id to filename mapping
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}

        # Build image id to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        self.image_ids = list(self.image_id_to_filename.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        annotations = self.image_id_to_annotations.get(image_id, [])
        boxes = [ann['bbox'] for ann in annotations]  # COCO format: [x, y, width, height]

        # Convert boxes to [x1, y1, x2, y2]
        boxes = np.array(boxes)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        if self.transform:
            image = self.transform(image)
        image = np.array(image) 
        return image_id, image, boxes, image_filename



def visualize_predictions(image: np.ndarray, boxes, masks, iou_predictions, person_logits, save_path, image_filname):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import os

    image = Image.fromarray(image)
    font = ImageFont.load_default()

    # 创建原始图像和带注释的图像副本
    original_image = image.copy()
    annotated_image = image.copy()

    # 用于存储每个对象的颜色
    colors = []

    # 遍历每个掩码，绘制在带注释的图像上
    for idx, mask in enumerate(masks):
        mask = mask.cpu().numpy().astype(np.uint8) * 255  # 转换为0-255
        # 生成随机颜色
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        colors.append(color)

        # 创建彩色掩码
        mask_pil = Image.fromarray(mask, mode='L')
        colored_mask = Image.new('RGBA', annotated_image.size, color + (0,))
        mask_pil = mask_pil.resize(annotated_image.size, resample=Image.NEAREST)
        mask_pil = mask_pil.point(lambda p: p > 0 and 200)
        annotated_image.paste(colored_mask, (0, 0), mask_pil)

    # 创建绘图对象
    draw_original = ImageDraw.Draw(original_image)
    draw_annotated = ImageDraw.Draw(annotated_image)

    # 在两张图上绘制边框和标签
    for idx, box in enumerate(boxes):
        box = box.tolist()
        iou_score = iou_predictions[idx].item()
        person_score = person_logits[idx].item()
        label = f"IoU: {iou_score:.2f}, Person: {person_score:.2f}"
        color = colors[idx]

        # 在原始图像上绘制
        draw_original.rectangle(box, outline=color, width=2)
        draw_original.text((box[0], box[1]), label, fill='yellow', font=font)

        # 在带注释的图像上绘制
        draw_annotated.rectangle(box, outline=color, width=2)
        draw_annotated.text((box[0], box[1]), label, fill='yellow', font=font)

    # 横向拼接两张图像
    width, height = original_image.size
    total_width = width * 2
    combined_image = Image.new('RGB', (total_width, height))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(annotated_image, (width, 0))

    # 保存拼接后的图像
    save_filename = f"{image_filname}.png"
    combined_image.save(os.path.join(save_path, save_filename))




def main():
    # Parameters (modify as needed)
    images_dir = '/data2/zly/mot_data/crowdhuman/Images'  # Path to images
    annotations_file = '/data2/zly/mot_data/crowdhuman/crowdhuman_val_split.json'  # Path to annotations
    save_path = '/data2/zly/mot_data/crowdhuman/SAM2_visualizations_val'  # Path to save visualized images
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)


    # Initialize dataset and dataloader
    dataset = CocoDataset(images_dir, annotations_file, transform=None) # set_image 接受原生的PIL Image
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./train_config/train_large.yaml")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    device = 0
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model, missing_keys, unexpected_keys = build_sam2_for_self_train(
        model_config_path_for_build_sam2,
        sam2_checkpoint,
        device=torch.device('cuda', 0),
        apply_postprocessing=True,
    )
    model = PedestrainSAM2(
        model=sam2_model,  # Replace with your model instance
        config=config,     # Replace with your config
        device_index=0     # Adjust device index as needed
    )
    model.to(device)
    model.eval()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image_id, image, boxes, image_filnames in tqdm(dataloader):
            image_id = image_id[0]
            image = image[0]  # Remove batch dimension
            boxes = boxes[0]  # Remove batch dimension
            image_filnames = image_filnames[0]
            # Convert boxes to tensorsq
            boxes = boxes.clone().detach().to(dtype=torch.float32, device=device)

            # Process boxes in batches to avoid exceeding GPU memory
            num_boxes = boxes.shape[0]
            masks_list = []
            iou_predictions_list = []
            person_logits_list = []
            image_ndarray = image.numpy()
            assert image_ndarray.shape[-1] == 3
            model.set_image(image_ndarray)

            for i in range(0, num_boxes, batch_size):
                batch_boxes = boxes[i:i + batch_size]
                # Reshape boxes to (B, 2, 2)
                batch_boxes = batch_boxes[:, None, :].view(-1, 2, 2)

                masks, iou_predictions, person_logits, _ = model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    box = batch_boxes,
                    sparse_embeddings=None,
                    dense_embeddings=None,
                    predict_logit=False,
                )
                assert masks.shape[1] == 1
                masks = masks[:, 0]
                masks_list.extend(masks) # masks [b,1,h,w]
                iou_predictions_list.extend(iou_predictions)
                person_logits_list.extend(person_logits)

            # Reset the image in the model
            model.reset_image()

            # Convert lists to tensors
            masks = torch.stack(masks_list)
            iou_predictions = torch.stack(iou_predictions_list)
            person_logits = torch.stack(person_logits_list)

            # Visualize and save the results
            visualize_predictions(
                image=image_ndarray,
                boxes=boxes.cpu(),
                masks=masks,
                iou_predictions=iou_predictions,
                person_logits=person_logits,
                save_path=save_path,
                image_filname=image_filnames
            )

            print(f"Processed and saved visualization for image ID: {image_id}")

if __name__ == '__main__':
    main()
