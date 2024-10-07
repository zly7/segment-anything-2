
# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import shutil
import glob
from sam2.build_sam import build_sam2
from typing import List, Optional, Tuple
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger
import yaml
from PIL.Image import Image

class PedestrainSAM2():
    def __init__(
        self,
        model:SAM2Base,
        config,
        mask_threshold = 0.0,
        max_hole_area = 0.0,
        max_sprinkle_area = 0.0,

    ):
        super().__init__()
        self.sam2_model = model
        self._transforms = SAM2Transforms(
            resolution=self.sam2_model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.config = config
        self.device = config["train"]["device"]
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image,point_coords,point_labels):
        """
        image tensor: (B, 3, H, W)
        # box: (B, 2, 2)
        points_coords: (B, 1~3,2)
        point_labels:(B,)
        """
        assert image.size(1) == 3, "Input image must have 3 channels."
        self._orig_hw = [(image.size(2), image.size(3))]
        image = self._transforms.transform_tensor(image)
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, None, None, normalize_coords=True
        )
        if point_coords is not None:
            concat_points = (unnorm_coords, labels)
        else:
            concat_points = None
        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
        )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features
    
    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box
    
    def infer():
        pass




class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        smooth = 1e-5
        preds = torch.sigmoid(logits)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss


class Trainer:
    def __init__(self, PedestrainModel:PedestrainSAM2, train_loader, val_loader, config):
        # 从 YAML 文件中加载配置
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PedestrainModel.sam2_model.to(self.device)
        self.pedestrain_sam2 = PedestrainModel
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 定义损失函数，使用 BCEWithLogitsLoss 适用于二分类分割任务
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = BCEDiceLoss()

        # 定义优化器，使用 AdamW
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=float(config["train"]['learning_rate'])
        )

        self.num_epochs = config["train"]['num_epochs']
        self.save_path = config["train"]['save_path']

        # 初始化日志记录
        self.writer = SummaryWriter(log_dir=config["log"]['tensorboard_log_dir'])
        logger.add(config["log"]['loguru_log_file'])
        logger.success("------------------------------------Trainer initialized.---------------------------------")
        self.log_interval = config["log"]["log_interval"]
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                # 将数据移动到设备上
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                # not (B, 3, H, W) is (B, H, W, 3) because Totensor() is used after transform.最后发现会自动转化成tesnor，所以还是（B,3,H,W）
                images = batch['image']  
                gt_masks = batch['mask']  # (B, 1, H, W)
                b,_,h,w = gt_masks.shape
                if h!=256 or w != 256:
                    gt_masks = F.interpolate(gt_masks, size=(256, 256), mode='nearest')
                click_points = batch['click_point']  # (B, N, 2)
                point_labels = batch['point_label']  # (B, N)
                # 梯度清零
                self.optimizer.zero_grad()
                # 前向传播
                outputs = self.pedestrain_sam2.forward(images, click_points, point_labels)
                # 计算损失
                loss = self.criterion.forward(outputs, gt_masks.float())
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # 日志记录
                if batch_idx % self.log_interval == 0:
                    logger.info(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
                    self.writer.add_scalar('Training/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

            val_loss = self.validate(epoch)

            # 保存模型检查点
            torch.save(self.model.state_dict(), f"{self.save_path}/model_epoch_{epoch+1}.pth")

            # 记录 epoch 的损失
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.writer.add_scalar('Epoch/Training Loss', avg_epoch_loss, epoch)
            self.writer.add_scalar('Epoch/Validation Loss', val_loss, epoch)
            logger.info(f'Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                images = batch['image']
                gt_masks = batch['mask']
                click_points = batch['click_point']
                point_labels = batch['point_label']
                b,_,h,w = gt_masks.shape
                if h!=256 or w != 256:
                    gt_masks = F.interpolate(gt_masks, size=(256, 256), mode='nearest')
                outputs = self.model(images, click_points, point_labels)
                loss = self.criterion(outputs, gt_masks.float())

                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        logger.info(f'Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}')
        return val_loss

def main():
    from dataset_mots import MOTSDataset
    config_path = "./train_config/train_large.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config["dataset"]["train_dataloader"] == "MOTS":
        train_dataset = MOTSDataset(root_path=config["dataset"]["train_dataset_root_path"],augment=False)
    if config["dataset"]["val_dataloader"] == "MOTS":
        val_dataset = MOTSDataset(root_path=config["dataset"]["val_dataset_root_path"],augment=False)
    train_dataloader = DataLoader(train_dataset,batch_size=2,shuffle=False)
    val_dataloader = DataLoader(val_dataset,batch_size=4,shuffle=False)
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model = build_sam2(model_config_path_for_build_sam2, sam2_checkpoint, device=config["train"]["device"], apply_postprocessing=True)
    pedestrain_sam_model = PedestrainSAM2(model=sam2_model,config=config)
    trainer_PedestrainSAM = Trainer(pedestrain_sam_model,train_loader=train_dataloader,val_loader=val_dataloader,config=config)
    trainer_PedestrainSAM.train()
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
# %%
