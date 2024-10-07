# %% setup environment
from concurrent.futures import ProcessPoolExecutor
import os
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from loguru import logger
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from typing import Optional
from dataset_mots import MOTSDataset
from PIL import Image, ImageDraw
# Ensure that your custom modules are properly imported
# from dataset_mots import MOTSDataset

class PedestrainSAM2():
    def __init__(
        self,
        model: SAM2Base,
        config,
        device_index,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
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
        self.device = torch.device('cuda', device_index)
        # Freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, point_coords, point_labels):
        """
        image tensor: (B, 3, H, W)
        points_coords: (B, N, 2)
        point_labels: (B, N)
        """
        assert image.size(1) == 3, "Input image must have 3 channels."
        self._orig_hw = [(image.size(2), image.size(3))]
        if not self.config["train"]["transfer_image_in_dataset"]:
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
        ) # spare [b,2,256],[6,256,64,64]
        if not self.config["train"]["transfer_image_in_dataset"]:
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in _features["high_res_feats"]] # 这句话直接把[b,32,256,256] 变成[1,32,256,256],想想都觉得铁错
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        low_res_masks_logits = torch.clamp(low_res_masks_logits, -32.0, 32.0)
        prd_masks = self._transforms.postprocess_masks(low_res_masks_logits, self._orig_hw[-1])
        return prd_masks

    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest res feat. map during training on videos
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
        pass  # Implement inference logic if needed


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        smooth = 1e-5
        preds = torch.sigmoid(logits)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )
        return bce_loss + dice_loss


class Trainer:
    def __init__(self, wrap_model: PedestrainSAM2, train_loader, val_loader, config, rank, device_index):
        self.rank = rank
        self.device = torch.device('cuda', device_index)
        self.wrap_model = wrap_model
        self.model = wrap_model.sam2_model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Define loss function
        self.criterion = BCEDiceLoss().to(self.device)

        # Define optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(config["train"]["learning_rate"]),
        )
        #Total number of training steps
        self.num_epochs = config["train"]["num_epochs"]
        self.total_steps = self.num_epochs * len(self.train_loader)

        # Define learning rate scheduler (Cosine Annealing to zero)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=0
        )

        # Initialize GradScaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        self.num_epochs = config["train"]["num_epochs"]
        self.save_path = config["train"]["save_path"]
        self.validate_save_path = os.path.join(self.save_path,"validate")

        # Initialize logging (only on main process)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=config["log"]["tensorboard_log_dir"])
            logger.add(config["log"]["loguru_log_file"])
            logger.success(
                f"------------------------------------{self.device}Trainer initialized.---------------------------------"
            )
        self.log_interval = config["log"]["log_interval"]
        self.config = config

    def train(self):
        start_time = time.time()
        total_steps = self.num_epochs * len(self.train_loader)
        global_step = 0  # Track the global step
        for epoch in range(self.num_epochs):
            self.model.train()
            if self.config["train"]["validate_first"] == True:
                val_loss = self.validate(epoch)
            if self.train_loader.sampler is not None and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                global_step += 1
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v,torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                images = batch["image"] # B,3, 1024,1024  一般情况下是已经resize成1024的
                gt_masks = batch["mask"]  # (B, 1, H, W)
                b, _, h, w = gt_masks.shape
                click_points = batch["click_point"]  # (B, N, 2) 2014的坐标
                point_labels = batch["point_label"]  # (B, N)
                is_person_labels = batch["is_person"]  # (B, 1)
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    outputs = self.wrap_model.forward(images, click_points, point_labels)
                    loss = self.criterion(outputs, gt_masks.float())

                # Backward and optimize
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                epoch_loss += loss.item()

                # Logging (only on main process)
                if batch_idx % self.log_interval == 0 and self.rank == 0:
                    # 计算当前步骤数和已经消耗的时间
                    current_step = epoch * len(self.train_loader) + batch_idx + 1
                    elapsed_time = time.time() - start_time

                    # 估计总的训练时间和剩余时间
                    estimated_total_time = elapsed_time / current_step * total_steps
                    remaining_time = estimated_total_time - elapsed_time

                    # 将剩余时间转换为小时、分钟和秒
                    remaining_hours = int(remaining_time // 3600)
                    remaining_minutes = int((remaining_time % 3600) // 60)
                    remaining_seconds = int(remaining_time % 60)

                    logger.info(
                        f"Epoch [{epoch+1}/{self.num_epochs}], "
                        f"Step [{batch_idx+1}/{len(self.train_loader)}], "
                        f"Loss: {loss.item():.4f}, "
                        f"This Epoch Remaining Time: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
                    )
                    self.writer.add_scalar(
                        "Training/Loss",
                        loss.item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )

            if self.config["train"]["validate_first"] == False:
                val_loss = self.validate(epoch)
            # Save checkpoint (only on main process)
            if self.rank == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_path}/model_epoch_{epoch+1}.pth",
                )

                # Record epoch loss
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.writer.add_scalar("Epoch/Training Loss", avg_epoch_loss, epoch)
                logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}] Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
                )
        if self.config["train"]["validate_first"] == True:
            val_loss = self.validate(epoch)
    
    def blend_image_with_mask(self, original_image, mask_image, mask_color=(255, 0, 0, 100)):
        # Convert mask to RGBA with desired color
        mask_rgba = Image.new('RGBA', original_image.size, mask_color)
        mask_alpha = mask_image.convert('L')  # Convert mask to grayscale
        mask_rgba.putalpha(mask_alpha)
        original_image_rgba = original_image.convert('RGBA')
        blended = Image.alpha_composite(original_image_rgba, mask_rgba)
        return blended

    def draw_click_points(self, image, click_points, point_labels, image_size):
        draw = ImageDraw.Draw(image)
        orig_w, orig_h = image_size
        # Assuming images are resized to (1024, 1024) during preprocessing
        scale_w = orig_w / 1024
        scale_h = orig_h / 1024

        points = click_points.cpu().numpy()
        labels = point_labels.cpu().numpy()
        for pt, label in zip(points, labels):
            x = pt[0] * scale_w
            y = pt[1] * scale_h
            # Draw the point
            if label == 1:
                color = 'green'  # Positive point
            else:
                color = 'red'    # Negative point
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        return image
    
    def save_image_and_masks(self, img_path, original_image, output_mask, gt_mask, click_points, point_labels, epoch):
        # Compute paths
        last_three_parts = os.path.join(*img_path.split(os.sep)[-3:])  # e.g., 'MOTS20-09/img1/000426.jpg'
        save_dir = os.path.join(self.validate_save_path, str(epoch+1), os.path.dirname(last_three_parts))
        os.makedirs(save_dir, exist_ok=True)

        # Save original image
        save_img_path = os.path.join(self.validate_save_path, str(epoch+1), last_three_parts)
        original_image.save(save_img_path)

        # Resize masks to original image size
        # Predicted mask
        pred_mask = Image.fromarray((output_mask * 255).astype(np.uint8))
        pred_mask = pred_mask.resize(original_image.size, resample=Image.BILINEAR)
        # Ground truth mask
        gt_mask_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
        gt_mask_img = gt_mask_img.resize(original_image.size, resample=Image.NEAREST)

        # Save predicted mask only image
        save_pred_mask_only_path = os.path.join(
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0] + '_pred_mask_only.jpg'
        )
        pred_mask.convert('L').save(save_pred_mask_only_path)

        # Save ground truth mask only image
        save_gt_mask_only_path = os.path.join(
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0] + '_gt_mask_only.jpg'
        )
        gt_mask_img.convert('L').save(save_gt_mask_only_path)

        # Create blended image of original image and predicted mask
        blended_pred = self.blend_image_with_mask(original_image, pred_mask, mask_color=(255, 0, 0, 100))
        # Draw click points on blended_pred
        blended_pred = self.draw_click_points(blended_pred, click_points, point_labels, image_size=original_image.size)
        # Save blended image with predicted mask
        save_pred_mask_img_path = os.path.join(
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0] + '_pred_mask_overlay.jpg'
        )
        blended_pred.convert('RGB').save(save_pred_mask_img_path)

        # Create blended image of original image and ground truth mask
        blended_gt = self.blend_image_with_mask(original_image, gt_mask_img, mask_color=(0, 255, 0, 100))
        # Save blended image with ground truth mask
        save_gt_mask_img_path = os.path.join(
            self.validate_save_path,
            str(epoch+1),
            os.path.splitext(last_three_parts)[0] + '_gt_mask_overlay.jpg'
        )
        blended_gt.convert('RGB').save(save_gt_mask_img_path)
        
        
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        start_time = time.time()
        total_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                images = batch["image"]
                gt_masks = batch["mask"]
                click_points = batch["click_point"]
                point_labels = batch["point_label"]
                image_paths = batch["image_path"]
                b, _, h, w = gt_masks.shape
                # if h != 256 or w != 256: # 现在会把输出的mask调回(1024,1024)
                #     gt_masks = F.interpolate(gt_masks, size=(256, 256), mode="nearest")

                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    outputs_low_res_mask_logits = self.wrap_model.forward(images, click_points, point_labels) # [b,how_many_points,256,256]
                    loss = self.criterion(outputs_low_res_mask_logits, gt_masks.float())

                val_loss += loss.item()

                for i in range(b):
                    img_path = image_paths[i]
                    original_image = Image.open(img_path).convert('RGB')
                    output_mask = (outputs_low_res_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    gt_mask = gt_masks[i].cpu().numpy().squeeze()
                    click_pts = click_points[i]
                    point_lbls = point_labels[i]

                    # Call the helper function
                    self.save_image_and_masks(img_path, original_image, output_mask, gt_mask, click_pts, point_lbls, epoch)
                
                if (batch_idx + 1) % 10 == 0:
                    elapsed_time = time.time() - start_time
                    batches_remaining = total_batches - (batch_idx + 1)
                    estimated_time_per_batch = elapsed_time / (batch_idx + 1)
                    estimated_remaining_time = batches_remaining * estimated_time_per_batch

                    # 记录日志，包括已处理的batch和预计剩余时间
                    logger.info(
                        f"Processed {batch_idx + 1}/{total_batches} batches. "
                        f"Elapsed Time: {elapsed_time:.2f} seconds, "
                        f"Estimated Remaining Time: {estimated_remaining_time:.2f} seconds."
                    )
        val_loss /= len(self.val_loader)

        # Aggregate validation loss across all processes
        val_loss_tensor = torch.tensor(val_loss).to(self.device)
        if dist.is_initialized():
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / dist.get_world_size()

        # Logging (only on main process)
        if self.rank == 0:
            logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}"
            )
        return val_loss


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./train_config/train_large.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()


    # # Initialize distributed environment
    # torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    # rank = torch.distributed.get_rank()
    # world_size = torch.distributed.get_world_size()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        
        # DDP 模式
        dist.init_process_group(backend="nccl")
        args.local_rank = dist.get_rank() # args.local_rank有问题
        torch.cuda.set_device(args.local_rank)#理论上torchrun 自动分配
        device_index =args.local_rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.success(f"!!!!!!!!!!!!!!!Come to the DDP mode--rank:{rank}--world_size:{world_size}--device:{device}!!!!!!!!!!!!!!!!")
    else:
        # 单 GPU 模式
        device_index = args.device_index
        rank = args.local_rank
        world_size = 1
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")
    config["train"]["save_path"] = os.path.join(config["train"]["save_path"] ,formatted_time)
    if not os.path.exists(config["train"]["save_path"]):
        os.makedirs(config["train"]["save_path"])
    # Build dataset and dataloader
    # Replace 'MOTSDataset' with your actual dataset class
    if config["dataset"]["train_dataloader"] == "MOTS":
        train_dataset = MOTSDataset(
            root_path=config["dataset"]["train_dataset_root_path"], use_SAM2_transform=config["train"]["transfer_image_in_dataset"], augment=False
        )
    if config["dataset"]["val_dataloader"] == "MOTS":
        val_dataset = MOTSDataset(
            root_path=config["dataset"]["val_dataset_root_path"], use_SAM2_transform=config["train"]["transfer_image_in_dataset"], augment=False, max_length_for_validate = 4096
        )

    if world_size > 1:
        # 多 GPU 模式，使用 DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        # 单 GPU 模式，使用普通的随机采样器
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size_one_gpu"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["train"]["num_workers_dataloader"],
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size_one_gpu"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["train"]["num_workers_dataloader"],
        pin_memory=True,
    )
    # Build model
    model_config_path_for_build_sam2 = config["model"]["model_config"]
    sam2_checkpoint = config["model"]["pretrain_model_path"]
    sam2_model = build_sam2(
        model_config_path_for_build_sam2,
        sam2_checkpoint,
        device=torch.device('cuda', device_index),
        apply_postprocessing=True,
    )
    pedestrain_sam_model = PedestrainSAM2(model=sam2_model, config=config,device_index=device_index)


    # Wrap model with DDP
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        pedestrain_sam_model.sam2_model = DDP(
            pedestrain_sam_model.sam2_model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Create trainer
    trainer = Trainer(
        pedestrain_sam_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        config=config,
        rank=rank,
        device_index=device_index,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
