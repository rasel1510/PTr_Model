

import os
import math
import time
import copy
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# --------------------------------------------------------------------------------------------------
# 1. Stochastic Depth (DropPath) for Regularization
# --------------------------------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------------------------------------------------
# 2. Overlapped Patch Embedding (Hierarchical Pyramid Level Downsampling)
# --------------------------------------------------------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding module to convert 2D feature maps to 1D patch tokens
    while preserving spatial continuity at boundary regions of small ulcers.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


# --------------------------------------------------------------------------------------------------
# 3. Spatial Reduction Attention (SRA)
# --------------------------------------------------------------------------------------------------
class SpatialReductionAttention(nn.Module):
    """
    Spatial Reduction Attention (SRA) layer.
    Reduces spatial dimensions of Keys (K) and Values (V) via scale ratio R_i to scale attention
    computation linearly rather than quadratically, preserving salient local ulcer cues.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Scaled Dot-Product Attention: Softmax((Q * K^T) / sqrt(d_k)) * V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# --------------------------------------------------------------------------------------------------
# 4. Multi-Layer Perceptron (MLP) with GELU Activation
# --------------------------------------------------------------------------------------------------
class Mlp(nn.Module):
    """Feed-Forward Network with GELU non-linear activation."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        # Depthwise conv for localized spatial inductive bias
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --------------------------------------------------------------------------------------------------
# 5. Pyramid Transformer Encoder Block
# --------------------------------------------------------------------------------------------------
class Block(nn.Module):
    """Hierarchical Transformer Block combining LayerNorm, SRA, DropPath, and FFN."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        sr_ratio: int = 1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# --------------------------------------------------------------------------------------------------
# 6. Complete Pyramid Transformer (PTr) Backbone and Classification Network
# --------------------------------------------------------------------------------------------------
class PyramidTransformer(nn.Module):
    """
    Proposed Pyramid Transformer (PTr) for Diabetic Foot Ulcer (DFU) Analysis.
    
    Architecture:
      - 4 Hierarchical Stages with progressively decreasing spatial resolution
      - Spatial Reduction Attention (SRA) at each level
      - Transfer Learning compatible backbone with ImageNet feature extraction
      - Binary or Multi-class classification head for DFU detection
    """
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 2,  # Binary: Ulcer (1) vs Normal/Healthy (0)
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[float] = [4.0, 4.0, 4.0, 4.0],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Patch Embeddings for 4 stages
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Stages
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path_rate=dpr[cur + i], sr_ratio=sr_ratios[0]
            ) for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path_rate=dpr[cur + i], sr_ratio=sr_ratios[1]
            ) for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path_rate=dpr[cur + i], sr_ratio=sr_ratios[2]
            ) for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path_rate=dpr[cur + i], sr_ratio=sr_ratios[3]
            ) for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Classification Head (Transfer Learning Adapter)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)

        # Global representation without destroying localized cues
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

    def load_pretrained_weights(self, pretrained_path: Optional[str] = None):
        """
        Loads pre-trained backbone weights for Transfer Learning (TL).
        """
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Remove head weights if shapes mismatch
            if 'head.weight' in state_dict and state_dict['head.weight'].shape != self.head.weight.shape:
                del state_dict['head.weight']
                del state_dict['head.bias']
            self.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Successfully loaded Transfer Learning weights from: {pretrained_path}")
        else:
            print("[INFO] Initialized PTr weights with truncated normal initialization.")


# --------------------------------------------------------------------------------------------------
# 7. DFU Preprocessing, Data Augmentations & Dataset Loader
# --------------------------------------------------------------------------------------------------
def get_dfu_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns training and validation/testing transformations according to the paper:
    Resizing (224x224), Crop, Intensity Normalization to [0, 1] / Standardized Distribution,
    and clinically plausible augmentations (horizontal flip, subtle rotation, regulated contrast).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def build_dfu_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    val_split: float = 0.30,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """
    Prepares DataLoaders for DFU patches / TL folders.
    Supports either pre-split folder structures (train/val/test) or a single folder split (70% train / 30% val).
    """
    train_tf, val_tf = get_dfu_transforms(img_size)

    # Check if directory has train/val subdirectories or raw class folders
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if {'train', 'val'}.issubset(set(subdirs)):
        train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
        val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_tf)
        test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_tf) if 'test' in subdirs else None
        classes = train_ds.classes
    else:
        # Single directory split: 70% train, 30% val
        full_ds = datasets.ImageFolder(data_dir)
        classes = full_ds.classes
        total_len = len(full_ds)
        train_len = int((1.0 - val_split) * total_len)
        val_len = total_len - train_len

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = torch.utils.data.random_split(full_ds, [train_len, val_len], generator=generator)

        # Apply respective transforms
        class TransformedSubset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, idx):
                x, y = self.subset[idx]
                if self.transform:
                    x = self.transform(x)
                return x, y

            def __len__(self):
                return len(self.subset)

        train_ds = TransformedSubset(train_subset, train_tf)
        val_ds = TransformedSubset(val_subset, val_tf)
        test_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None

    return train_loader, val_loader, test_loader, classes


# --------------------------------------------------------------------------------------------------
# 8. Comprehensive Journal-Grade Training and Evaluation Suite
# --------------------------------------------------------------------------------------------------
class DFUTrainer:
    """
    Handles model training, validation, transfer-learning optimization, and metric calculations.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_classes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

        # Objective and Optimizer (as specified in methodology)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        total = len(all_targets)
        loss = running_loss / total
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Classification metrics
        acc = accuracy_score(all_targets, all_preds) * 100.0
        prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100.0
        rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100.0
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100.0

        # Specificity calculation for binary case
        cm = confusion_matrix(all_targets, all_preds)
        if self.num_classes == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
            auroc = roc_auc_score(all_targets, all_probs[:, 1]) * 100.0
        else:
            specificity = 0.0
            auroc = roc_auc_score(all_targets, all_probs, multi_class='ovr') * 100.0

        return {
            "loss": loss,
            "accuracy": acc,
            "precision": prec,
            "sensitivity_recall": rec,
            "specificity": specificity,
            "f1_score": f1,
            "auroc": auroc,
            "confusion_matrix": cm,
            "targets": all_targets,
            "probs": all_probs
        }

    def fit(self, num_epochs: int = 50, save_path: str = "best_dfu_ptr_model.pth") -> Dict[str, List[float]]:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auroc": []}
        best_val_auroc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        print(f"\n[START] Training PTr Model on {self.device} for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_auroc"].append(val_metrics["auroc"])

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2f}% | "
                f"AUROC: {val_metrics['auroc']:.2f}% | Rec: {val_metrics['sensitivity_recall']:.2f}% | Spec: {val_metrics['specificity']:.2f}%"
            )

            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_model_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), save_path)

        print(f"\n[FINISHED] Optimal Model saved with Best Validation AUROC: {best_val_auroc:.2f}%")
        self.model.load_state_dict(best_model_weights)
        return history


# --------------------------------------------------------------------------------------------------
# 9. Main Execution and Example Usage
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENVIRONMENT] Execution target: {device}")

    # Initialize the proposed PTr model
    model = PyramidTransformer(
        img_size=224,
        in_chans=3,
        num_classes=2,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.1,
        drop_path_rate=0.1
    )

    # Optional: Load pre-trained weights if available for transfer learning
    # model.load_pretrained_weights("pvt_v2_b0.pth")

    # Display model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Proposed PTr loaded. Total Trainable Parameters: {total_params / 1e6:.2f}M")

    # Synthetic verification run to validate tensor dimensions through all stages
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_out = model(dummy_input)
    print(f"[FORWARD CHECK] Input: {dummy_input.shape} -> Logits Output: {dummy_out.shape} (Valid)")

    # ----------------------------------------------------------------------------------------------
    # To run on your DFU Dataset directory:
    # Set your DFU dataset path (e.g., './DFU_dataset/patches')
    # ----------------------------------------------------------------------------------------------
    DFU_DATA_DIR = "./DFU_dataset"  # Point to your DFU patches directory

    if os.path.exists(DFU_DATA_DIR):
        train_loader, val_loader, test_loader, classes = build_dfu_dataloaders(
            data_dir=DFU_DATA_DIR,
            batch_size=32,
            val_split=0.30,  # 70% Train, 30% Val split as described
            img_size=224
        )
        print(f"[DATA] Classes detected: {classes}")

        trainer = DFUTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=1e-4,
            weight_decay=1e-4,
            num_classes=len(classes),
            device=device
        )

        history = trainer.fit(num_epochs=50, save_path="ptr_dfu_best_model.pth")
        final_metrics = trainer.evaluate(val_loader)
        print("\n" + "="*50)
        print(" FINAL VALIDATION METRICS (ELSEVIER Q1 REPORTING) ")
        print("="*50)
        print(f" Accuracy            : {final_metrics['accuracy']:.2f}%")
        print(f" AUROC (Micro/Macro) : {final_metrics['auroc']:.2f}%")
        print(f" Sensitivity / Recall: {final_metrics['sensitivity_recall']:.2f}%")
        print(f" Specificity         : {final_metrics['specificity']:.2f}%")
        print(f" Precision           : {final_metrics['precision']:.2f}%")
        print(f" F1-Score            : {final_metrics['f1_score']:.2f}%")
        print(" Confusion Matrix    :\n", final_metrics['confusion_matrix'])
        print("="*50)
    else:
        print(f"\n[NOTE] Place your DFU images in '{DFU_DATA_DIR}' to run live dataset training.")
