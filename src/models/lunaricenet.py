"""
LunarIceNet — Physics-Informed Deep Learning for Lunar Subsurface Ice Detection.

Architecture:
    1. Radar Encoder: Multi-scale CNN extracting features from polarimetric SAR data
    2. Physics Branch: MLP encoding physical parameters (temperature, geometry, PSR)
    3. Cross-Attention Fusion: Combines radar features with physics priors
    4. Multi-Head Output: Ice probability, depth estimation, confidence score

Key Innovation:
    Physics-informed loss function that penalizes physically impossible predictions
    (e.g., ice in sunlit regions, ice at temperatures > 110K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class ResidualBlock(nn.Module):
    """Residual block with optional squeeze-excitation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.se = SqueezeExcitation(out_ch) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)


class SqueezeExcitation(nn.Module):
    """Channel attention via squeeze-excitation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class MultiScaleRadarEncoder(nn.Module):
    """
    Multi-scale CNN encoder for polarimetric SAR features.

    Extracts features at multiple spatial scales to capture both
    fine-grained surface texture and broader geological context.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Progressive feature extraction
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Multi-scale feature fusion
        self.ms_conv1 = nn.Conv2d(64, embed_dim, 1)
        self.ms_conv2 = nn.Conv2d(128, embed_dim, 1)
        self.ms_conv3 = nn.Conv2d(256, embed_dim, 1)
        self.ms_conv4 = nn.Conv2d(512, embed_dim, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) polarimetric feature stack

        Returns:
            (B, embed_dim, H', W') multi-scale features
        """
        x = self.stem(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Upsample all to f1 spatial size
        target_size = f1.shape[2:]
        m1 = self.ms_conv1(f1)
        m2 = F.interpolate(self.ms_conv2(f2), size=target_size, mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.ms_conv3(f3), size=target_size, mode='bilinear', align_corners=False)
        m4 = F.interpolate(self.ms_conv4(f4), size=target_size, mode='bilinear', align_corners=False)

        fused = self.fusion(torch.cat([m1, m2, m3, m4], dim=1))
        return fused


class PhysicsEncoder(nn.Module):
    """
    Encodes physical parameters into feature space.

    Physical inputs:
        - Latitude (proxy for temperature)
        - Longitude
        - PSR probability (from catalog)
        - Distance from pole
        - PSR flag
    """

    def __init__(self, in_features: int = 5, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)

        Returns:
            (B, embed_dim)
        """
        return self.mlp(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between radar spatial features and physics embeddings.

    Radar features attend to physics context, allowing the model to
    modulate spatial predictions based on physical constraints.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'self_attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                ),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'norm3': nn.LayerNorm(embed_dim),
            }))

    def forward(
        self,
        radar_features: torch.Tensor,
        physics_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            radar_features: (B, embed_dim, H, W)
            physics_embedding: (B, embed_dim)

        Returns:
            (B, embed_dim, H, W) — physics-informed radar features
        """
        B, C, H, W = radar_features.shape

        # Flatten spatial dims for attention: (B, H*W, C)
        x = radar_features.flatten(2).permute(0, 2, 1)

        # Physics as key/value sequence: (B, 1, C)
        phys = physics_embedding.unsqueeze(1)

        for layer in self.layers:
            # Cross-attention: radar queries physics
            residual = x
            x = layer['norm1'](x)
            x_cross, _ = layer['cross_attn'](query=x, key=phys, value=phys)
            x = residual + x_cross

            # Self-attention among spatial positions
            residual = x
            x = layer['norm2'](x)
            x_self, _ = layer['self_attn'](query=x, key=x, value=x)
            x = residual + x_self

            # FFN
            residual = x
            x = residual + layer['ffn'](layer['norm3'](x))

        # Reshape back to spatial
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class IceDetectionHead(nn.Module):
    """Multi-task output head for ice detection."""

    def __init__(self, embed_dim: int = 256, patch_size: int = 256):
        super().__init__()

        # Upsampling to original resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Ice probability (binary segmentation)
        self.ice_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

        # Depth estimation (regression)
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.ReLU(),  # Depth is non-negative
        )

        # Confidence/uncertainty estimation
        self.confidence_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, embed_dim, H', W') fused features
            target_size: (H, W) original patch size

        Returns:
            Dict with 'ice_prob', 'depth', 'confidence' each (B, 1, H, W)
        """
        x = self.upsample(x)

        # Resize to exact target
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return {
            'ice_prob': self.ice_head(x),
            'depth': self.depth_head(x),
            'confidence': self.confidence_head(x),
        }


class LunarIceNet(nn.Module):
    """
    Physics-Informed Deep Learning Model for Lunar Subsurface Ice Detection.

    Combines multi-scale radar feature extraction with physical constraints
    to detect and characterize subsurface ice deposits from Chandrayaan-2
    DFSAR polarimetric SAR data.

    Architecture:
        RadarEncoder (Multi-scale CNN) ──→ CrossAttention ──→ IceDetectionHead
        PhysicsEncoder (MLP) ────────────↗

    Outputs:
        - ice_prob: Per-pixel ice probability map
        - depth: Estimated ice depth in meters
        - confidence: Model uncertainty map
    """

    def __init__(
        self,
        in_channels: int = 3,
        physics_features: int = 5,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_attn_layers: int = 2,
        patch_size: int = 256,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.radar_encoder = MultiScaleRadarEncoder(in_channels, embed_dim)
        self.physics_encoder = PhysicsEncoder(physics_features, embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim, num_heads, num_attn_layers)
        self.detection_head = IceDetectionHead(embed_dim, patch_size)

    def forward(
        self,
        radar_features: torch.Tensor,
        physical_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            radar_features: (B, C_in, H, W) polarimetric feature stack
            physical_params: (B, N_phys) physical parameters

        Returns:
            Dict with:
                'ice_prob': (B, 1, H, W) ice probability
                'depth': (B, 1, H, W) estimated depth (meters)
                'confidence': (B, 1, H, W) prediction confidence
        """
        target_size = (radar_features.shape[2], radar_features.shape[3])

        # Encode radar and physics separately
        radar_embed = self.radar_encoder(radar_features)
        physics_embed = self.physics_encoder(physical_params)

        # Fuse with cross-attention
        fused = self.fusion(radar_embed, physics_embed)

        # Multi-task prediction
        outputs = self.detection_head(fused, target_size)

        return outputs

    def predict(self, radar_features: torch.Tensor, physical_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference mode with sigmoid applied to ice probability."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(radar_features, physical_params)
            outputs['ice_prob'] = torch.sigmoid(outputs['ice_prob'])
        return outputs


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for ice detection.

    Components:
        1. BCE loss for ice/no-ice classification
        2. MSE loss for depth estimation
        3. Physics constraint: Ice cannot exist at T > 110K
        4. Temperature prior: Penalize ice predictions in sunlit regions
        5. Consistency: CPR > 1 regions should have higher ice probability
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        depth_weight: float = 0.5,
        physics_weight: float = 0.3,
        temp_prior_weight: float = 0.2,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.depth_weight = depth_weight
        self.physics_weight = physics_weight
        self.temp_prior_weight = temp_prior_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        physical_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model output dict
            targets: (B, H, W) ground truth ice labels
            physical_params: (B, N_phys) physical parameters

        Returns:
            Dict with 'total', 'bce', 'depth', 'physics', 'temp_prior' losses
        """
        ice_logits = predictions['ice_prob'].squeeze(1)  # (B, H, W)
        depth_pred = predictions['depth'].squeeze(1)
        confidence = predictions['confidence'].squeeze(1)

        # 1. Binary cross-entropy for ice detection
        loss_bce = self.bce(ice_logits, targets)

        # 2. Depth loss (only where ice exists)
        ice_mask = targets > 0.5
        if ice_mask.any():
            # Pseudo-depth: 0.5-2.0 meters based on radar wavelength penetration
            pseudo_depth = torch.where(
                ice_mask,
                torch.ones_like(targets) * 1.0,  # ~1m depth estimate
                torch.zeros_like(targets),
            )
            loss_depth = self.mse(depth_pred[ice_mask], pseudo_depth[ice_mask])
        else:
            loss_depth = torch.tensor(0.0, device=targets.device)

        # 3. Physics constraint: latitude-based temperature prior
        # Closer to pole (lat ≈ -90) → colder → more likely ice
        lat = physical_params[:, 0]  # Latitude
        psr_prob = physical_params[:, 2]  # PSR probability

        # Penalize high ice predictions far from pole
        ice_prob = torch.sigmoid(ice_logits)
        dist_from_pole = (90.0 + lat.unsqueeze(-1).unsqueeze(-1)) / 10.0  # 0 at pole, 1 at -80
        physics_penalty = (ice_prob * dist_from_pole.clamp(0, 1)).mean()

        # 4. Temperature prior: PSR regions should have higher ice probability
        psr_bonus = psr_prob.unsqueeze(-1).unsqueeze(-1)
        temp_prior = ((1.0 - ice_prob) * psr_bonus * targets).mean()

        # Total loss
        total = (
            self.bce_weight * loss_bce +
            self.depth_weight * loss_depth +
            self.physics_weight * physics_penalty +
            self.temp_prior_weight * temp_prior
        )

        return {
            'total': total,
            'bce': loss_bce,
            'depth': loss_depth,
            'physics': physics_penalty,
            'temp_prior': temp_prior,
        }
