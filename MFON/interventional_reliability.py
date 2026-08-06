import torch
import torch.nn.functional as F
from torch import nn


def active_step_mask(features, eps=1e-8):
    if features.ndim != 3:
        raise ValueError('features must have shape [batch, time, feature].')
    return features.abs().sum(dim=-1, keepdim=True) > eps


def active_rms(features, eps=1e-8):
    mask = active_step_mask(features).expand_as(features)
    count = mask.sum(dim=(1, 2)).clamp_min(1)
    squared = (features * mask).pow(2).sum(dim=(1, 2))
    return torch.sqrt(squared / count + eps)


def sample_ordered_severities(features, max_severity, corrupt_prob):
    if max_severity <= 0:
        raise ValueError('max_severity must be positive.')
    if not 0.0 <= corrupt_prob <= 1.0:
        raise ValueError('corrupt_prob must be in [0, 1].')
    batch = features.size(0)
    draws = torch.rand(batch, 2, device=features.device, dtype=features.dtype)
    low = draws.min(dim=1).values * float(max_severity)
    high = draws.max(dim=1).values * float(max_severity)
    enabled = (
        torch.rand(batch, device=features.device, dtype=features.dtype)
        < float(corrupt_prob)
    ).to(features.dtype)
    return low * enabled, high * enabled


def gaussian_corruption_pair(features, low_severity, high_severity):
    if low_severity.ndim != 1 or high_severity.ndim != 1:
        raise ValueError('severity tensors must be one-dimensional.')
    if low_severity.shape != high_severity.shape or low_severity.numel() != features.size(0):
        raise ValueError('severity tensors must match the batch size.')
    if torch.any(low_severity < 0) or torch.any(high_severity < low_severity):
        raise ValueError('severities must satisfy 0 <= low <= high.')

    mask = active_step_mask(features).expand_as(features)
    scale = active_rms(features).view(-1, 1, 1)
    direction = torch.randn_like(features) * scale * mask
    low = features + direction * low_severity.view(-1, 1, 1)
    high = features + direction * high_severity.view(-1, 1, 1)
    return low, high


def scale_active_content(features, factors):
    if factors.ndim != 1 or factors.numel() != features.size(0):
        raise ValueError('factors must be a batch vector.')
    if torch.any(factors <= 0):
        raise ValueError('scale factors must be positive.')
    mask = active_step_mask(features).expand_as(features)
    scaled = features * factors.view(-1, 1, 1)
    return torch.where(mask, scaled, features)


class ReliabilityHead(nn.Module):
    """Estimate reliability from scale-normalized temporal feature statistics."""

    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError('feature_dim and hidden_dim must be positive.')
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def descriptor(self, features):
        if features.ndim != 3 or features.size(-1) != self.feature_dim:
            raise ValueError('unexpected reliability-head input shape.')
        step_mask = active_step_mask(features)
        mask = step_mask.expand_as(features)
        normalized = F.layer_norm(features, (self.feature_dim,)) * mask
        count = step_mask.sum(dim=1).clamp_min(1).to(features.dtype)
        mean = normalized.sum(dim=1) / count
        centered = (normalized - mean.unsqueeze(1)) * mask
        std = torch.sqrt(centered.pow(2).sum(dim=1) / count + 1e-8)

        pair_mask = (step_mask[:, 1:] & step_mask[:, :-1]).expand(
            -1, -1, self.feature_dim
        )
        pair_count = pair_mask[:, :, :1].sum(dim=1).clamp_min(1).to(features.dtype)
        temporal = (
            (normalized[:, 1:] - normalized[:, :-1]).abs() * pair_mask
        ).sum(dim=1) / pair_count
        return torch.cat([mean, std, temporal], dim=-1)

    def forward(self, features):
        return torch.sigmoid(self.net(self.descriptor(features)).view(-1))


class TemporalReliabilityHead(nn.Module):
    """Reliability head for low-dimensional acoustic temporal features."""

    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError('feature_dim and hidden_dim must be positive.')
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def descriptor(self, features):
        if features.ndim != 3 or features.size(-1) != self.feature_dim:
            raise ValueError('unexpected temporal reliability-head input shape.')
        step_mask = active_step_mask(features)
        mask = step_mask.expand_as(features)
        count = step_mask.sum(dim=1).clamp_min(1).to(features.dtype)
        mean = (features * mask).sum(dim=1) / count
        centered = (features - mean.unsqueeze(1)) * mask
        std = torch.sqrt(centered.pow(2).sum(dim=1) / count + 1e-8)
        normalized = centered / std.unsqueeze(1).clamp_min(1e-4)

        abs_level = normalized.abs().sum(dim=1) / count
        kurtosis = normalized.pow(4).sum(dim=1) / count

        pair_mask = (step_mask[:, 1:] & step_mask[:, :-1]).expand(
            -1, -1, self.feature_dim
        )
        pair_count = pair_mask[:, :, :1].sum(dim=1).clamp_min(1).to(features.dtype)
        first_difference = (
            (normalized[:, 1:] - normalized[:, :-1]).abs() * pair_mask
        ).sum(dim=1) / pair_count
        first_difference_rms = torch.sqrt(
            (
                (normalized[:, 1:] - normalized[:, :-1]).pow(2) * pair_mask
            ).sum(dim=1)
            / pair_count
            + 1e-8
        )
        lag_correlation = (
            normalized[:, 1:] * normalized[:, :-1] * pair_mask
        ).sum(dim=1) / pair_count
        return torch.cat(
            [
                abs_level,
                first_difference,
                first_difference_rms,
                lag_correlation,
                kurtosis.clamp_max(20.0),
            ],
            dim=-1,
        )

    def forward(self, features):
        return torch.sigmoid(self.net(self.descriptor(features)).view(-1))


def blend_corruption(clean, corrupted, progress):
    if clean.shape != corrupted.shape:
        raise ValueError('clean and corrupted tensors must have identical shapes.')
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError('progress must be in [0, 1].')
    return clean + float(progress) * (corrupted - clean)


def ordinal_reliability_pair(
    head,
    features,
    max_severity=1.0,
    corrupt_prob=0.5,
    margin=0.2,
    invariance_weight=0.1,
):
    if margin < 0 or invariance_weight < 0:
        raise ValueError('loss weights must be non-negative.')
    low_severity, high_severity = sample_ordered_severities(
        features, max_severity, corrupt_prob
    )
    low_features, high_features = gaussian_corruption_pair(
        features, low_severity, high_severity
    )
    q_clean = head(features)
    q_low = head(low_features)
    q_high = head(high_features)

    severity_scale = max(float(max_severity), 1e-8)
    clean_low_margin = float(margin) * low_severity / severity_scale
    low_high_margin = float(margin) * (high_severity - low_severity) / severity_scale
    rank_clean_low = F.relu(clean_low_margin - (q_clean - q_low))
    rank_low_high = F.relu(low_high_margin - (q_low - q_high))
    rank_loss = (rank_clean_low + rank_low_high).mean()

    log_scale = torch.empty(
        features.size(0), device=features.device, dtype=features.dtype
    ).uniform_(-0.7, 0.7)
    scaled_features = scale_active_content(features, log_scale.exp())
    invariance_loss = (head(scaled_features) - q_clean).abs().mean()

    return {
        'corrupted': high_features,
        'q_clean': q_clean,
        'q_low': q_low,
        'q_high': q_high,
        'severity_low': low_severity,
        'severity_high': high_severity,
        'rank_loss': rank_loss,
        'invariance_loss': invariance_loss,
        'loss': rank_loss + float(invariance_weight) * invariance_loss,
    }
