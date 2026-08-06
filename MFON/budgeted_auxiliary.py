import torch
import torch.nn.functional as F


def _check_pair(input1, input2, name):
    if input1.ndim != 2 or input2.ndim != 2:
        raise ValueError('%s expects two [batch, feature] tensors.' % name)
    if input1.shape != input2.shape:
        raise ValueError('%s expects tensors with identical shapes.' % name)


def per_sample_kl(student_embed, teacher_embed):
    """Return KL divergence for each sample without reducing the batch."""
    _check_pair(student_embed, teacher_embed, 'per_sample_kl')
    log_prob = F.log_softmax(student_embed, dim=-1)
    target_prob = F.softmax(teacher_embed, dim=-1)
    return F.kl_div(log_prob, target_prob, reduction='none').sum(dim=-1)


def per_sample_infonce(input1, input2, eps=1e-8):
    """Return the original MFON in-batch InfoNCE objective per sample."""
    _check_pair(input1, input2, 'per_sample_infonce')
    x1 = F.normalize(input1, p=2, dim=-1, eps=eps)
    x2 = F.normalize(input2, p=2, dim=-1, eps=eps)
    logits = torch.matmul(x1, x2.t())
    positive = torch.diagonal(logits)
    return torch.logsumexp(logits, dim=-1) - positive


def sample_quality_score(embed, target_embed=None, pred=None, q_type='align', temperature=1.0):
    """Compute a detached quality proxy for every sample in a batch."""
    if embed.ndim != 2:
        raise ValueError('sample_quality_score expects a [batch, feature] tensor.')
    if temperature <= 0:
        raise ValueError('temperature must be positive.')

    embed = embed.detach()
    if q_type == 'norm':
        dim_scale = max(1, embed.size(-1)) ** 0.5
        quality = torch.sigmoid(embed.norm(dim=-1) / (dim_scale * temperature))
    elif q_type == 'conf':
        if pred is None:
            quality = torch.ones(embed.size(0), device=embed.device, dtype=embed.dtype)
        else:
            quality = torch.sigmoid(pred.detach().view(-1).abs() / temperature)
    elif q_type == 'align':
        if target_embed is None:
            raise ValueError('align quality requires target_embed.')
        _check_pair(embed, target_embed, 'sample_quality_score')
        quality = (F.cosine_similarity(embed, target_embed.detach(), dim=-1) + 1.0) / 2.0
    else:
        raise ValueError('Unsupported quality type: %s' % q_type)
    return quality.clamp(0.0, 1.0)


def fixed_budget_weights(reliability, base_weight, progress=1.0, eps=1e-8):
    """Redistribute a fixed mean weight across samples using reliability."""
    if reliability.ndim != 1:
        raise ValueError('reliability must be a one-dimensional batch vector.')
    if reliability.numel() == 0:
        raise ValueError('reliability must not be empty.')
    if base_weight < 0:
        raise ValueError('base_weight must be non-negative.')
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError('progress must be in [0, 1].')

    scores = reliability.detach().clamp_min(0.0) + eps
    budget = torch.as_tensor(base_weight * float(progress), dtype=scores.dtype, device=scores.device)
    return budget * scores.numel() * scores / scores.sum()


def apply_quality_control(quality, mode='learned', oracle_quality=None):
    """Transform sample scores for equal-budget actionability controls."""
    if quality.ndim != 1:
        raise ValueError('quality control expects a one-dimensional tensor.')
    if mode == 'learned':
        controlled = quality
    elif mode == 'constant':
        controlled = torch.ones_like(quality)
    elif mode == 'permuted':
        controlled = quality.roll(1) if quality.numel() > 1 else quality
    elif mode == 'reversed':
        order = quality.argsort()
        controlled = torch.empty_like(quality)
        controlled[order] = quality[order.flip(0)]
    elif mode == 'oracle':
        if oracle_quality is None:
            raise ValueError('oracle control requires oracle_quality.')
        if oracle_quality.shape != quality.shape:
            raise ValueError('oracle_quality must match quality shape.')
        controlled = oracle_quality
    else:
        raise ValueError('Unsupported quality control: %s' % mode)
    return controlled.clamp(0.0, 1.0)


def weighted_batch_mean(per_sample_loss, weights):
    if per_sample_loss.ndim != 1 or weights.ndim != 1:
        raise ValueError('loss and weights must be one-dimensional batch vectors.')
    if per_sample_loss.shape != weights.shape:
        raise ValueError('loss and weights must have identical shapes.')
    return torch.sum(per_sample_loss * weights) / per_sample_loss.numel()


def build_budgeted_auxiliary(
    x_v_embed,
    x_a_embed,
    x_v_target,
    x_a_target,
    pred,
    loss_v_each,
    loss_a_each,
    loss_nce_v_each,
    loss_nce_a_each,
    q_type,
    temperature,
    delta_va,
    delta_nce,
    progress,
    eps=1e-8,
    quality_v=None,
    quality_a=None,
):
    """Build the complete fixed-budget auxiliary objective and audit values."""
    q_v = (
        sample_quality_score(x_v_embed, x_v_target, pred, q_type, temperature)
        if quality_v is None
        else quality_v
    )
    q_a = (
        sample_quality_score(x_a_embed, x_a_target, pred, q_type, temperature)
        if quality_a is None
        else quality_a
    )
    if q_v.ndim != 1 or q_a.ndim != 1:
        raise ValueError('quality overrides must be one-dimensional.')
    if q_v.size(0) != x_v_embed.size(0) or q_a.size(0) != x_a_embed.size(0):
        raise ValueError('quality overrides must match the batch size.')
    w_v = fixed_budget_weights(q_v, delta_va, progress, eps)
    w_a = fixed_budget_weights(q_a, delta_va, progress, eps)
    w_nce_v = fixed_budget_weights(q_v, delta_nce, progress, eps)
    w_nce_a = fixed_budget_weights(q_a, delta_nce, progress, eps)
    weighted_v = weighted_batch_mean(loss_v_each, w_v)
    weighted_a = weighted_batch_mean(loss_a_each, w_a)
    weighted_nce = (
        weighted_batch_mean(loss_nce_v_each, w_nce_v)
        + weighted_batch_mean(loss_nce_a_each, w_nce_a)
    )
    return {
        'loss': weighted_v + weighted_a + weighted_nce,
        'loss_v': weighted_v,
        'loss_a': weighted_a,
        'loss_nce': weighted_nce,
        'q_v': q_v,
        'q_a': q_a,
        'w_v': w_v,
        'w_a': w_a,
        'w_nce_v': w_nce_v,
        'w_nce_a': w_nce_a,
        'progress': torch.as_tensor(progress, dtype=x_v_embed.dtype, device=x_v_embed.device),
    }
