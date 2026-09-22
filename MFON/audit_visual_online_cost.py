#!/usr/bin/env python3
"""Read-only training-batch cost screen for visual directional allocation.

Compare a normal P4 loss/backward on a 32-sample training batch with the
additional eval-mode meta-gradient and cached two-sided visual KL score. The
model is never stepped or saved. Timings exclude data loading and synchronize
CUDA; this is an engineering screen, not a prediction of 25-epoch wall time.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from audit_task_aligned_gradient import sha256, tensors
from audit_visual_directional_difference import (
    cached_text_embedding,
    central_visual_kl_derivative,
)
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--meta-batch-size', type=int, default=4)
    parser.add_argument('--aux-batch-size', type=int, default=32)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--sample-order-seed', type=int, default=271828)
    parser.add_argument('--exclude-first', type=int, default=256)
    parser.add_argument('--step-ratio', type=float, default=0.0003)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.meta_batch_size, args.aux_batch_size, args.pairs) < 1:
        parser.error('batch sizes and pairs must be positive')
    if args.exclude_first < 0 or not 0 < args.step_ratio <= 0.01:
        parser.error('require exclude-first >= 0 and 0 < step-ratio <= 0.01')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def batch_for_indices(dataset, indices):
    subset = torch.utils.data.Subset(dataset, indices)
    return next(iter(torch.utils.data.DataLoader(
        subset, batch_size=len(indices), shuffle=False, pin_memory=True
    )))


def positive_visual_weights(utility, base_weight=0.3):
    """Exact per-batch mean budget for positive scores; otherwise abstain."""
    if utility.ndim != 1 or utility.numel() == 0:
        raise ValueError('utility must be a nonempty sample vector')
    if not bool(torch.isfinite(utility).all()) or base_weight < 0:
        raise ValueError('nonfinite utility or negative budget')
    positive = utility > 0
    count = int(positive.sum().item())
    if count == 0:
        return torch.zeros_like(utility), False
    weights = (positive.to(dtype=utility.dtype) *
               (base_weight * utility.numel() / count))
    return weights, True


def benchmark_pair(model, meta_inputs, aux_inputs, step_ratio):
    text, vision, audio, label = aux_inputs
    saved_buffers = [(buffer, buffer.detach().clone())
                     for buffer in model.buffers()]
    model.zero_grad()
    model.train()
    synchronize()
    begin = time.perf_counter()
    try:
        pred, _ = model(text, vision, audio, mode='train', epoch=25)
        budget = model.current_budgeted_aux
        if budget is None:
            raise RuntimeError('budgeted objective was not constructed')
        prediction_loss = torch.mean((pred - label).pow(2))
        training_loss = prediction_loss + budget['loss']
        training_loss.backward()
        synchronize()
        baseline_seconds = time.perf_counter() - begin
        baseline_loss = float(training_loss.detach().item())
        baseline_weight = float(budget['w_v'].detach().mean().item())
    finally:
        model.zero_grad()
        with torch.no_grad():
            for buffer, original in saved_buffers:
                buffer.copy_(original)
    del pred, budget, prediction_loss, training_loss
    model.eval()

    synchronize()
    begin = time.perf_counter()
    meta_text, meta_vision, meta_audio, meta_label = meta_inputs
    meta_pred, _ = model(meta_text, meta_vision, meta_audio, mode='test')
    meta_loss = torch.mean((meta_pred - meta_label).pow(2))
    meta_gradient = torch.autograd.grad(
        meta_loss, model.proj_v.weight
    )[0].detach()
    meta_norm = float(meta_gradient.norm().item())
    meta_loss_value = float(meta_loss.detach().item())
    del meta_pred, meta_loss
    if not meta_norm > 1e-12:
        raise ValueError('zero visual meta gradient')
    synchronize()
    meta_gradient_seconds = time.perf_counter() - begin

    cache_begin = time.perf_counter()
    direction = meta_gradient / meta_norm
    with torch.no_grad():
        cached_text = cached_text_embedding(model, text)
        teacher = model.vision_encoder_froze(vision).detach()
    parameter = model.proj_v.weight
    displacement = step_ratio * float(parameter.detach().norm().item())
    synchronize()
    cached_inputs_seconds = time.perf_counter() - cache_begin

    derivative_begin = time.perf_counter()
    derivative, _, _ = central_visual_kl_derivative(
        model, aux_inputs, teacher, parameter, direction, displacement,
        cached_text=cached_text
    )
    synchronize()
    directional_derivative_seconds = time.perf_counter() - derivative_begin

    utility = derivative.detach() * meta_norm
    positive = utility > 0
    weights, active = positive_visual_weights(utility)
    synchronize()
    score_seconds = time.perf_counter() - begin
    return {
        'baseline_train_backward_seconds': baseline_seconds,
        'additional_score_seconds': score_seconds,
        'additional_score_to_baseline_ratio': score_seconds / baseline_seconds,
        'meta_gradient_seconds': meta_gradient_seconds,
        'cached_inputs_seconds': cached_inputs_seconds,
        'directional_derivative_seconds': directional_derivative_seconds,
        'weight_and_accounting_seconds': score_seconds - (
            meta_gradient_seconds + cached_inputs_seconds +
            directional_derivative_seconds
        ),
        'baseline_loss': baseline_loss,
        'baseline_visual_weight_mean': baseline_weight,
        'meta_task_loss': meta_loss_value,
        'meta_gradient_norm': meta_norm,
        'positive_utility_count': int(positive.sum().item()),
        'aux_count': len(label),
        'visual_kl_active': active,
        'proposed_visual_weight_mean': float(weights.mean().item()),
        'proposed_visual_weight_min': float(weights.min().item()),
        'proposed_visual_weight_max': float(weights.max().item()),
        'utility_min': float(utility.min().item()),
        'utility_max': float(utility.max().item()),
    }


def main():
    args = arguments()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _, config, data_loader, _, tva_train, _, _ = load_dataset_modules('MOSEI')
    config.seed = args.seed
    dataset_cfg = config.MOSEI
    train_cfg = dataset_cfg.downStream.TVAtrain
    train_cfg.exp_name = args.exp_name
    train_cfg.use_alw = False
    train_cfg.use_budgeted_aux = True
    train_cfg.budget_warmup_mode = 'allocation'
    train_cfg.budget_warmup_epoch = 10
    train_cfg.use_interventional_reliability = True
    train_cfg.reliability_task_corrupt_scale = 0.0
    train_cfg.reliability_allocation_control = 'learned'
    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    dataset_cfg.path.raw_data_path = raw_data_path
    checkpoint = os.path.join(dataset_cfg.path.model_path, str(args.seed),
                              args.exp_name, 'TVA_fusion_model.pt')
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    require_files([
        raw_data_path, checkpoint,
        os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
        os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
    ], 'visual online cost audit')
    dataset = data_loader.MOSEIDataset('train', raw_data_path)
    indices = list(range(args.exclude_first, len(dataset)))
    needed = args.pairs * (args.meta_batch_size + args.aux_batch_size)
    if len(indices) < needed:
        raise ValueError('not enough disjoint training examples')
    generator = torch.Generator().manual_seed(args.sample_order_seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    indices = [indices[position] for position in order[:needed]]

    model = tva_train.TVA_fusion(config).to(config.DEVICE)
    model.load_froze()
    state = torch.load(checkpoint, map_location=config.DEVICE)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError('checkpoint keys do not match: missing=%s unexpected=%s'
                           % (incompatible.missing_keys, incompatible.unexpected_keys))
    del state
    if torch.cuda.is_available() and hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
    result = {
        'design': 'read-only benchmark: baseline train backward vs eval-mode visual score',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_sha256': sha256(checkpoint),
        'meta_batch_size': args.meta_batch_size,
        'aux_batch_size': args.aux_batch_size, 'pairs': args.pairs,
        'sample_order_seed': args.sample_order_seed,
        'exclude_first': args.exclude_first,
        'step_ratio': args.step_ratio,
        'records': [],
    }
    for pair_number in range(args.pairs):
        start = pair_number * (args.meta_batch_size + args.aux_batch_size)
        middle = start + args.meta_batch_size
        end = middle + args.aux_batch_size
        meta_ids = indices[start:middle]
        aux_ids = indices[middle:end]
        if set(meta_ids) & set(aux_ids):
            raise ValueError('meta and auxiliary samples overlap')
        meta_inputs = tensors(
            batch_for_indices(dataset, meta_ids), config.DEVICE
        )
        aux_inputs = tensors(
            batch_for_indices(dataset, aux_ids), config.DEVICE
        )
        record = benchmark_pair(model, meta_inputs, aux_inputs, args.step_ratio)
        record.update({
            'pair_number': pair_number,
            'meta_indices': meta_ids, 'aux_indices': aux_ids,
        })
        result['records'].append(record)
        print('pair %d: normal %.3fs, score %.3fs, ratio %.2fx' %
              (pair_number, record['baseline_train_backward_seconds'],
               record['additional_score_seconds'],
               record['additional_score_to_baseline_ratio']), flush=True)
        print('  score parts: meta %.3fs, cache %.3fs, direction %.3fs, other %.3fs' %
              (record['meta_gradient_seconds'],
               record['cached_inputs_seconds'],
               record['directional_derivative_seconds'],
               record['weight_and_accounting_seconds']), flush=True)
    if args.pairs > 1:
        kept = result['records'][1:]
        result['median_after_first_normal_seconds'] = statistics.median(
            x['baseline_train_backward_seconds'] for x in kept
        )
        result['median_after_first_score_seconds'] = statistics.median(
            x['additional_score_seconds'] for x in kept
        )
        result['median_after_first_score_ratio'] = statistics.median(
            x['additional_score_to_baseline_ratio'] for x in kept
        )
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('saved online cost audit to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
