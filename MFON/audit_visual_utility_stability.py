#!/usr/bin/env python3
"""Audit whether visual update utility is stable across independent meta batches.

Each auxiliary batch is evaluated against several disjoint meta batches. The output
keeps every context-specific utility and exposes the mean utility as the prediction
target. This distinguishes sample-specific signal from random meta-batch context.
The audit is read-only and never updates or writes a checkpoint.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from audit_task_aligned_gradient import sha256, tensors
from audit_task_aligned_step import task_mse
from audit_visual_directional_difference import (
    cached_text_embedding,
    central_visual_kl_derivative,
)
from audit_visual_online_cost import batch_for_indices, synchronize
from audit_visual_utility_dataset import FEATURE_NAMES, cheap_visual_features
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-stage', default='final')
    parser.add_argument('--meta-batch-size', type=int, default=4)
    parser.add_argument('--aux-batch-size', type=int, default=32)
    parser.add_argument('--groups', type=int, default=24)
    parser.add_argument('--meta-contexts', type=int, default=6)
    parser.add_argument('--sample-order-seed', type=int, default=314159)
    parser.add_argument('--exclude-first', type=int, default=1024)
    parser.add_argument('--exclude-json', type=Path)
    parser.add_argument('--step-ratio', type=float, default=0.0003)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.meta_batch_size, args.aux_batch_size,
           args.groups, args.meta_contexts) < 1:
        parser.error('batch sizes, groups, and meta-contexts must be positive')
    if args.meta_contexts < 2:
        parser.error('meta-contexts must be at least two')
    if args.exclude_first < 0 or not 0 < args.step_ratio <= 0.01:
        parser.error('require exclude-first >= 0 and 0 < step-ratio <= 0.01')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    if args.exclude_json is not None:
        args.exclude_json = args.exclude_json.resolve()
        if not args.exclude_json.is_file():
            raise FileNotFoundError(args.exclude_json)
    return args


def excluded_indices(path):
    if path is None:
        return set()
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    excluded = set()
    for group in data.get('records', data.get('pairs', [])):
        excluded.update(int(value) for value in group.get('meta_indices', []))
        excluded.update(int(value) for value in group.get('aux_indices', []))
    if not excluded:
        raise ValueError('exclude JSON contains no sample indices')
    return excluded


def stability_columns(utility):
    """Return per-sample context stability for a [contexts, samples] tensor."""
    if utility.ndim != 2 or utility.size(0) < 2:
        raise ValueError('utility must have shape [contexts >= 2, samples]')
    signs = utility.sign()
    positive_fraction = (signs > 0).to(utility.dtype).mean(dim=0)
    negative_fraction = (signs < 0).to(utility.dtype).mean(dim=0)
    zero_fraction = (signs == 0).to(utility.dtype).mean(dim=0)
    majority_fraction = torch.stack(
        [positive_fraction, negative_fraction], dim=0
    ).max(dim=0)[0]
    pair_equal = []
    for left in range(signs.size(0)):
        for right in range(left + 1, signs.size(0)):
            valid = (signs[left] != 0) & (signs[right] != 0)
            pair_equal.append(
                ((signs[left] == signs[right]) & valid).to(utility.dtype)
            )
    sign_agreement = torch.stack(pair_equal, dim=0).mean(dim=0)
    mean = utility.mean(dim=0)
    standard_deviation = utility.std(dim=0, unbiased=False)
    first_matches_mean = ((signs[0] == mean.sign())
                          & (signs[0] != 0) & (mean.sign() != 0))
    return {
        'mean': mean,
        'standard_deviation': standard_deviation,
        'positive_fraction': positive_fraction,
        'negative_fraction': negative_fraction,
        'zero_fraction': zero_fraction,
        'majority_fraction': majority_fraction,
        'context_sign_agreement': sign_agreement,
        'first_matches_mean': first_matches_mean,
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
    train_cfg.use_interventional_reliability = True
    train_cfg.reliability_task_corrupt_scale = 0.0
    train_cfg.reliability_allocation_control = 'learned'
    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    dataset_cfg.path.raw_data_path = raw_data_path
    checkpoint = (str(args.checkpoint) if args.checkpoint is not None else
                  os.path.join(dataset_cfg.path.model_path, str(args.seed),
                               args.exp_name, 'TVA_fusion_model.pt'))
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    require_files([
        raw_data_path, checkpoint,
        os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
        os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
    ], 'visual utility stability audit')

    dataset = data_loader.MOSEIDataset('train', raw_data_path)
    prior = excluded_indices(args.exclude_json)
    excluded = set(range(min(args.exclude_first, len(dataset)))) | prior
    available = [index for index in range(len(dataset)) if index not in excluded]
    per_group = args.aux_batch_size + args.meta_contexts * args.meta_batch_size
    needed = args.groups * per_group
    if len(available) < needed:
        raise ValueError('not enough fresh disjoint training examples')
    generator = torch.Generator().manual_seed(args.sample_order_seed)
    order = torch.randperm(len(available), generator=generator).tolist()
    indices = [available[position] for position in order[:needed]]

    model = tva_train.TVA_fusion(config).to(config.DEVICE)
    model.load_froze()
    state = torch.load(checkpoint, map_location=config.DEVICE)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError('checkpoint keys do not match: missing=%s unexpected=%s'
                           % (incompatible.missing_keys,
                              incompatible.unexpected_keys))
    del state
    model.eval()
    if torch.cuda.is_available() and hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    parameter = model.proj_v.weight
    displacement = args.step_ratio * float(parameter.detach().norm().item())
    if not displacement > 0:
        raise ValueError('zero visual projection norm')
    result = {
        'design': 'repeated-meta visual utility stability audit',
        'scope': 'fresh disjoint training samples; eval-mode labels; no update',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_stage': args.checkpoint_stage,
        'checkpoint_path': os.path.relpath(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'parameter_block': 'proj_v.weight',
        'meta_batch_size': args.meta_batch_size,
        'aux_batch_size': args.aux_batch_size,
        'groups': args.groups, 'meta_contexts': args.meta_contexts,
        'sample_order_seed': args.sample_order_seed,
        'exclude_first': args.exclude_first,
        'exclude_json': (args.exclude_json.name
                         if args.exclude_json is not None else None),
        'excluded_prior_indices': len(prior),
        'step_ratio': args.step_ratio,
        'feature_names': FEATURE_NAMES,
        'records': [],
    }
    synchronize()
    start = time.perf_counter()
    for group_number in range(args.groups):
        start_index = group_number * per_group
        meta_count = args.meta_contexts * args.meta_batch_size
        meta_flat = indices[start_index:start_index + meta_count]
        aux_ids = indices[start_index + meta_count:start_index + per_group]
        meta_groups = [
            meta_flat[index:index + args.meta_batch_size]
            for index in range(0, meta_count, args.meta_batch_size)
        ]
        if len(set(meta_flat + aux_ids)) != len(meta_flat) + len(aux_ids):
            raise ValueError('samples overlap in group %d' % group_number)
        aux_inputs = tensors(batch_for_indices(dataset, aux_ids), config.DEVICE)
        columns, teacher = cheap_visual_features(model, aux_inputs)
        with torch.no_grad():
            cached_text = cached_text_embedding(model, aux_inputs[0])
        context_records = []
        utilities = []
        for context_number, meta_ids in enumerate(meta_groups):
            meta_inputs = tensors(
                batch_for_indices(dataset, meta_ids), config.DEVICE
            )
            meta_loss = task_mse(model, meta_inputs)
            meta_gradient = torch.autograd.grad(meta_loss, parameter)[0].detach()
            meta_norm = float(meta_gradient.norm().item())
            meta_loss_value = float(meta_loss.detach().item())
            del meta_loss
            if not meta_norm > 1e-12:
                raise ValueError('zero meta gradient in group %d context %d' %
                                 (group_number, context_number))
            direction = meta_gradient / meta_norm
            derivative, _, _ = central_visual_kl_derivative(
                model, aux_inputs, teacher, parameter, direction,
                displacement, cached_text=cached_text
            )
            utility = derivative.detach() * meta_norm
            if not bool(torch.isfinite(utility).all()):
                raise ValueError('nonfinite utility in group %d context %d' %
                                 (group_number, context_number))
            utilities.append(utility)
            context_records.append({
                'context_number': context_number,
                'meta_indices': [int(value) for value in meta_ids],
                'meta_task_mse': meta_loss_value,
                'meta_gradient_norm': meta_norm,
                'positive_utility_count': int((utility > 0).sum().item()),
            })
            del meta_gradient, direction, derivative, utility
        utility_matrix = torch.stack(utilities, dim=0)
        stability = stability_columns(utility_matrix)
        samples = []
        for position, sample_index in enumerate(aux_ids):
            samples.append({
                'sample_index': int(sample_index),
                'sample_position': position,
                'utility': float(stability['mean'][position].item()),
                'positive_utility': bool(stability['mean'][position].item() > 0),
                'single_context_utility': float(
                    utility_matrix[0, position].item()
                ),
                'context_utilities': [
                    float(value) for value in utility_matrix[:, position].tolist()
                ],
                'utility_standard_deviation': float(
                    stability['standard_deviation'][position].item()
                ),
                'positive_context_fraction': float(
                    stability['positive_fraction'][position].item()
                ),
                'majority_fraction': float(
                    stability['majority_fraction'][position].item()
                ),
                'context_sign_agreement': float(
                    stability['context_sign_agreement'][position].item()
                ),
                'first_matches_mean': bool(
                    stability['first_matches_mean'][position].item()
                ),
                'features': {
                    name: float(columns[name][position].item())
                    for name in FEATURE_NAMES
                },
            })
        stable_count = sum(sample['majority_fraction'] >= 5.0 / 6.0
                           for sample in samples)
        result['records'].append({
            'pair_number': group_number,
            'group_number': group_number,
            'meta_indices': [int(value) for value in meta_flat],
            'aux_indices': [int(value) for value in aux_ids],
            'contexts': context_records,
            'positive_mean_utility_count': sum(
                int(sample['positive_utility']) for sample in samples
            ),
            'stable_majority_count': stable_count,
            'samples': samples,
        })
        print('group %d: positive mean %d/%d, stable %d/%d' % (
            group_number,
            result['records'][-1]['positive_mean_utility_count'],
            len(samples), stable_count, len(samples)
        ), flush=True)
        del columns, teacher, cached_text, utilities, utility_matrix, stability
    synchronize()
    result['elapsed_seconds'] = time.perf_counter() - start
    result['sample_count'] = sum(
        len(group['samples']) for group in result['records']
    )
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('saved %d repeated-context samples to %s' %
          (result['sample_count'], args.output), flush=True)


if __name__ == '__main__':
    main()
