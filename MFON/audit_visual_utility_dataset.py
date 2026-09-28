#!/usr/bin/env python3
"""Collect cheap visual features with sparse parameter-space utility labels.

For each disjoint meta/auxiliary training-batch pair, compute one visual task
gradient on the meta batch and a cached two-sided directional derivative for all
auxiliary samples. The signed derivative is the local utility label. Features are
read from the unperturbed forward pass and detached before writing JSON. The script
does not update an optimizer or write a checkpoint.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_task_aligned_gradient import sha256, tensors
from audit_task_aligned_step import task_mse
from audit_visual_directional_difference import (
    cached_text_embedding,
    central_visual_kl_derivative,
)
from audit_visual_online_cost import batch_for_indices, synchronize
from budgeted_auxiliary import per_sample_kl
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


FEATURE_NAMES = [
    'old_reliability',
    'visual_kl',
    'student_teacher_cosine',
    'student_norm',
    'teacher_norm',
    'task_absolute_residual',
    'prediction_magnitude',
    'vision_rms_energy',
    'vision_nonzero_fraction',
    'vision_active_timesteps',
    'vision_temporal_activity',
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-stage', default='final')
    parser.add_argument('--meta-batch-size', type=int, default=4)
    parser.add_argument('--aux-batch-size', type=int, default=32)
    parser.add_argument('--pairs', type=int, default=64)
    parser.add_argument('--sample-order-seed', type=int, default=161803)
    parser.add_argument('--exclude-first', type=int, default=1024)
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
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    return args


def per_sample_mean(value):
    if value.ndim < 2:
        raise ValueError('expected a batch tensor with at least two dimensions')
    return value.reshape(value.size(0), -1).mean(dim=1)


def cheap_visual_features(model, inputs):
    """Return detached feature columns and baseline visual teacher embeddings."""
    text, vision, audio, label = inputs
    with torch.no_grad():
        prediction, embeddings = model(text, vision, audio, mode='test')
        student = embeddings[1]
        teacher = model.vision_encoder_froze(vision).detach()
        reliability = model.vision_reliability(vision).detach().view(-1)
        visual_kl = per_sample_kl(student, teacher)
        cosine = F.cosine_similarity(student, teacher, dim=-1)
        label = label.view(-1)
        prediction = prediction.view(-1)
        energy = per_sample_mean(vision.pow(2)).sqrt()
        nonzero = per_sample_mean((vision.abs() > 1e-8).to(vision.dtype))
        if vision.ndim >= 3:
            active = (vision.abs().sum(dim=-1) > 1e-8).to(vision.dtype).sum(dim=1)
            if vision.size(1) > 1:
                temporal = per_sample_mean((vision[:, 1:] - vision[:, :-1]).abs())
            else:
                temporal = torch.zeros_like(active)
        else:
            active = (vision.abs() > 1e-8).to(vision.dtype).sum(dim=1)
            temporal = torch.zeros_like(active)
        columns = {
            'old_reliability': reliability,
            'visual_kl': visual_kl,
            'student_teacher_cosine': cosine,
            'student_norm': student.norm(dim=-1),
            'teacher_norm': teacher.norm(dim=-1),
            'task_absolute_residual': (prediction - label).abs(),
            'prediction_magnitude': prediction.abs(),
            'vision_rms_energy': energy,
            'vision_nonzero_fraction': nonzero,
            'vision_active_timesteps': active,
            'vision_temporal_activity': temporal,
        }
    batch_size = vision.size(0)
    for name in FEATURE_NAMES:
        value = columns[name]
        if value.shape != (batch_size,) or not bool(torch.isfinite(value).all()):
            raise ValueError('invalid feature column %s' % name)
    return columns, teacher


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
    ], 'visual utility dataset audit')

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
    model.eval()
    if torch.cuda.is_available() and hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    parameter = model.proj_v.weight
    displacement = args.step_ratio * float(parameter.detach().norm().item())
    if not displacement > 0:
        raise ValueError('zero visual projection norm')
    result = {
        'design': 'read-only sparse visual task-utility label dataset',
        'scope': 'training split, eval-mode local labels; no optimizer update',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_stage': args.checkpoint_stage,
        'checkpoint_path': os.path.relpath(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'parameter_block': 'proj_v.weight',
        'meta_batch_size': args.meta_batch_size,
        'aux_batch_size': args.aux_batch_size, 'pairs': args.pairs,
        'sample_order_seed': args.sample_order_seed,
        'exclude_first': args.exclude_first, 'step_ratio': args.step_ratio,
        'feature_names': FEATURE_NAMES, 'records': [],
    }
    synchronize()
    start = time.perf_counter()
    for pair_number in range(args.pairs):
        start_index = pair_number * (args.meta_batch_size + args.aux_batch_size)
        middle = start_index + args.meta_batch_size
        end = middle + args.aux_batch_size
        meta_ids, aux_ids = indices[start_index:middle], indices[middle:end]
        if set(meta_ids) & set(aux_ids):
            raise ValueError('meta and auxiliary samples overlap')
        meta_inputs = tensors(batch_for_indices(dataset, meta_ids), config.DEVICE)
        aux_inputs = tensors(batch_for_indices(dataset, aux_ids), config.DEVICE)
        meta_loss = task_mse(model, meta_inputs)
        meta_gradient = torch.autograd.grad(meta_loss, parameter)[0].detach()
        meta_norm = float(meta_gradient.norm().item())
        meta_loss_value = float(meta_loss.detach().item())
        del meta_loss
        if not meta_norm > 1e-12:
            raise ValueError('zero visual meta gradient in pair %d' % pair_number)
        direction = meta_gradient / meta_norm
        columns, teacher = cheap_visual_features(model, aux_inputs)
        with torch.no_grad():
            cached_text = cached_text_embedding(model, aux_inputs[0])
        derivative, _, _ = central_visual_kl_derivative(
            model, aux_inputs, teacher, parameter, direction, displacement,
            cached_text=cached_text
        )
        utility = derivative.detach() * meta_norm
        if not bool(torch.isfinite(utility).all()):
            raise ValueError('nonfinite utility label in pair %d' % pair_number)
        records = []
        for position, sample_index in enumerate(aux_ids):
            records.append({
                'sample_index': int(sample_index),
                'sample_position': position,
                'utility': float(utility[position].item()),
                'positive_utility': bool(utility[position].item() > 0),
                'features': {
                    name: float(columns[name][position].item())
                    for name in FEATURE_NAMES
                },
            })
        positive = sum(int(record['positive_utility']) for record in records)
        result['records'].append({
            'pair_number': pair_number,
            'meta_indices': [int(i) for i in meta_ids],
            'aux_indices': [int(i) for i in aux_ids],
            'meta_task_mse': meta_loss_value,
            'meta_gradient_norm': meta_norm,
            'positive_utility_count': positive,
            'samples': records,
        })
        print('pair %d: positive utility %d/%d' %
              (pair_number, positive, len(records)), flush=True)
        del meta_gradient, direction, columns, teacher, cached_text, derivative, utility
    synchronize()
    result['elapsed_seconds'] = time.perf_counter() - start
    result['positive_utility_count'] = sum(
        pair['positive_utility_count'] for pair in result['records']
    )
    result['sample_count'] = sum(len(pair['samples']) for pair in result['records'])
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    rendered = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(rendered + '\n')
    print('saved %d utility labels to %s' %
          (result['sample_count'], args.output), flush=True)


if __name__ == '__main__':
    main()

