#!/usr/bin/env python3
"""Read-only screen of reusing a visual task gradient across auxiliary batches.

At one fixed checkpoint, compare each batch's fresh meta-task gradient with
gradients refreshed every four or eight batches. A cached two-sided visual KL
probe supplies per-sample positive/nonpositive allocation decisions. Agreement
with the fresh decision is an engineering diagnostic, not an actual optimizer
step or evidence of final sentiment performance.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from audit_task_aligned_gradient import sha256, tensors
from audit_visual_directional_difference import (
    cached_text_embedding,
    central_visual_kl_derivative,
)
from audit_visual_online_cost import batch_for_indices, synchronize
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--meta-batch-size', type=int, default=4)
    parser.add_argument('--aux-batch-size', type=int, default=32)
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--sample-order-seed', type=int, default=314159)
    parser.add_argument('--exclude-first', type=int, default=256)
    parser.add_argument('--step-ratio', type=float, default=0.0003)
    parser.add_argument('--exclude-cost-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.meta_batch_size, args.aux_batch_size, args.pairs) < 1:
        parser.error('batch sizes and pairs must be positive')
    if args.exclude_first < 0 or not 0 < args.step_ratio <= 0.01:
        parser.error('require exclude-first >= 0 and 0 < step-ratio <= 0.01')
    args.exclude_cost_json = args.exclude_cost_json.resolve()
    args.output = args.output.resolve()
    if not args.exclude_cost_json.is_file():
        raise FileNotFoundError(args.exclude_cost_json)
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def excluded_indices(path, checkpoint_hash):
    with path.open('r', encoding='utf-8') as handle:
        reference = json.load(handle)
    if (reference.get('dataset') != 'MOSEI'
            or reference.get('split') != 'train'
            or reference.get('checkpoint_sha256') != checkpoint_hash):
        raise ValueError('cost reference dataset, split, or checkpoint differs')
    records = reference.get('records', [])
    if not records:
        raise ValueError('cost reference has no records')
    excluded = set()
    for record in records:
        excluded.update(record['meta_indices'])
        excluded.update(record['aux_indices'])
    return excluded


def counts(fresh_positive, reused_positive):
    if fresh_positive.shape != reused_positive.shape:
        raise ValueError('positive decision shapes differ')
    return {
        'tp': int((fresh_positive & reused_positive).sum().item()),
        'fp': int((~fresh_positive & reused_positive).sum().item()),
        'tn': int((~fresh_positive & ~reused_positive).sum().item()),
        'fn': int((fresh_positive & ~reused_positive).sum().item()),
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
    ], 'visual meta-gradient reuse audit')
    checkpoint_hash = sha256(checkpoint)
    excluded = excluded_indices(args.exclude_cost_json, checkpoint_hash)
    dataset = data_loader.MOSEIDataset('train', raw_data_path)
    indices = [i for i in range(args.exclude_first, len(dataset))
               if i not in excluded]
    needed = args.pairs * (args.meta_batch_size + args.aux_batch_size)
    if len(indices) < needed:
        raise ValueError('not enough unseen disjoint training examples')
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
        'design': 'read-only fixed-checkpoint fresh versus reused meta-gradient screen',
        'scope': 'training data, eval mode; no optimizer steps or saved model',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_sha256': checkpoint_hash,
        'exclude_cost_json': args.exclude_cost_json.name,
        'excluded_index_count': len(excluded),
        'meta_batch_size': args.meta_batch_size,
        'aux_batch_size': args.aux_batch_size, 'pairs': args.pairs,
        'sample_order_seed': args.sample_order_seed,
        'exclude_first': args.exclude_first, 'step_ratio': args.step_ratio,
        'refresh_intervals': [4, 8],
        'records': [],
    }
    anchors = {}
    aggregate = {interval: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                            'nonanchor_pairs': 0}
                 for interval in (4, 8)}
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
        meta_text, meta_vision, meta_audio, meta_label = meta_inputs
        meta_prediction, _ = model(meta_text, meta_vision, meta_audio,
                                   mode='test')
        meta_loss = torch.mean((meta_prediction - meta_label).pow(2))
        gradient = torch.autograd.grad(meta_loss, parameter)[0].detach()
        del meta_prediction, meta_loss
        gradient_norm = float(gradient.norm().item())
        if not gradient_norm > 1e-12:
            raise ValueError('zero meta gradient in pair %d' % pair_number)
        fresh_direction = gradient / gradient_norm
        with torch.no_grad():
            cached_text = cached_text_embedding(model, aux_inputs[0])
            teacher = model.vision_encoder_froze(aux_inputs[1]).detach()
        fresh, _, _ = central_visual_kl_derivative(
            model, aux_inputs, teacher, parameter, fresh_direction,
            displacement, cached_text=cached_text
        )
        fresh_positive = fresh > 0
        record = {
            'pair_number': pair_number, 'meta_indices': meta_ids,
            'aux_indices': aux_ids,
            'fresh_positive_count': int(fresh_positive.sum().item()),
            'fresh_zero_derivative_count': int((fresh == 0).sum().item()),
            'reuse': {},
        }
        for interval in (4, 8):
            anchor = pair_number % interval == 0
            if anchor:
                anchors[interval] = fresh_direction.detach().clone()
                continue
            reused_direction = anchors[interval]
            reused, _, _ = central_visual_kl_derivative(
                model, aux_inputs, teacher, parameter, reused_direction,
                displacement, cached_text=cached_text
            )
            confusion = counts(fresh_positive, reused > 0)
            cosine = float(torch.sum(fresh_direction * reused_direction).item())
            record['reuse'][str(interval)] = {
                'anchor_pair_number': pair_number - pair_number % interval,
                'gradient_cosine': cosine,
                'reused_positive_count': int((reused > 0).sum().item()),
                'confusion': confusion,
            }
            for name in ('tp', 'fp', 'tn', 'fn'):
                aggregate[interval][name] += confusion[name]
            aggregate[interval]['nonanchor_pairs'] += 1
        result['records'].append(record)
        print('pair %d: fresh positive %d/%d, reuse4 %s, reuse8 %s' %
              (pair_number, record['fresh_positive_count'], len(aux_ids),
               str(record['reuse'].get('4', {}).get('confusion', 'anchor')),
               str(record['reuse'].get('8', {}).get('confusion', 'anchor'))),
              flush=True)
        del gradient, fresh, cached_text, teacher
    synchronize()
    result['elapsed_seconds'] = time.perf_counter() - start
    result['aggregate_nonanchor'] = {str(k): v for k, v in aggregate.items()}
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for interval, values in aggregate.items():
        total = sum(values[name] for name in ('tp', 'fp', 'tn', 'fn'))
        correct = values['tp'] + values['tn']
        print('reuse every %d: %d/%d same positive decisions on nonanchor pairs'
              % (interval, correct, total), flush=True)
    print('saved meta reuse audit to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
