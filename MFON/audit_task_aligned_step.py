#!/usr/bin/env python3
"""Probe actual task-loss changes after reversible single-block KL steps.

For each selected sample, this program changes only proj_v.weight or
proj_a.weight in memory, evaluates a disjoint training meta-batch, then restores
the parameter exactly. The checkpoint file is never modified. The step is a
controlled SGD-style perturbation, not a reconstruction of Adam's training
update; optimizer state is absent from the saved MFON checkpoint.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from audit_task_aligned_gradient import sha256, tensors
from budgeted_auxiliary import per_sample_kl
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-pairs', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=4)
    parser.add_argument('--step-ratio', type=float, default=0.001,
                        help='L2 step norm as a fraction of parameter L2 norm')
    parser.add_argument('--sample-order-seed', type=int, default=None,
                        help='seeded random order of train samples; default is original order')
    parser.add_argument('--exclude-first', type=int, default=0,
                        help='exclude this many initial train indices before sampling')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.max_pairs, args.max_samples) < 1:
        parser.error('batch-size, max-pairs, and max-samples must be positive')
    if args.max_samples > args.batch_size or not 0 < args.step_ratio <= 0.01:
        parser.error('require max-samples <= batch-size and 0 < step-ratio <= 0.01')
    if args.exclude_first < 0:
        parser.error('exclude-first must be nonnegative')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def task_mse(model, inputs):
    text, vision, audio, labels = inputs
    predictions, _ = model(text, vision, audio, mode='test')
    return torch.mean((predictions - labels).pow(2))


def training_loader(data_loader, raw_data_path, batch_size, sample_order_seed,
                    exclude_first=0):
    if sample_order_seed is None and exclude_first == 0:
        return data_loader.MOSEIDataloader(
            'train', raw_data_path, batch_size=batch_size, shuffle=False
        )
    dataset = data_loader.MOSEIDataset('train', raw_data_path)
    if not 0 <= exclude_first < len(dataset):
        raise ValueError('exclude-first must leave at least one train sample')
    indices = list(range(exclude_first, len(dataset)))
    if sample_order_seed is not None:
        generator = torch.Generator().manual_seed(sample_order_seed)
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[position] for position in order]
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, pin_memory=True
    )


def probe_one(model, inputs, parameter, gradient, dot, baseline, step_ratio):
    """Return both directional losses and restore one parameter in all cases."""
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError('nonfinite auxiliary gradient')
    original = parameter.detach().clone()
    parameter_norm = float(original.norm().item())
    gradient_norm = float(gradient.norm().item())
    if gradient_norm <= 1e-12:
        return {'probe_status': 'zero_gradient',
                'parameter_norm': parameter_norm,
                'gradient_norm': gradient_norm}
    step_norm = step_ratio * max(parameter_norm, 1e-12)
    scale = step_norm / gradient_norm
    try:
        with torch.no_grad():
            parameter.add_(gradient, alpha=-scale)
            loss_minus = float(task_mse(model, inputs).item())
            parameter.copy_(original)
            parameter.add_(gradient, alpha=scale)
            loss_plus = float(task_mse(model, inputs).item())
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    if not torch.equal(parameter.detach(), original):
        raise RuntimeError('parameter restoration failed')
    return {
        'probe_status': 'ok',
        'parameter_norm': parameter_norm,
        'gradient_norm': gradient_norm,
        'step_norm': step_norm,
        'gradient_scale': scale,
        'predicted_minus_change': -scale * dot,
        'actual_minus_change': loss_minus - baseline,
        'central_minus_change': (loss_minus - loss_plus) / 2.0,
        'loss_after_minus_step': loss_minus,
        'loss_after_plus_step': loss_plus,
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
    ], 'task-aligned finite-step pilot')
    loader = training_loader(
        data_loader, raw_data_path, args.batch_size, args.sample_order_seed,
        args.exclude_first
    )
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

    result = {
        'design': 'reversible single-parameter-block SGD-style perturbation',
        'scope': 'disjoint training batches; not Adam replay or held-out validation',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_sha256': sha256(checkpoint),
        'parameter_blocks': ['proj_v.weight', 'proj_a.weight'],
        'batch_size': args.batch_size, 'max_pairs': args.max_pairs,
        'max_samples_per_pair': args.max_samples,
        'step_ratio': args.step_ratio,
        'sample_order_seed': args.sample_order_seed,
        'exclude_first': args.exclude_first,
        'pairs': [],
    }
    iterator = iter(loader)
    start = time.perf_counter()
    for pair_number in range(args.max_pairs):
        try:
            meta_batch = next(iterator)
            aux_batch = next(iterator)
        except StopIteration:
            raise ValueError('not enough disjoint training batches')
        meta_ids = set(meta_batch['index'].detach().cpu().view(-1).tolist())
        aux_ids = set(aux_batch['index'].detach().cpu().view(-1).tolist())
        if meta_ids & aux_ids:
            raise ValueError('meta and auxiliary batches overlap')
        meta_inputs = tensors(meta_batch, config.DEVICE)
        meta_loss = task_mse(model, meta_inputs)
        meta_gradients = torch.autograd.grad(
            meta_loss, (model.proj_v.weight, model.proj_a.weight)
        )
        meta_gradients = [gradient.detach() for gradient in meta_gradients]
        baseline = float(meta_loss.detach().item())
        del meta_loss

        text, vision, audio, _ = tensors(aux_batch, config.DEVICE)
        _, embeddings = model(text, vision, audio, mode='test')
        _, visual_embedding, audio_embedding = embeddings
        with torch.no_grad():
            targets = (model.vision_encoder_froze(vision),
                       model.audio_encoder_froze(audio))
            scores = (model.vision_reliability(vision),
                      model.audio_reliability(audio))
        kl_losses = (per_sample_kl(visual_embedding, targets[0]),
                     per_sample_kl(audio_embedding, targets[1]))
        parameters = (model.proj_v.weight, model.proj_a.weight)
        names = ('vision', 'audio')
        requests = []
        records = []
        for sample in range(min(args.max_samples, len(aux_ids))):
            for modality_index, name in enumerate(names):
                gradient = torch.autograd.grad(
                    kl_losses[modality_index][sample],
                    parameters[modality_index], retain_graph=True
                )[0].detach()
                dot = float(torch.sum(meta_gradients[modality_index] * gradient).item())
                denominator = (meta_gradients[modality_index].norm()
                               * gradient.norm()).clamp_min(1e-12)
                cosine = float((torch.sum(meta_gradients[modality_index] * gradient)
                                / denominator).item())
                record = {
                    'modality': name, 'sample_position': sample,
                    'old_reliability_score': float(scores[modality_index][sample].item()),
                    'kl': float(kl_losses[modality_index][sample].detach().item()),
                    'meta_gradient_dot_kl_gradient': dot,
                    'meta_gradient_cosine': cosine,
                    'meta_gradient_norm': float(meta_gradients[modality_index].norm().item()),
                    'kl_gradient_norm': float(gradient.norm().item()),
                }
                records.append(record)
                requests.append((record, parameters[modality_index], gradient, dot))
        del kl_losses, embeddings, visual_embedding, audio_embedding
        with torch.no_grad():
            baseline_repeat = float(task_mse(model, meta_inputs).item())
        if abs(baseline_repeat - baseline) > max(1e-6, abs(baseline) * 1e-5):
            raise RuntimeError('meta baseline changed before perturbation')
        for record, parameter, gradient, dot in requests:
            record.update(probe_one(model, meta_inputs, parameter, gradient,
                                    dot, baseline_repeat, args.step_ratio))
        result['pairs'].append({
            'pair_number': pair_number, 'meta_indices': sorted(meta_ids),
            'aux_indices': sorted(aux_ids), 'baseline_meta_task_mse': baseline_repeat,
            'records': records,
        })
        print('probed pair %d: %d sample-modality steps' %
              (pair_number, len(records)), flush=True)

    result['elapsed_seconds'] = time.perf_counter() - start
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    rendered = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(rendered + '\n')
    print(rendered, flush=True)
    print('saved step audit to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
