#!/usr/bin/env python3
"""Read-only pilot of parameter-gradient alignment for MOSEI auxiliary KL.

This uses disjoint batches from the *training* split. A meta-batch defines the
primary-task gradient; individual auxiliary KL losses on the following batch
define candidate updates. The dot product is a first-order screening proxy,
not a measured validation/test gain or a full-optimizer guarantee.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from budgeted_auxiliary import per_sample_kl
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-pairs', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.max_pairs, args.max_samples) < 1:
        parser.error('batch-size, max-pairs, and max-samples must be positive')
    if args.max_samples > args.batch_size:
        parser.error('max-samples cannot exceed batch-size')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def tensors(batch, device):
    return (
        batch['raw_text'],
        batch['vision'].detach().to(device).float(),
        batch['audio'].detach().to(device).float(),
        batch['labels']['M'].detach().view(-1).to(device).float(),
    )


def record_one(modality, sample, scores, kl_losses, parameter, meta_gradient):
    sample_gradient = torch.autograd.grad(
        kl_losses[sample], parameter, retain_graph=True
    )[0].detach()
    meta_norm = meta_gradient.norm()
    sample_norm = sample_gradient.norm()
    dot = torch.sum(meta_gradient * sample_gradient)
    denominator = (meta_norm * sample_norm).clamp_min(1e-12)
    return {
        'modality': modality,
        'sample_position': sample,
        'old_reliability_score': float(scores[sample].detach().cpu()),
        'kl': float(kl_losses[sample].detach().cpu()),
        'meta_gradient_dot_kl_gradient': float(dot.cpu()),
        'meta_gradient_cosine': float((dot / denominator).cpu()),
        'meta_gradient_norm': float(meta_norm.cpu()),
        'kl_gradient_norm': float(sample_norm.cpu()),
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
    checkpoint = os.path.join(
        dataset_cfg.path.model_path, str(args.seed), args.exp_name,
        'TVA_fusion_model.pt'
    )
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    require_files([
        raw_data_path, checkpoint,
        os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
        os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
    ], 'task-aligned gradient pilot')
    loader = data_loader.MOSEIDataloader(
        'train', raw_data_path, batch_size=args.batch_size, shuffle=False
    )
    model = tva_train.TVA_fusion(config).to(config.DEVICE)
    model.load_froze()
    state = torch.load(checkpoint, map_location=config.DEVICE)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'checkpoint keys do not exactly match model: missing=%s unexpected=%s'
            % (incompatible.missing_keys, incompatible.unexpected_keys)
        )
    del state
    model.eval()
    if torch.cuda.is_available() and hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = {
        'design': 'read-only first-order screening; disjoint training batches',
        'dataset': 'MOSEI',
        'split': 'train',
        'seed': args.seed,
        'exp_name': args.exp_name,
        'checkpoint_sha256': sha256(checkpoint),
        'parameter_blocks': ['proj_v.weight', 'proj_a.weight'],
        'batch_size': args.batch_size,
        'max_pairs': args.max_pairs,
        'max_samples_per_pair': args.max_samples,
        'pairs': [],
    }
    iterator = iter(loader)
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

        text, vision, audio, labels = tensors(meta_batch, config.DEVICE)
        predictions, _ = model(text, vision, audio, mode='test')
        meta_loss = torch.mean((predictions - labels).pow(2))
        visual_meta_gradient, audio_meta_gradient = torch.autograd.grad(
            meta_loss, (model.proj_v.weight, model.proj_a.weight)
        )
        visual_meta_gradient = visual_meta_gradient.detach()
        audio_meta_gradient = audio_meta_gradient.detach()
        meta_loss_value = float(meta_loss.detach().cpu())
        del predictions, meta_loss, text, vision, audio, labels

        text, vision, audio, _ = tensors(aux_batch, config.DEVICE)
        _, embeddings = model(text, vision, audio, mode='test')
        _, visual_embedding, audio_embedding = embeddings
        with torch.no_grad():
            visual_target = model.vision_encoder_froze(vision)
            audio_target = model.audio_encoder_froze(audio)
            visual_q = model.vision_reliability(vision)
            audio_q = model.audio_reliability(audio)
        visual_kl = per_sample_kl(visual_embedding, visual_target)
        audio_kl = per_sample_kl(audio_embedding, audio_target)
        records = []
        sample_count = min(args.max_samples, len(aux_ids))
        for sample in range(sample_count):
            records.append(record_one(
                'vision', sample, visual_q, visual_kl,
                model.proj_v.weight, visual_meta_gradient
            ))
            records.append(record_one(
                'audio', sample, audio_q, audio_kl,
                model.proj_a.weight, audio_meta_gradient
            ))
        result['pairs'].append({
            'pair_number': pair_number,
            'meta_task_mse': meta_loss_value,
            'meta_count': len(meta_ids),
            'aux_count': len(aux_ids),
            'records': records,
        })
        print('audited pair %d: %d sample-modality records' %
              (pair_number, len(records)), flush=True)
        del visual_kl, audio_kl, embeddings

    result['elapsed_seconds'] = time.perf_counter() - start
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    rendered = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(rendered + '\n')
    print(rendered, flush=True)
    print('saved audit to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
