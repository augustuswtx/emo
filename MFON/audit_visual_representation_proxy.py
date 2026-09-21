#!/usr/bin/env python3
"""Compare a cheap visual representation-gradient proxy with step probes.

This read-only audit uses the same sequential disjoint training batches as a
previous audit_task_aligned_step.py run. It compares a meta-task gradient at
the visual embedding with the analytic KL gradient of each auxiliary sample
at that embedding. The reference JSON supplies the measured single-block
finite-step loss changes. A match is exploratory evidence for a cheaper
proxy, not a guarantee about parameter updates or full training.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_task_aligned_gradient import sha256, tensors
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-pairs', type=int, default=8)
    parser.add_argument('--reference-step-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_pairs < 1:
        parser.error('batch-size and max-pairs must be positive')
    args.reference_step_json = args.reference_step_json.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.reference_step_json.is_file():
        raise FileNotFoundError(args.reference_step_json)
    return args


def visual_kl_representation_gradient(student, teacher):
    """Analytic d KL(teacher || student) / d student for each row."""
    if student.ndim != 2 or teacher.shape != student.shape:
        raise ValueError('student and teacher must have the same [batch, dim] shape')
    return F.softmax(student, dim=-1) - F.softmax(teacher, dim=-1)


def reference_records(pair):
    visual = [item for item in pair['records'] if item['modality'] == 'vision']
    by_position = {item['sample_position']: item for item in visual}
    if len(by_position) != len(visual):
        raise ValueError('duplicate visual sample position in reference')
    return by_position


def check_reference(reference, args, checkpoint_hash):
    if reference.get('dataset') != 'MOSEI' or reference.get('split') != 'train':
        raise ValueError('reference must use MOSEI train split')
    if reference.get('seed') != args.seed or reference.get('exp_name') != args.exp_name:
        raise ValueError('reference seed or experiment does not match')
    if reference.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('reference checkpoint hash does not match')
    if reference.get('batch_size') != args.batch_size:
        raise ValueError('reference batch size does not match')
    if len(reference.get('pairs', [])) < args.max_pairs:
        raise ValueError('reference has fewer pairs than requested')


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
    ], 'visual representation proxy audit')
    checkpoint_hash = sha256(checkpoint)
    with args.reference_step_json.open('r', encoding='utf-8') as handle:
        reference = json.load(handle)
    check_reference(reference, args, checkpoint_hash)

    loader = data_loader.MOSEIDataloader(
        'train', raw_data_path, batch_size=args.batch_size, shuffle=False
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
        'design': 'read-only representation proxy versus measured visual step',
        'scope': 'same checkpoint and sequential training batches as reference',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_sha256': checkpoint_hash,
        'reference_step_json': args.reference_step_json.name,
        'batch_size': args.batch_size, 'max_pairs': args.max_pairs,
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
        meta_ids = sorted(meta_batch['index'].detach().cpu().view(-1).tolist())
        aux_ids = sorted(aux_batch['index'].detach().cpu().view(-1).tolist())
        ref_pair = reference['pairs'][pair_number]
        if (ref_pair['pair_number'] != pair_number
                or meta_ids != ref_pair['meta_indices']
                or aux_ids != ref_pair['aux_indices']):
            raise ValueError('reference batch indices do not match pair %d' % pair_number)
        if set(meta_ids) & set(aux_ids):
            raise ValueError('meta and auxiliary batches overlap')

        meta_text, meta_vision, meta_audio, meta_labels = tensors(
            meta_batch, config.DEVICE
        )
        predictions, meta_embeddings = model(
            meta_text, meta_vision, meta_audio, mode='test'
        )
        meta_loss = torch.mean((predictions - meta_labels).pow(2))
        visual_meta_gradient = torch.autograd.grad(
            meta_loss, meta_embeddings[1]
        )[0].detach().sum(dim=0)
        del predictions, meta_loss, meta_embeddings

        aux_text, aux_vision, aux_audio, _ = tensors(aux_batch, config.DEVICE)
        with torch.no_grad():
            _, aux_embeddings = model(aux_text, aux_vision, aux_audio,
                                      mode='test')
            visual_student = aux_embeddings[1]
            visual_teacher = model.vision_encoder_froze(aux_vision)
            kl_gradient = visual_kl_representation_gradient(
                visual_student, visual_teacher
            )
        reference_by_position = reference_records(ref_pair)
        if len(reference_by_position) != len(aux_ids):
            raise ValueError('reference does not contain every visual sample')
        meta_norm = visual_meta_gradient.norm()
        records = []
        valid_agreement = 0
        valid_count = 0
        for position in range(len(aux_ids)):
            source = reference_by_position[position]
            gradient = kl_gradient[position]
            dot = torch.sum(visual_meta_gradient * gradient)
            cosine = dot / (meta_norm * gradient.norm()).clamp_min(1e-12)
            predicted_sign = 1 if float(dot.item()) > 0 else -1 if float(dot.item()) < 0 else 0
            source_ok = source['probe_status'] == 'ok'
            actual_change = source.get('actual_minus_change')
            actual_helpful = actual_change < 0 if source_ok else None
            agrees = (predicted_sign != 0 and
                      ((predicted_sign > 0) == actual_helpful)) if source_ok else None
            if source_ok:
                valid_count += 1
                valid_agreement += int(agrees)
            records.append({
                'sample_position': position,
                'old_reliability_score': source['old_reliability_score'],
                'representation_dot': float(dot.item()),
                'representation_cosine': float(cosine.item()),
                'representation_meta_gradient_norm': float(meta_norm.item()),
                'representation_kl_gradient_norm': float(gradient.norm().item()),
                'reference_probe_status': source['probe_status'],
                'reference_actual_minus_change': actual_change,
                'proxy_predicts_helpful': predicted_sign > 0 if predicted_sign else None,
                'reference_actual_helpful': actual_helpful,
                'proxy_matches_actual': agrees,
            })
        result['pairs'].append({
            'pair_number': pair_number, 'meta_indices': meta_ids,
            'aux_indices': aux_ids, 'valid_count': valid_count,
            'proxy_actual_agreement': valid_agreement,
            'records': records,
        })
        print('audited pair %d: proxy agrees %d/%d' %
              (pair_number, valid_agreement, valid_count), flush=True)

    result['valid_count'] = sum(pair['valid_count'] for pair in result['pairs'])
    result['proxy_actual_agreement'] = sum(
        pair['proxy_actual_agreement'] for pair in result['pairs']
    )
    result['elapsed_seconds'] = time.perf_counter() - start
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    rendered = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(rendered + '\n')
    print('representation proxy agrees %d/%d visual steps' %
          (result['proxy_actual_agreement'], result['valid_count']), flush=True)
    print('saved proxy audit to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
