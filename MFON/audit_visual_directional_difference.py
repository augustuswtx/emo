#!/usr/bin/env python3
"""Audit a batchwise finite-difference estimate of visual KL update utility.

For one disjoint meta/aux training-batch pair, the meta-task gradient defines
one direction in proj_v.weight. Two reversible forward passes estimate the
directional derivative of *every* auxiliary sample's visual KL at once. The
reference step JSON supplies exact first-order dots and measured one-block
task-loss changes. No checkpoint is written or optimizer update performed.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_task_aligned_gradient import sha256, tensors
from audit_task_aligned_step import task_mse, training_loader
from audit_visual_representation_proxy import check_reference, reference_records
from budgeted_auxiliary import per_sample_kl
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', default='p5_mosei_p4_learned_true_budget')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-pairs', type=int, default=24)
    parser.add_argument('--sample-order-seed', type=int, default=271828)
    parser.add_argument('--exclude-first', type=int, default=256)
    parser.add_argument('--step-ratio', type=float, default=0.0003,
                        help='central-difference displacement / parameter norm')
    parser.add_argument('--cached-visual', action='store_true',
                        help='cache text and recompute only the affected visual branch')
    parser.add_argument('--reference-step-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_pairs < 1 or args.exclude_first < 0:
        parser.error('batch-size and max-pairs must be positive; exclude-first >= 0')
    if not 0 < args.step_ratio <= 0.01:
        parser.error('require 0 < step-ratio <= 0.01')
    args.reference_step_json = args.reference_step_json.resolve()
    args.output = args.output.resolve()
    if not args.reference_step_json.is_file():
        raise FileNotFoundError(args.reference_step_json)
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def cached_visual_embedding(model, last_hidden_text, vision):
    """The visual path of TVA_fusion.forward after text projection."""
    proj_base = model.proj_v(vision).permute(1, 0, 2)
    if model.use_dpg:
        prompt = model.dynamic_prompt_v(
            last_hidden_text, proj_base, model.promptv_m
        )
    else:
        prompt = model.promptv_m.unsqueeze(1)
    projected = proj_base + prompt
    return model.vision_with_text(
        last_hidden_text, projected, projected
    )[0]


def cached_text_embedding(model, text):
    last_hidden = model.text_encoder(text)
    return F.dropout(
        model.proj_t(last_hidden.permute(1, 0, 2)),
        p=model.text_dropout, training=model.training
    )


def central_visual_kl_derivative(model, inputs, target, parameter, direction,
                                 displacement, cached_text=None):
    """Return d KL_i / d displacement for the whole batch and restore weight."""
    if not displacement > 0 or not bool(torch.isfinite(direction).all()):
        raise ValueError('invalid directional probe')
    original = parameter.detach().clone()
    text, vision, audio, _ = inputs

    def visual_kl():
        with torch.no_grad():
            if cached_text is None:
                _, embeddings = model(text, vision, audio, mode='test')
                visual = embeddings[1]
            else:
                visual = cached_visual_embedding(model, cached_text, vision)
            return per_sample_kl(visual, target).detach().clone()

    try:
        with torch.no_grad():
            parameter.add_(direction, alpha=displacement)
        plus = visual_kl()
        with torch.no_grad():
            parameter.copy_(original)
            parameter.add_(direction, alpha=-displacement)
        minus = visual_kl()
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    if not torch.equal(parameter.detach(), original):
        raise RuntimeError('parameter restoration failed')
    return (plus - minus) / (2.0 * displacement), plus, minus


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


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
    ], 'visual directional difference audit')
    checkpoint_hash = sha256(checkpoint)
    with args.reference_step_json.open('r', encoding='utf-8') as handle:
        reference = json.load(handle)
    check_reference(reference, args, checkpoint_hash)
    if 'proj_v.weight' not in reference.get('parameter_blocks', []):
        raise ValueError('reference lacks proj_v.weight')

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
        'design': 'read-only batchwise central directional difference',
        'scope': 'same checkpoint/train batch pairs as reversible step reference',
        'dataset': 'MOSEI', 'split': 'train', 'seed': args.seed,
        'exp_name': args.exp_name, 'checkpoint_sha256': checkpoint_hash,
        'reference_step_json': args.reference_step_json.name,
        'parameter_block': 'proj_v.weight',
        'batch_size': args.batch_size, 'max_pairs': args.max_pairs,
        'sample_order_seed': args.sample_order_seed,
        'exclude_first': args.exclude_first, 'step_ratio': args.step_ratio,
        'cached_visual': args.cached_visual,
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
        sources = reference_records(ref_pair)
        if len(sources) != len(aux_ids):
            raise ValueError('reference lacks visual records in pair %d' % pair_number)

        meta_inputs = tensors(meta_batch, config.DEVICE)
        meta_loss = task_mse(model, meta_inputs)
        meta_gradient = torch.autograd.grad(
            meta_loss, model.proj_v.weight
        )[0].detach()
        meta_norm = float(meta_gradient.norm().item())
        del meta_loss
        if not meta_norm > 1e-12:
            raise ValueError('zero visual meta gradient in pair %d' % pair_number)
        direction = meta_gradient / meta_norm
        parameter = model.proj_v.weight
        displacement = args.step_ratio * float(parameter.detach().norm().item())
        aux_inputs = tensors(aux_batch, config.DEVICE)
        with torch.no_grad():
            target = model.vision_encoder_froze(aux_inputs[1]).detach()
            cached_text = (cached_text_embedding(model, aux_inputs[0])
                           if args.cached_visual else None)
            if args.cached_visual and pair_number == 0:
                _, full_embeddings = model(
                    aux_inputs[0], aux_inputs[1], aux_inputs[2], mode='test'
                )
                cached_embedding = cached_visual_embedding(
                    model, cached_text, aux_inputs[1]
                )
                max_diff = float((full_embeddings[1] - cached_embedding)
                                 .abs().max().item())
                if not torch.allclose(full_embeddings[1], cached_embedding,
                                      atol=1e-5, rtol=1e-5):
                    raise RuntimeError('cached visual path differs from full forward')
                result['cached_full_max_abs_diff'] = max_diff
        directional, plus, minus = central_visual_kl_derivative(
            model, aux_inputs, target, parameter, direction, displacement,
            cached_text=cached_text
        )

        records = []
        exact_matches = actual_matches = valid_count = 0
        for position in range(len(aux_ids)):
            source = sources[position]
            estimate = float(directional[position].item()) * meta_norm
            exact = float(source['meta_gradient_dot_kl_gradient'])
            actual = source.get('actual_minus_change')
            valid = source.get('probe_status') == 'ok'
            match_exact = sign(estimate) == sign(exact) and sign(exact) != 0
            match_actual = (sign(estimate) == sign(-actual)
                            and sign(estimate) != 0) if valid else None
            exact_matches += int(match_exact)
            if valid:
                valid_count += 1
                actual_matches += int(match_actual)
            records.append({
                'sample_position': position,
                'exact_meta_gradient_dot_kl_gradient': exact,
                'directional_dot_estimate': estimate,
                'kl_plus': float(plus[position].item()),
                'kl_minus': float(minus[position].item()),
                'reference_probe_status': source['probe_status'],
                'reference_actual_minus_change': actual,
                'matches_exact_dot_sign': match_exact,
                'matches_actual_step_sign': match_actual,
            })
        result['pairs'].append({
            'pair_number': pair_number, 'meta_indices': meta_ids,
            'aux_indices': aux_ids, 'meta_gradient_norm': meta_norm,
            'displacement': displacement, 'valid_count': valid_count,
            'exact_sign_agreement': exact_matches,
            'actual_sign_agreement': actual_matches, 'records': records,
        })
        print('audited pair %d: exact %d/%d, actual %d/%d' %
              (pair_number, exact_matches, len(aux_ids),
               actual_matches, valid_count), flush=True)

    result['valid_count'] = sum(p['valid_count'] for p in result['pairs'])
    result['exact_sign_agreement'] = sum(
        p['exact_sign_agreement'] for p in result['pairs']
    )
    result['exact_count'] = sum(len(p['records']) for p in result['pairs'])
    result['actual_sign_agreement'] = sum(
        p['actual_sign_agreement'] for p in result['pairs']
    )
    result['elapsed_seconds'] = time.perf_counter() - start
    if torch.cuda.is_available():
        result['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    rendered = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(rendered + '\n')
    print('directional difference agrees with exact dot %d/%d and actual step %d/%d' %
          (result['exact_sign_agreement'], result['exact_count'],
           result['actual_sign_agreement'], result['valid_count']), flush=True)
    print('saved directional audit to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
