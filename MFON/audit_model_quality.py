#!/usr/bin/env python3
"""Audit model-derived quality scores under controlled feature degradation."""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from run_experiment import DATASETS, load_dataset_modules, require_files, resolve_repo_path


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def rankdata(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left, right):
    return pearson(rankdata(left), rankdata(right))


def active_step_mask(features, eps=1e-8):
    if features.ndim != 3:
        raise ValueError('features must have shape [batch, time, feature].')
    return features.abs().sum(dim=-1, keepdim=True) > eps


def active_steps(features):
    return active_step_mask(features).sum(dim=(1, 2)).float()


def active_rms(features, eps=1e-8):
    mask = active_step_mask(features).expand_as(features)
    count = mask.sum(dim=(1, 2)).clamp_min(1)
    squared = (features * mask).pow(2).sum(dim=(1, 2))
    return torch.sqrt(squared / count + eps)


def add_active_gaussian_noise(features, severity, generator):
    if severity < 0:
        raise ValueError('severity must be non-negative.')
    if severity == 0:
        return features.clone()
    mask = active_step_mask(features).expand_as(features)
    scale = active_rms(features).view(-1, 1, 1)
    noise = torch.randn(
        features.shape,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    )
    return features + noise * scale * float(severity) * mask


def parse_severities(raw):
    values = [float(value) for value in raw.split(',')]
    if not values or values[0] != 0.0:
        raise ValueError('severities must start with 0.')
    if any(value < 0 for value in values):
        raise ValueError('severities must be non-negative.')
    if values != sorted(set(values)):
        raise ValueError('severities must be unique and increasing.')
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit align/conf/norm quality under modality-specific Gaussian noise.'
    )
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='MOSI')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument(
        '--q-type', choices=['align', 'norm', 'conf', 'learned'], default='align'
    )
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--severities', default='0,0.25,0.5,1.0')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--noise-seed', type=int, default=20260729)
    return parser.parse_args()


def to_numpy(parts):
    return torch.cat(parts).detach().cpu().numpy()


def evaluate_condition(model, text, vision, audio, learned):
    if learned:
        pred, _ = model(text, vision, audio, mode='test')
        return pred, model.vision_reliability(vision), model.audio_reliability(audio)
    pred, _ = model(text, vision, audio, mode='train', epoch=1)
    budget = model.current_budgeted_aux
    return pred, budget['q_v'], budget['q_a']


def summarize_condition(name, severity, quality, clean_quality, predictions, labels, metrics):
    delta = quality - clean_quality
    result = metrics(labels, predictions)
    task_fields = []
    for key in ('Has0_acc_2', 'Mult_acc_7', 'MAE', 'Corr'):
        if key in result:
            task_fields.append('%s=%.4f' % (key, result[key]))
    fields = [
        '%s severity=%.2f' % (name, severity),
        'q_mean=%.6f' % quality.mean(),
        'q_std=%.6f' % quality.std(),
        'mean_delta=%+.6f' % delta.mean(),
        'fraction_below_clean=%.6f' % np.mean(quality < clean_quality),
    ] + task_fields
    print(' | '.join(fields))


def main():
    args = parse_args()
    severities = parse_severities(args.severities)
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('--max-batches must be positive.')

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    dataset_dir, config, data_loader, utils, tva_train, _, _ = load_dataset_modules(args.dataset)
    dataset_cfg = getattr(config, args.dataset)
    train_cfg = dataset_cfg.downStream.TVAtrain
    config.seed = args.seed
    train_cfg.use_alw = False
    train_cfg.use_budgeted_aux = True
    train_cfg.alw_q_type = 'align' if args.q_type == 'learned' else args.q_type
    train_cfg.use_interventional_reliability = args.q_type == 'learned'
    train_cfg.exp_name = args.exp_name

    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    checkpoint = os.path.join(
        dataset_cfg.path.model_path,
        str(args.seed),
        args.exp_name,
        'TVA_fusion_model.pt',
    )
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    require_files(
        [
            raw_data_path,
            checkpoint,
            os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
            os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
        ],
        'quality audit',
    )
    dataset_cfg.path.raw_data_path = raw_data_path

    loader_cls = getattr(data_loader, DATASETS[args.dataset]['loader'])
    loader = loader_cls(
        args.split,
        raw_data_path,
        batch_size=dataset_cfg.downStream.batch_size,
        shuffle=False,
    )
    model = tva_train.TVA_fusion(config).to(config.DEVICE)
    model.load_froze()
    model.load_model(checkpoint)
    model.eval()

    condition_data = {
        ('clean', 0.0): {'q_v': [], 'q_a': [], 'pred': []},
    }
    for modality in ('vision', 'audio'):
        for severity in severities[1:]:
            condition_data[(modality, severity)] = {'q_v': [], 'q_a': [], 'pred': []}

    labels = []
    vision_lengths = []
    audio_lengths = []
    vision_energy = []
    audio_energy = []
    generator = torch.Generator(device='cpu')
    generator.manual_seed(args.noise_seed)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            text = batch['raw_text']
            vision_cpu = batch['vision'].clone().detach().float()
            audio_cpu = batch['audio'].clone().detach().float()
            label = batch['labels']['M'].clone().detach().view(-1).float()
            labels.append(label)
            vision_lengths.append(active_steps(vision_cpu))
            audio_lengths.append(active_steps(audio_cpu))
            vision_energy.append(active_rms(vision_cpu))
            audio_energy.append(active_rms(audio_cpu))

            vision = vision_cpu.to(config.DEVICE)
            audio = audio_cpu.to(config.DEVICE)
            pred, q_v, q_a = evaluate_condition(
                model, text, vision, audio, args.q_type == 'learned'
            )
            condition_data[('clean', 0.0)]['q_v'].append(q_v.cpu())
            condition_data[('clean', 0.0)]['q_a'].append(q_a.cpu())
            condition_data[('clean', 0.0)]['pred'].append(pred.cpu())

            for severity in severities[1:]:
                noisy_vision = add_active_gaussian_noise(
                    vision_cpu, severity, generator
                ).to(config.DEVICE)
                pred, q_v, q_a = evaluate_condition(
                    model, text, noisy_vision, audio, args.q_type == 'learned'
                )
                condition_data[('vision', severity)]['q_v'].append(q_v.cpu())
                condition_data[('vision', severity)]['q_a'].append(q_a.cpu())
                condition_data[('vision', severity)]['pred'].append(pred.cpu())

                noisy_audio = add_active_gaussian_noise(
                    audio_cpu, severity, generator
                ).to(config.DEVICE)
                pred, q_v, q_a = evaluate_condition(
                    model, text, vision, noisy_audio, args.q_type == 'learned'
                )
                condition_data[('audio', severity)]['q_v'].append(q_v.cpu())
                condition_data[('audio', severity)]['q_a'].append(q_a.cpu())
                condition_data[('audio', severity)]['pred'].append(pred.cpu())

    labels_np = to_numpy(labels)
    clean_q_v = to_numpy(condition_data[('clean', 0.0)]['q_v'])
    clean_q_a = to_numpy(condition_data[('clean', 0.0)]['q_a'])
    clean_pred = to_numpy(condition_data[('clean', 0.0)]['pred'])
    metric_fn = (
        utils.Metrics().eval_sims_regression
        if args.dataset == 'SIMS'
        else utils.Metrics().eval_mosei_regression
    )

    print(
        'Audit %s split=%s seed=%d exp=%s q_type=%s n=%d'
        % (args.dataset, args.split, args.seed, args.exp_name, args.q_type, labels_np.size)
    )
    clean_metrics = metric_fn(
        torch.from_numpy(labels_np), torch.from_numpy(clean_pred)
    )
    print('clean metrics:', clean_metrics)
    print(
        'clean confounds | '
        'corr(q_v,vision_length)=%.6f | corr(q_v,vision_energy)=%.6f | '
        'corr(q_a,audio_length)=%.6f | corr(q_a,audio_energy)=%.6f | '
        'corr(q_v,abs_label)=%.6f | corr(q_a,abs_label)=%.6f'
        % (
            pearson(clean_q_v, to_numpy(vision_lengths)),
            pearson(clean_q_v, to_numpy(vision_energy)),
            pearson(clean_q_a, to_numpy(audio_lengths)),
            pearson(clean_q_a, to_numpy(audio_energy)),
            pearson(clean_q_v, np.abs(labels_np)),
            pearson(clean_q_a, np.abs(labels_np)),
        )
    )

    for modality, clean_quality, quality_key in (
        ('vision', clean_q_v, 'q_v'),
        ('audio', clean_q_a, 'q_a'),
    ):
        severity_axis = []
        quality_axis = []
        highest_quality = None
        for severity in severities[1:]:
            item = condition_data[(modality, severity)]
            quality = to_numpy(item[quality_key])
            predictions = to_numpy(item['pred'])
            summarize_condition(
                modality,
                severity,
                quality,
                clean_quality,
                torch.from_numpy(predictions),
                torch.from_numpy(labels_np),
                metric_fn,
            )
            severity_axis.extend([severity] * quality.size)
            quality_axis.extend(quality.tolist())
            highest_quality = quality

        all_quality = np.concatenate(
            [
                clean_quality,
                *[
                    to_numpy(condition_data[(modality, severity)][quality_key])
                    for severity in severities[1:]
                ],
            ]
        )
        all_severity = np.concatenate(
            [
                np.zeros(clean_quality.size),
                *[
                    np.full(clean_quality.size, severity)
                    for severity in severities[1:]
                ],
            ]
        )
        binary_labels = np.concatenate(
            [
                np.ones(clean_quality.size),
                np.zeros(all_quality.size - clean_quality.size),
            ]
        )
        print(
            '%s audit | spearman(severity,q)=%.6f | clean/corrupt_AUROC=%.6f | '
            'fraction(highest_below_clean)=%.6f'
            % (
                modality,
                spearman(all_severity, all_quality),
                roc_auc_score(binary_labels, all_quality),
                np.mean(highest_quality < clean_quality),
            )
        )

    sys.path.remove(dataset_dir)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print('Error: %s' % exc, file=sys.stderr)
        sys.exit(1)
