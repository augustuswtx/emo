#!/usr/bin/env python3
"""Analyze repeated-meta utility stability and fixed predictor baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_visual_utility_predictor import (
    FEATURE_GROUPS,
    evaluate,
    fit_logistic,
    join_groups,
    load_groups,
    predict,
    split_groups,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--split-seed', type=int, default=20261003)
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 0.5 <= args.train_fraction < 1:
        parser.error('train-fraction must be in [0.5, 1)')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def repeated_context_summary(data):
    samples = [sample for group in data['records']
               for sample in group['samples']]
    matrix = np.asarray([sample['context_utilities'] for sample in samples],
                        dtype=np.float64)
    mean = matrix.mean(axis=1)
    within_variance = matrix.var(axis=1)
    between_variance = float(mean.var())
    average_within = float(within_variance.mean())
    denominator = between_variance + average_within
    majority = np.asarray([sample['majority_fraction'] for sample in samples])
    agreement = np.asarray([
        sample['context_sign_agreement'] for sample in samples
    ])
    first_matches = np.asarray([
        sample['first_matches_mean'] for sample in samples
    ], dtype=np.float64)
    return {
        'samples': int(len(samples)),
        'meta_contexts': int(matrix.shape[1]),
        'first_context_matches_mean_rate': float(first_matches.mean()),
        'mean_context_sign_agreement': float(agreement.mean()),
        'median_context_sign_agreement': float(np.median(agreement)),
        'mean_majority_fraction': float(majority.mean()),
        'fraction_majority_at_least_five_sixths': float(
            (majority >= 5.0 / 6.0).mean()
        ),
        'between_sample_mean_utility_variance': between_variance,
        'average_within_sample_context_variance': average_within,
        'context_reliability_ratio': (
            between_variance / denominator if denominator > 0 else 0.0
        ),
    }


def fit_predict_mlp(train_x, train_y, test_x, seed, steps=1000):
    torch.manual_seed(seed)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = torch.tensor((train_x - mean) / scale, dtype=torch.float32)
    test = torch.tensor((test_x - mean) / scale, dtype=torch.float32)
    target = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)
    model = torch.nn.Sequential(
        torch.nn.Linear(train.size(1), 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=1e-3)
    loss_function = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_function(model(train), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(test)).view(-1).numpy().astype(np.float64)


def main():
    args = arguments()
    with args.input.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    stability = repeated_context_summary(raw)
    groups, feature_names, sources = load_groups([args.input])
    train_groups, test_groups = split_groups(
        groups, args.train_fraction, args.split_seed
    )
    train_x, train_utility = join_groups(train_groups)
    test_x, test_utility = join_groups(test_groups)
    train_y = (train_utility > 0).astype(np.float64)
    test_y = (test_utility > 0).astype(np.float64)
    index = {name: position for position, name in enumerate(feature_names)}
    result = {
        'design': 'repeated-meta stability and fixed predictor repair screen',
        'input': args.input.name,
        'split': 'complete auxiliary-group holdout',
        'split_seed': args.split_seed,
        'train_groups': len(train_groups),
        'test_groups': len(test_groups),
        'train_samples': int(len(train_y)),
        'test_samples': int(len(test_y)),
        'sources': sources,
        'stability': stability,
        'models': {},
    }
    for group_name, names in FEATURE_GROUPS.items():
        if group_name == 'all':
            names = feature_names
        elif group_name == 'all_without_q':
            names = [name for name in feature_names
                     if name != 'old_reliability']
        columns = [index[name] for name in names]
        linear = fit_logistic(
            train_x[:, columns], train_y, 2500, 0.05, 1e-3
        )
        score = predict(linear, test_x[:, columns])
        metrics = evaluate(test_y, test_utility, score)
        metrics['features'] = names
        result['models']['linear_' + group_name] = metrics
    for group_name in ('all', 'all_without_q'):
        names = (feature_names if group_name == 'all' else
                 [name for name in feature_names
                  if name != 'old_reliability'])
        columns = [index[name] for name in names]
        score = fit_predict_mlp(
            train_x[:, columns], train_y, test_x[:, columns], args.split_seed
        )
        metrics = evaluate(test_y, test_utility, score)
        metrics['features'] = names
        result['models']['mlp_' + group_name] = metrics
    primary = result['models']['mlp_all']
    checks = {
        'stable_majority_fraction_at_least_0_50': bool(
            stability['fraction_majority_at_least_five_sixths'] >= 0.50
        ),
        'auroc_at_least_0_70': bool(primary['auroc'] >= 0.70),
        'auprc_margin_at_least_0_10': bool(
            primary['auprc_above_prevalence'] >= 0.10
        ),
        'spearman_at_least_0_25': bool(
            primary['spearman_score_utility'] >= 0.25
        ),
        'harmful_reduction_at_least_0_30': bool(
            primary['top_quarter_relative_harmful_reduction'] >= 0.30
        ),
    }
    checks['all_repair_checks_pass'] = all(checks.values())
    result['repair_gate'] = checks
    print('stability: first/mean %.3f, sign agreement %.3f, stable %.3f, '
          'reliability ratio %.3f' % (
              stability['first_context_matches_mean_rate'],
              stability['mean_context_sign_agreement'],
              stability['fraction_majority_at_least_five_sixths'],
              stability['context_reliability_ratio'],
          ), flush=True)
    for name, metrics in result['models'].items():
        print('%s: AUROC %.3f AUPRC %.3f rho %.3f top-quarter harmful %.3f' % (
            name, metrics['auroc'], metrics['auprc'],
            metrics['spearman_score_utility'],
            metrics['top_quarter_harmful_rate'],
        ), flush=True)
    print('repair gate %s' % checks, flush=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('saved stability analysis to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
