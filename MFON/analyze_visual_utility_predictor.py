#!/usr/bin/env python3
"""Fit deterministic linear feasibility baselines to visual utility JSON files.

The split unit is a complete meta/auxiliary pair, never an individual sample. With
--test-input, all primary inputs train the model and all test inputs evaluate it;
otherwise groups are shuffled deterministically and held out within the supplied
files. This is a go/no-go screen for predictability, not the final utility head.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


FEATURE_GROUPS = {
    'q_only': ['old_reliability'],
    'q_kl_alignment': [
        'old_reliability', 'visual_kl', 'student_teacher_cosine',
        'student_norm', 'teacher_norm',
    ],
    'all': None,
    'all_without_q': None,
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--test-input', type=Path, nargs='*', default=[])
    parser.add_argument('--split-seed', type=int, default=20260928)
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0.5 <= args.train_fraction < 1:
        parser.error('train-fraction must be in [0.5, 1)')
    if args.steps < 1 or args.learning_rate <= 0 or args.l2 < 0:
        parser.error('invalid optimization settings')
    for path in args.input + args.test_input:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def load_groups(paths):
    groups = []
    feature_names = None
    sources = []
    for path in paths:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        names = data.get('feature_names')
        if not names or len(names) != len(set(names)):
            raise ValueError('%s has invalid feature names' % path)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError('feature columns differ across files')
        for pair in data.get('records', []):
            samples = pair.get('samples', [])
            if not samples:
                raise ValueError('%s pair has no samples' % path)
            x = np.asarray([
                [sample['features'][name] for name in feature_names]
                for sample in samples
            ], dtype=np.float64)
            utility = np.asarray([sample['utility'] for sample in samples],
                                 dtype=np.float64)
            if not np.isfinite(x).all() or not np.isfinite(utility).all():
                raise ValueError('%s contains nonfinite data' % path)
            groups.append({
                'source': path.name, 'pair_number': pair['pair_number'],
                'x': x, 'utility': utility,
            })
        sources.append({
            'file': path.name, 'seed': data.get('seed'),
            'checkpoint_stage': data.get('checkpoint_stage'),
            'checkpoint_sha256': data.get('checkpoint_sha256'),
            'pairs': len(data.get('records', [])),
        })
    if not groups:
        raise ValueError('no utility records found')
    return groups, feature_names, sources


def join_groups(groups):
    return (np.concatenate([group['x'] for group in groups], axis=0),
            np.concatenate([group['utility'] for group in groups], axis=0))


def split_groups(groups, fraction, seed):
    if len(groups) < 4:
        raise ValueError('need at least four batch-pair groups for holdout')
    order = np.random.RandomState(seed).permutation(len(groups))
    count = min(len(groups) - 1, max(1, int(math.floor(len(groups) * fraction))))
    train_ids = set(int(i) for i in order[:count])
    train = [group for i, group in enumerate(groups) if i in train_ids]
    test = [group for i, group in enumerate(groups) if i not in train_ids]
    return train, test


def sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(x, y, steps, learning_rate, l2):
    y = np.asarray(y, dtype=np.float64)
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError('training labels must contain both utility signs')
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    weight = np.zeros(design.shape[1], dtype=np.float64)
    for step in range(steps):
        probability = sigmoid(design @ weight)
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += l2 * weight[1:]
        rate = learning_rate / math.sqrt(1.0 + step / 250.0)
        weight -= rate * gradient
    return {'mean': mean, 'scale': scale, 'weight': weight}


def predict(model, x):
    standardized = (x - model['mean']) / model['scale']
    design = np.column_stack([np.ones(len(x)), standardized])
    return sigmoid(design @ model['weight'])


def roc_auc(y, score):
    positive = score[y == 1]
    negative = score[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float('nan')
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).sum() + 0.5 * (comparison == 0).sum()) /
                 comparison.size)


def average_precision(y, score):
    positive = int(y.sum())
    if positive == 0:
        return float('nan')
    order = np.argsort(-score, kind='mergesort')
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positive)


def rankdata(value):
    order = np.argsort(value, kind='mergesort')
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left, right):
    a, b = rankdata(left), rankdata(right)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def calibration_error(y, score, bins=10):
    total = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = ((score >= lower) & (score < upper if index < bins - 1 else
                                    score <= upper))
        if mask.any():
            total += mask.mean() * abs(float(score[mask].mean() - y[mask].mean()))
    return float(total)


def evaluate(y, utility, score):
    count = max(1, int(math.ceil(len(y) * 0.25)))
    selected = np.argsort(-score, kind='mergesort')[:count]
    base_harmful = float((utility <= 0).mean())
    selected_harmful = float((utility[selected] <= 0).mean())
    reduction = ((base_harmful - selected_harmful) / base_harmful
                 if base_harmful > 0 else float('nan'))
    return {
        'n': int(len(y)), 'positive': int(y.sum()),
        'positive_prevalence': float(y.mean()),
        'auroc': roc_auc(y, score),
        'auprc': average_precision(y, score),
        'auprc_above_prevalence': average_precision(y, score) - float(y.mean()),
        'spearman_score_utility': spearman(score, utility),
        'brier': float(np.mean((score - y) ** 2)),
        'expected_calibration_error': calibration_error(y, score),
        'top_quarter_count': int(count),
        'base_harmful_rate': base_harmful,
        'top_quarter_harmful_rate': selected_harmful,
        'top_quarter_relative_harmful_reduction': reduction,
    }


def main():
    args = arguments()
    train_groups, feature_names, train_sources = load_groups(args.input)
    if args.test_input:
        test_groups, test_names, test_sources = load_groups(args.test_input)
        if test_names != feature_names:
            raise ValueError('test feature columns differ')
        split = 'explicit held-out input files'
    else:
        train_groups, test_groups = split_groups(
            train_groups, args.train_fraction, args.split_seed
        )
        test_sources = train_sources
        split = 'deterministic batch-pair holdout within input files'
    train_x, train_utility = join_groups(train_groups)
    test_x, test_utility = join_groups(test_groups)
    train_y = (train_utility > 0).astype(np.float64)
    test_y = (test_utility > 0).astype(np.float64)
    if len(np.unique(train_y)) != 2 or len(np.unique(test_y)) != 2:
        raise ValueError('train and test partitions must each contain both utility signs')
    index = {name: position for position, name in enumerate(feature_names)}
    result = {
        'design': 'linear feasibility baseline for sparse visual utility labels',
        'split': split, 'split_seed': args.split_seed,
        'train_sources': train_sources, 'test_sources': test_sources,
        'train_groups': len(train_groups), 'test_groups': len(test_groups),
        'train_samples': len(train_y), 'test_samples': len(test_y),
        'feature_names': feature_names, 'models': {},
    }
    for group_name, names in FEATURE_GROUPS.items():
        if group_name == 'all':
            names = feature_names
        elif group_name == 'all_without_q':
            names = [name for name in feature_names if name != 'old_reliability']
        columns = [index[name] for name in names]
        model = fit_logistic(
            train_x[:, columns], train_y, args.steps,
            args.learning_rate, args.l2
        )
        score = predict(model, test_x[:, columns])
        metrics = evaluate(test_y, test_utility, score)
        metrics['features'] = names
        result['models'][group_name] = metrics
        print('%s: AUROC %.3f AUPRC %.3f rho %.3f top-quarter harmful %.3f' %
              (group_name, metrics['auroc'], metrics['auprc'],
               metrics['spearman_score_utility'],
               metrics['top_quarter_harmful_rate']), flush=True)
    main_metrics = result['models']['all']
    result['internal_gate_b'] = {
        'auroc_at_least_0_70': bool(main_metrics['auroc'] >= 0.70),
        'auprc_margin_at_least_0_10': bool(
            main_metrics['auprc_above_prevalence'] >= 0.10
        ),
        'spearman_at_least_0_25': bool(
            main_metrics['spearman_score_utility'] >= 0.25
        ),
        'top_quarter_harmful_reduction_at_least_0_30': bool(
            main_metrics['top_quarter_relative_harmful_reduction'] >= 0.30
        ),
    }
    result['internal_gate_b']['all_numeric_checks_pass'] = all(
        result['internal_gate_b'].values()
    )
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('saved predictor feasibility analysis to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
