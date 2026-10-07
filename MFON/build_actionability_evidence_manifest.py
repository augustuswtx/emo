#!/usr/bin/env python3
"""Build a source-linked, read-only evidence manifest for the actionability audit.

This script does not compute new experimental results. It normalizes existing
summary files and records missing primary artifacts explicitly. It uses only the
Python standard library so it can run on the desktop or the GPU server.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 'actionability-evidence/1.0'
EXPECTED_SOURCES = (
    'f3', 'f4', 'c1', 'c2', 'c2_primary', 'c5', 'main',
    'p6', 'p6_records', 'p7', 'p7_records',
)
INVENTORY_ONLY = ('c2_primary', 'p6_records', 'p7_records')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError('%s must be a finite number' % field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be a finite number' % field)
    if not math.isfinite(number):
        raise ValueError('%s must be a finite number' % field)
    return number


def read_json(path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_csv(path):
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def source_entry(path, kind, role, expected_path):
    return {
        'kind': kind,
        'status': 'present' if path else 'missing',
        'path': str(path) if path else None,
        'expected_path': expected_path,
        'sha256': sha256(path) if path else None,
        'role': role,
    }


def metric_record(source, level, dataset, split, seed, score, modality,
                  metric, value, n=None, analysis_status='exploratory',
                  comparator=None, checkpoint_sha256=None, note=None,
                  condition=None, sd=None):
    result = {
        'source': source,
        'level': level,
        'dataset': dataset,
        'split': split,
        'seed': int(seed) if seed is not None else None,
        'score': score,
        'modality': modality,
        'metric': metric,
        'value': finite_number(value, metric),
        'sd': finite_number(sd, metric + '_sd') if sd is not None else None,
        'n': int(n) if n is not None else None,
        'condition': condition or {},
        'analysis_status': analysis_status,
        'comparator': comparator,
        'checkpoint_sha256': checkpoint_sha256,
        'note': note,
    }
    if result['n'] is not None and result['n'] < 1:
        raise ValueError('n must be positive')
    return result


def import_c1(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError('empty C1 CSV')
    results = []
    for row in rows:
        modality = row['modality']
        if modality not in ('vision', 'audio'):
            raise ValueError('invalid C1 modality %s' % modality)
        for column in (
            'global_spearman_score_fidelity',
            'partial_spearman_length_energy',
            'pairwise_concordance',
            'within_weighted_proxy_spearman_mean',
            'corr_score_length',
            'corr_score_energy',
        ):
            results.append(metric_record(
                'c1', ('score_confound' if column.startswith('corr_score_')
                       else 'target_fidelity'), 'MOSEI', 'valid', row['seed'],
                'learned_reliability', modality, column, row[column], row['n'],
                note=('KL is cross-sample comparable; weighted proxy is within-batch only'
                      if column == 'within_weighted_proxy_spearman_mean' else None),
                condition=({'comparison_scope': 'within_batch'}
                           if column == 'within_weighted_proxy_spearman_mean'
                           else {'comparison_scope': 'global'}),
                sd=(row['within_weighted_proxy_spearman_sd']
                    if column == 'within_weighted_proxy_spearman_mean' else None),
            ))
    return results


def import_f3(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError('empty F3 CSV')
    results = []
    for row in rows:
        if row['section'] not in ('audit', 'confound'):
            raise ValueError('unexpected F3 section')
        modality = row['modality'].lower()
        if modality not in ('vision', 'audio'):
            raise ValueError('invalid F3 modality')
        results.append(metric_record(
            'f3', ('degradation_sensitivity' if row['section'] == 'audit'
                   else 'score_confound'),
            'MOSEI', 'test', None, 'learned_reliability',
            modality, row['metric'], row['mean'],
            analysis_status='three_seed_aggregate',
            sd=row['sd'],
            condition={'corruption': 'gaussian'} if row['section'] == 'audit' else {},
            note='mean over three frozen seeds; 4659 test samples per seed',
        ))
    return results


def import_f4(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError('empty F4 CSV')
    results = []
    for row in rows:
        modality = row['modality'].lower()
        if modality not in ('vision', 'audio'):
            raise ValueError('invalid F4 modality')
        for column, level in (
            ('auroc', 'degradation_sensitivity'),
            ('delta_mae', 'task_impact'),
        ):
            results.append(metric_record(
                'f4', level, 'MOSEI', 'test', None,
                'learned_reliability', modality,
                row['corruption'] + '_' + column, row[column],
                analysis_status='three_seed_aggregate',
                sd=row[column + '_sd'],
                condition={'corruption': row['corruption'],
                           'severity': finite_number(row['severity'], 'severity')},
                note='mean over three frozen seeds; 4659 test samples per seed',
            ))
    return results


def import_c2(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError('empty C2 CSV')
    results = []
    for row in rows:
        if row['allocation'] not in ('constant', 'learned', 'permuted', 'inverse'):
            raise ValueError('invalid C2 allocation')
        for column in ('mae', 'corr', 'loss'):
            results.append(metric_record(
                'c2', 'allocation_actionability', 'MOSEI', 'test',
                row['seed'], 'learned_reliability', 'vision_audio',
                column, row[column],
                analysis_status='single_seed_transcribed_summary',
                comparator=row['allocation'],
                note='primary train/test logs and some checkpoint hashes not in repository',
            ))
    return results


def import_c5(path):
    data = read_json(path)
    if data.get('design') != 'post-hoc exploratory; baseline-defined modality-utility thirds':
        raise ValueError('unexpected C5 design')
    results = []
    seed = data['seed']
    for row in data['strata']:
        modality = row['modality']
        stratum = row['stratum']
        if modality not in ('vision', 'audio') or stratum not in ('low', 'middle', 'high'):
            raise ValueError('invalid C5 stratum')
        for metric in ('MAE', 'Corr', 'Loss'):
            results.append(metric_record(
                'c5', 'modality_sensitivity', data['dataset'], data['split'],
                seed, 'baseline_zero_ablation_thirds', modality,
                'learned_minus_constant_' + metric, row['learned_minus_constant'][metric],
                row['n'], comparator='learned_minus_constant',
                checkpoint_sha256=data['checkpoint_sha256'].get('learned'),
                condition={'stratum': stratum},
                note='post-hoc zero-ablation proxy',
            ))
    return results


def import_main(path):
    rows = read_csv(path)
    if not rows:
        raise ValueError('empty main-result CSV')
    results = []
    for row in rows:
        if row['metric'] not in ('MAE', 'Corr', 'Has0 Acc-2', 'Non0 Acc-2'):
            continue
        results.append(metric_record(
            'main', 'final_performance', 'MOSEI', 'test', row['seed'],
            row['method'], None, row['metric'], row['value'],
            analysis_status='frozen_result_table',
            note='Frozen manuscript table value; per-sample predictions are not in this manifest',
        ))
    return results


def import_p6(path):
    data = read_json(path)
    if data.get('design') != 'linear feasibility baseline for sparse visual utility labels':
        raise ValueError('unexpected P6 design')
    sources = data.get('test_sources', [])
    if len(sources) != 1 or sources[0].get('seed') is None or not sources[0].get('checkpoint_sha256'):
        raise ValueError('P6 importer requires one identified test source')
    seed = sources[0]['seed']
    checkpoint = sources[0]['checkpoint_sha256']
    results = []
    for model, metrics in data['models'].items():
        for metric in (
            'auroc', 'auprc', 'spearman_score_utility',
            'top_quarter_harmful_rate', 'positive_prevalence',
        ):
            results.append(metric_record(
                'p6', 'utility_predictability', 'MOSEI', 'train', seed,
                model, 'vision', metric, metrics[metric], metrics['n'],
                checkpoint_sha256=checkpoint,
                note=data.get('split'),
            ))
    return results


def import_p7(path):
    data = read_json(path)
    if data.get('design') != 'repeated-meta stability and fixed predictor repair screen':
        raise ValueError('unexpected P7 design')
    sources = data.get('sources', [])
    if len(sources) != 1 or sources[0].get('seed') is None or not sources[0].get('checkpoint_sha256'):
        raise ValueError('P7 importer requires one identified source')
    seed = sources[0]['seed']
    checkpoint = sources[0]['checkpoint_sha256']
    results = []
    stability = data['stability']
    for metric in (
        'first_context_matches_mean_rate',
        'mean_context_sign_agreement',
        'fraction_majority_at_least_five_sixths',
        'context_reliability_ratio',
    ):
        results.append(metric_record(
            'p7', 'update_stability', 'MOSEI', 'train', seed,
            'local_visual_gradient_utility', 'vision', metric,
            stability[metric], stability['samples'],
            checkpoint_sha256=checkpoint,
            note='six independent meta contexts; final checkpoint',
        ))
    for model, metrics in data['models'].items():
        for metric in (
            'auroc', 'auprc', 'spearman_score_utility',
            'top_quarter_harmful_rate',
        ):
            results.append(metric_record(
                'p7', 'utility_predictability', 'MOSEI', 'train', seed,
                model, 'vision', metric, metrics[metric], metrics['n'],
                checkpoint_sha256=checkpoint,
                note='mean utility across six meta contexts; held-out auxiliary groups',
            ))
    return results


def check_raw_link(summary_path, raw_path, source_key):
    """Check a summary names the supplied label file and agrees on seed/checkpoint."""
    if not summary_path or not raw_path:
        return None
    summary = read_json(summary_path)
    raw = read_json(raw_path)
    sources = summary['test_sources'] if source_key == 'p6' else summary['sources']
    matches = [item for item in sources if item['file'] == raw_path.name]
    if len(matches) != 1:
        raise ValueError('%s summary does not uniquely name raw file %s' %
                         (source_key, raw_path.name))
    item = matches[0]
    if (item.get('seed') != raw.get('seed') or
            item.get('checkpoint_sha256') != raw.get('checkpoint_sha256')):
        raise ValueError('%s raw file seed/checkpoint disagrees with summary' % source_key)
    count = sum(len(group['samples']) for group in raw['records'])
    if source_key == 'p6' and len(sources) == 1:
        if count != summary['train_samples'] + summary['test_samples']:
            raise ValueError('P6 raw sample count disagrees with train/test summary')
    if source_key == 'p7' and count != summary['stability']['samples']:
        raise ValueError('P7 raw sample count disagrees with stability summary')
    return {'raw_sample_count': count, 'raw_group_count': len(raw['records'])}


IMPORTERS = {
    'f3': import_f3,
    'f4': import_f4,
    'c1': import_c1,
    'c2': import_c2,
    'c5': import_c5,
    'main': import_main,
    'p6': import_p6,
    'p7': import_p7,
}


def build_manifest(paths):
    sources = {}
    evidence = []
    for kind in EXPECTED_SOURCES:
        path = paths.get(kind)
        source_role = ('source summary' if kind in ('c5', 'p6', 'p7')
                       else 'raw utility labels' if kind in ('p6_records', 'p7_records')
                       else 'derived table' if kind != 'c2_primary'
                       else 'primary C2 artifact inventory; metrics not yet parsed')
        expected_path = ('primary C2 logs or JSON required; no importer until acquired'
                         if kind == 'c2_primary' else 'supply with --' + kind)
        sources[kind] = source_entry(path, kind, source_role, expected_path)
        if path and kind not in INVENTORY_ONLY:
            evidence.extend(IMPORTERS[kind](path))
    for kind in ('p6', 'p7'):
        link = check_raw_link(paths.get(kind), paths.get(kind + '_records'), kind)
        if link:
            sources[kind + '_records']['summary_link_check'] = link
    levels = {
        level: sorted({row['source'] for row in evidence if row['level'] == level})
        for level in (
            'degradation_sensitivity', 'target_fidelity',
            'score_confound', 'task_impact', 'modality_sensitivity',
            'update_stability',
            'utility_predictability', 'allocation_actionability',
            'final_performance',
        )
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'scope': 'source-linked summaries; missing sources are not imputed',
        'sources': sources,
        'coverage_by_level': levels,
        'evidence': evidence,
        'limitations': [
            'C2 Permuted/Inverse metrics are transcribed summaries; '
            'primary logs or JSON are required for independent verification.',
            'C1 and C5 are post-hoc exploratory analyses.',
            'F3/F4 are rounded three-seed aggregates without per-seed raw records here.',
            'P6/P7 only concern visual utility at one final checkpoint and one seed.',
            'P6/P7 summary metrics cannot be recomputed without their raw utility-label JSON files.',
            'Main CSV contains rounded aggregate table values, not per-sample predictions.',
            'No field-level generalization is inferred from this manifest.',
        ],
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f3', type=Path)
    parser.add_argument('--f4', type=Path)
    parser.add_argument('--c1', type=Path)
    parser.add_argument('--c2', type=Path)
    parser.add_argument('--c2-primary', dest='c2_primary', type=Path,
                        help='Inventory a primary C2 log/archive; metrics are not imported')
    parser.add_argument('--c5', type=Path)
    parser.add_argument('--main', type=Path)
    parser.add_argument('--p6', type=Path)
    parser.add_argument('--p6-records', dest='p6_records', type=Path)
    parser.add_argument('--p7', type=Path)
    parser.add_argument('--p7-records', dest='p7_records', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {kind: getattr(args, kind, None) for kind in EXPECTED_SOURCES}
    for kind, path in paths.items():
        if path is not None and not path.is_file():
            parser.error('%s file is missing: %s' % (kind, path))
    if args.output.exists():
        parser.error('output already exists: %s' % args.output)
    return paths, args.output


def main():
    paths, output = arguments()
    manifest = build_manifest(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print('wrote %d evidence rows to %s' % (len(manifest['evidence']), output))
    print('present=%s missing=%s' % (
        ','.join(kind for kind, source in manifest['sources'].items()
                 if source['status'] == 'present'),
        ','.join(kind for kind, source in manifest['sources'].items()
                 if source['status'] == 'missing'),
    ))


if __name__ == '__main__':
    main()
