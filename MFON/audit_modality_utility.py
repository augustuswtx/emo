#!/usr/bin/env python3
"""Post-hoc, read-only MOSEI zero-ablation sensitivity strata.

The repaired MFON checkpoint defines each sample's visual/acoustic sensitivity
as the increase in absolute prediction error when that modality's features are
set to zero. This is a perturbation proxy, not causal or deployment utility. Low,
middle, and high thirds are fixed from the baseline before inspecting the P4
comparisons. Equal utility ties are ordered by dataset index for deterministic
group sizes; a boundary-tie flag is reported for interpretation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from modality_utility_stats import regression_metrics, thirds_by_baseline_utility
from run_experiment import load_dataset_modules, require_files, resolve_repo_path


DEFAULT_EXPERIMENTS = {
    'baseline': 'p5_mosei_repaired_baseline',
    'constant': 'p5_mosei_p4_constant_true_budget',
    'learned': 'p5_mosei_p4_learned_true_budget',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--baseline-exp-name', default=DEFAULT_EXPERIMENTS['baseline'])
    parser.add_argument('--constant-exp-name', default=DEFAULT_EXPERIMENTS['constant'])
    parser.add_argument('--learned-exp-name', default=DEFAULT_EXPERIMENTS['learned'])
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional new JSON output; an existing file is never overwritten.')
    return parser.parse_args()


def collect_predictions(model, loader, device, max_batches, baseline=False):
    fields = {key: [] for key in ('index', 'label', 'clean')}
    if baseline:
        fields['without_vision'] = []
        fields['without_audio'] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            text = batch['raw_text']
            vision = batch['vision'].clone().detach().to(device).float()
            audio = batch['audio'].clone().detach().to(device).float()
            label = batch['labels']['M'].clone().detach().view(-1).float()
            clean, _ = model(text, vision, audio, mode='test')
            fields['index'].append(batch['index'].detach().cpu().view(-1))
            fields['label'].append(label.cpu())
            fields['clean'].append(clean.detach().cpu().view(-1))
            if baseline:
                without_vision, _ = model(
                    text, torch.zeros_like(vision), audio, mode='test'
                )
                without_audio, _ = model(
                    text, vision, torch.zeros_like(audio), mode='test'
                )
                fields['without_vision'].append(without_vision.detach().cpu().view(-1))
                fields['without_audio'].append(without_audio.detach().cpu().view(-1))
    if not fields['index']:
        raise ValueError('audit loader returned no samples')
    return {key: torch.cat(parts).numpy() for key, parts in fields.items()}


def checkpoint_path(dataset_cfg, seed, exp_name):
    return os.path.join(dataset_cfg.path.model_path, str(seed), exp_name,
                        'TVA_fusion_model.pt')


def main():
    args = parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('--max-batches must be positive')
    if args.output is not None:
        args.output = args.output.resolve()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    _, config, data_loader, _, tva_train, _, _ = load_dataset_modules('MOSEI')
    dataset_cfg = config.MOSEI
    train_cfg = dataset_cfg.downStream.TVAtrain
    config.seed = args.seed
    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    experiments = {
        'baseline': args.baseline_exp_name,
        'constant': args.constant_exp_name,
        'learned': args.learned_exp_name,
    }
    checkpoints = {
        name: checkpoint_path(dataset_cfg, args.seed, exp_name)
        for name, exp_name in experiments.items()
    }
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    require_files([raw_data_path] + list(checkpoints.values()) + [
        os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
        os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
    ],
                  'modality-utility audit')
    dataset_cfg.path.raw_data_path = raw_data_path
    loader = data_loader.MOSEIDataloader(
        args.split, raw_data_path,
        batch_size=dataset_cfg.downStream.batch_size, shuffle=False
    )

    predictions = {}
    checkpoint_hashes = {}
    for name in ('baseline', 'constant', 'learned'):
        train_cfg.exp_name = experiments[name]
        train_cfg.use_alw = False
        train_cfg.use_budgeted_aux = name != 'baseline'
        train_cfg.use_interventional_reliability = name != 'baseline'
        train_cfg.reliability_task_corrupt_scale = 0.0
        train_cfg.budget_warmup_mode = 'allocation'
        train_cfg.reliability_allocation_control = name if name != 'baseline' else 'learned'
        model = tva_train.TVA_fusion(config).to(config.DEVICE)
        model.load_froze()
        model.load_model(checkpoints[name])
        predictions[name] = collect_predictions(
            model, loader, config.DEVICE, args.max_batches, baseline=name == 'baseline'
        )
        checkpoint_hashes[name] = sha256(checkpoints[name])
        print('collected %s predictions: n=%d exp=%s' %
              (name, predictions[name]['index'].size, experiments[name]), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = predictions['baseline']
    for name in ('constant', 'learned'):
        if not np.array_equal(predictions[name]['index'], baseline['index']):
            raise ValueError('%s sample order differs from baseline' % name)
        if not np.array_equal(predictions[name]['label'], baseline['label']):
            raise ValueError('%s labels differ from baseline' % name)

    labels = baseline['label']
    result = {
        'design': 'post-hoc exploratory; baseline-defined zero-ablation sensitivity thirds',
        'utility_definition': 'absolute-error increase after zeroing one modality in the repaired baseline; perturbation proxy only',
        'dataset': 'MOSEI',
        'split': args.split,
        'seed': args.seed,
        'n': int(labels.size),
        'max_batches': args.max_batches,
        'experiments': experiments,
        'checkpoint_sha256': checkpoint_hashes,
        'full_sample_metrics': {
            name: regression_metrics(labels, predictions[name]['clean'])
            for name in ('baseline', 'constant', 'learned')
        },
        'strata': [],
    }
    for modality, missing_key in (
        ('vision', 'without_vision'), ('audio', 'without_audio')
    ):
        utility = (
            np.abs(baseline[missing_key] - labels) -
            np.abs(baseline['clean'] - labels)
        )
        groups, boundary_ties = thirds_by_baseline_utility(utility, baseline['index'])
        result.setdefault('utility_diagnostics', {})[modality] = {
            'unique_values': int(np.unique(utility).size),
            'zero_fraction': float(np.mean(utility == 0)),
            'standard_deviation': float(np.std(utility)),
            'boundary_ties': boundary_ties,
        }
        for group, label in enumerate(('low', 'middle', 'high')):
            mask = groups == group
            values = utility[mask]
            metrics = {
                name: regression_metrics(labels[mask], predictions[name]['clean'][mask])
                for name in ('baseline', 'constant', 'learned')
            }
            result['strata'].append({
                'modality': modality,
                'stratum': label,
                'n': int(mask.sum()),
                'utility_min': float(values.min()),
                'utility_median': float(np.median(values)),
                'utility_max': float(values.max()),
                'lower_boundary_tied': boundary_ties[group - 1] if group > 0 else False,
                'upper_boundary_tied': boundary_ties[group] if group < 2 else False,
                'metrics': metrics,
                'learned_minus_constant': {
                    metric: (
                        metrics['learned'][metric] - metrics['constant'][metric]
                        if metrics['learned'][metric] is not None and
                        metrics['constant'][metric] is not None else None
                    )
                    for metric in ('MAE', 'Corr', 'Loss')
                },
            })

    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
        print('saved summary to %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
