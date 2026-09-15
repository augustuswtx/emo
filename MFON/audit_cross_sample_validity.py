#!/usr/bin/env python3
"""Read-only, post-hoc audit of score comparability and target fidelity.

KL is the primary cross-sample proxy because its per-sample value has a stable
definition across batches. InfoNCE depends on the negative examples in the
current mini-batch, so InfoNCE and the coefficient-weighted training proxy are
reported only through within-batch associations.
"""

import argparse
import os

import numpy as np
import torch

from audit_model_quality import active_rms, active_steps, pearson, spearman, to_numpy
from run_experiment import DATASETS, load_dataset_modules, require_files, resolve_repo_path


def pairwise_concordance(score, fidelity, max_pairs=200000, seed=20260915):
    """Probability that a higher score accompanies higher target fidelity."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    fidelity = np.asarray(fidelity, dtype=np.float64).reshape(-1)
    if score.size != fidelity.size or score.size < 2:
        return float('nan')
    rng = np.random.RandomState(seed)
    left = rng.randint(0, score.size, size=max_pairs)
    right = rng.randint(0, score.size, size=max_pairs)
    valid = (left != right) & (score[left] != score[right]) & (fidelity[left] != fidelity[right])
    if not valid.any():
        return float('nan')
    return float(np.mean(np.sign(score[left[valid]] - score[right[valid]]) ==
                         np.sign(fidelity[left[valid]] - fidelity[right[valid]])))


def residualize(values, controls):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(values.size), controls])
    finite = np.isfinite(design).all(axis=1) & np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if finite.sum() <= design.shape[1]:
        return output
    beta, _, _, _ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
    output[finite] = values[finite] - design[finite].dot(beta)
    return output


def summarize(name, score, loss, length, energy):
    fidelity = -np.asarray(loss, dtype=np.float64)
    residual_score = residualize(score, np.column_stack([length, energy]))
    residual_fidelity = residualize(fidelity, np.column_stack([length, energy]))
    finite = np.isfinite(residual_score) & np.isfinite(residual_fidelity)
    return {
        'name': name,
        'n': int(np.asarray(score).size),
        'spearman_score_fidelity': spearman(score, fidelity),
        'pearson_score_fidelity': pearson(score, fidelity),
        'partial_spearman_length_energy': spearman(
            residual_score[finite], residual_fidelity[finite]
        ),
        'pairwise_concordance': pairwise_concordance(score, fidelity),
        'corr_score_length': pearson(score, length),
        'corr_score_energy': pearson(score, energy),
    }


def within_batch_summary(name, score_batches, loss_batches):
    """Summarize score--fidelity association without comparing batches."""
    correlations = []
    concordant = 0
    comparable = 0
    samples = 0
    for score, loss in zip(score_batches, loss_batches):
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        fidelity = -np.asarray(loss, dtype=np.float64).reshape(-1)
        if score.size != fidelity.size:
            raise ValueError('score and loss batch sizes must match')
        samples += score.size
        correlation = spearman(score, fidelity)
        if np.isfinite(correlation):
            correlations.append(correlation)
        if score.size < 2:
            continue
        left, right = np.triu_indices(score.size, k=1)
        score_delta = score[left] - score[right]
        fidelity_delta = fidelity[left] - fidelity[right]
        valid = (score_delta != 0) & (fidelity_delta != 0)
        concordant += int(np.sum(np.sign(score_delta[valid]) == np.sign(fidelity_delta[valid])))
        comparable += int(valid.sum())
    correlations = np.asarray(correlations, dtype=np.float64)
    return {
        'name': name,
        'n': int(samples),
        'n_batches': int(len(score_batches)),
        'within_batch_spearman_mean': (
            float(correlations.mean()) if correlations.size else float('nan')
        ),
        'within_batch_spearman_sd': (
            float(correlations.std(ddof=1)) if correlations.size > 1 else float('nan')
        ),
        'within_batch_pairwise_concordance': (
            float(concordant) / comparable if comparable else float('nan')
        ),
        'comparable_pairs': int(comparable),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='MOSEI')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='valid')
    parser.add_argument('--max-batches', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    _, config, data_loader, _, tva_train, _, _ = load_dataset_modules(args.dataset)
    dataset_cfg = getattr(config, args.dataset)
    train_cfg = dataset_cfg.downStream.TVAtrain
    config.seed = args.seed
    train_cfg.use_alw = False
    train_cfg.use_budgeted_aux = True
    train_cfg.use_interventional_reliability = True
    train_cfg.reliability_task_corrupt_scale = 0.0
    train_cfg.reliability_allocation_control = 'learned'
    train_cfg.budget_warmup_mode = 'allocation'
    train_cfg.exp_name = args.exp_name

    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    checkpoint = os.path.join(dataset_cfg.path.model_path, str(args.seed), args.exp_name,
                              'TVA_fusion_model.pt')
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    require_files([raw_data_path, checkpoint,
                   os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
                   os.path.join(encoder_dir, 'best_loss_vision_encoder.pt')],
                  'cross-sample validity audit')
    dataset_cfg.path.raw_data_path = raw_data_path
    loader_cls = getattr(data_loader, DATASETS[args.dataset]['loader'])
    loader = loader_cls(args.split, raw_data_path,
                        batch_size=dataset_cfg.downStream.batch_size, shuffle=False)
    model = tva_train.TVA_fusion(config).to(config.DEVICE)
    model.load_froze()
    model.load_model(checkpoint)
    model.eval()

    fields = {key: [] for key in ('q_v', 'q_a', 'loss_v_each', 'loss_a_each',
                                   'loss_nce_v_each', 'loss_nce_a_each')}
    v_length, a_length, v_energy, a_energy = [], [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            vision_cpu = batch['vision'].clone().detach().float()
            audio_cpu = batch['audio'].clone().detach().float()
            model(batch['raw_text'], vision_cpu.to(config.DEVICE),
                  audio_cpu.to(config.DEVICE), mode='train', epoch=25)
            evidence = model.current_budgeted_aux
            for key in fields:
                fields[key].append(evidence[key].cpu())
            v_length.append(active_steps(vision_cpu))
            a_length.append(active_steps(audio_cpu))
            v_energy.append(active_rms(vision_cpu))
            a_energy.append(active_rms(audio_cpu))

    arrays = {key: to_numpy(value) for key, value in fields.items()}
    delta_va = float(train_cfg.delta_va)
    delta_nce = float(train_cfg.delta_nce)
    q_v_batches = [part.numpy() for part in fields['q_v']]
    q_a_batches = [part.numpy() for part in fields['q_a']]
    kl_v_batches = [part.numpy() for part in fields['loss_v_each']]
    kl_a_batches = [part.numpy() for part in fields['loss_a_each']]
    nce_v_batches = [part.numpy() for part in fields['loss_nce_v_each']]
    nce_a_batches = [part.numpy() for part in fields['loss_nce_a_each']]
    weighted_v_batches = [
        delta_va * kl + delta_nce * nce
        for kl, nce in zip(kl_v_batches, nce_v_batches)
    ]
    weighted_a_batches = [
        delta_va * kl + delta_nce * nce
        for kl, nce in zip(kl_a_batches, nce_a_batches)
    ]
    primary_results = [
        summarize('vision_kl_global', arrays['q_v'], arrays['loss_v_each'],
                  to_numpy(v_length), to_numpy(v_energy)),
        summarize('audio_kl_global', arrays['q_a'], arrays['loss_a_each'],
                  to_numpy(a_length), to_numpy(a_energy)),
    ]
    secondary_results = [
        within_batch_summary('vision_kl_within_batch', q_v_batches, kl_v_batches),
        within_batch_summary('audio_kl_within_batch', q_a_batches, kl_a_batches),
        within_batch_summary('vision_infonce_within_batch', q_v_batches, nce_v_batches),
        within_batch_summary('audio_infonce_within_batch', q_a_batches, nce_a_batches),
        within_batch_summary('vision_weighted_proxy_within_batch', q_v_batches,
                             weighted_v_batches),
        within_batch_summary('audio_weighted_proxy_within_batch', q_a_batches,
                             weighted_a_batches),
    ]
    print('Cross-sample validity audit (post-hoc exploratory) '
          'dataset=%s split=%s seed=%d exp=%s delta_va=%g delta_nce=%g' %
          (args.dataset, args.split, args.seed, args.exp_name, delta_va, delta_nce))
    print('Primary analysis: global score--KL association; KL has a stable '
          'per-sample definition across batches.')
    for result in primary_results:
        print(' | '.join('%s=%s' % (key, value) for key, value in result.items()))
    print('Secondary analysis: within-batch only; InfoNCE depends on the current '
          'batch negative set. weighted_proxy=delta_va*KL+delta_nce*InfoNCE.')
    for result in secondary_results:
        print(' | '.join('%s=%s' % (key, value) for key, value in result.items()))
    print('Interpretation: positive score--fidelity association and concordance above '
          '0.5 support, but do not prove, allocation semantics. Validation and test '
          'results are post-hoc exploratory and must not be described as preregistered.')


if __name__ == '__main__':
    main()
