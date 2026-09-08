#!/usr/bin/env python3
"""Benchmark existing MFON/P4 checkpoints without starting model training.

The audit uses one real batch repeatedly in a single process so Baseline,
Constant, and Learned are compared on the same device and data shape. It
reports parameter scopes, checkpoint size, end-to-end inference latency, and
training forward/backward latency plus peak CUDA memory. It never creates or
updates a model checkpoint and never calls an optimizer step.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from run_experiment import DATASETS, load_dataset_modules, require_files, resolve_repo_path


DEFAULT_EXPERIMENTS = {
    'baseline': 'p5_mosei_repaired_baseline',
    'constant': 'p5_mosei_p4_constant_true_budget',
    'learned': 'p5_mosei_p4_learned_true_budget',
}


def parse_variants(raw):
    variants = [value.strip() for value in raw.split(',') if value.strip()]
    if not variants:
        raise ValueError('at least one efficiency variant is required.')
    unknown = sorted(set(variants) - set(DEFAULT_EXPERIMENTS))
    if unknown:
        raise ValueError('unknown efficiency variants: %s' % ', '.join(unknown))
    if len(variants) != len(set(variants)):
        raise ValueError('efficiency variants must be unique.')
    return variants


def count_unique_parameters(modules=(), parameters=()):
    unique = {}
    for module in modules:
        for parameter in module.parameters():
            unique[id(parameter)] = parameter
    for parameter in parameters:
        unique[id(parameter)] = parameter
    return int(sum(parameter.numel() for parameter in unique.values()))


def parameter_summary(model, interventional):
    inference_modules = (
        model.text_encoder,
        model.proj_t,
        model.proj_v,
        model.vision_with_text,
        model.proj_a,
        model.audio_with_text,
        model.TVA_decoder,
    )
    inference_parameters = (model.promptv_m, model.prompta_m)
    reliability_modules = (model.vision_reliability, model.audio_reliability)
    teacher_modules = (model.vision_encoder_froze, model.audio_encoder_froze)
    optimizer_modules = list(inference_modules)
    if interventional:
        optimizer_modules.extend(reliability_modules)
    return {
        'stored_model_parameters': count_unique_parameters(modules=(model,)),
        'task_inference_parameters': count_unique_parameters(
            modules=inference_modules, parameters=inference_parameters
        ),
        'training_optimizer_parameters': count_unique_parameters(
            modules=optimizer_modules, parameters=inference_parameters
        ),
        'p4_reliability_parameters': count_unique_parameters(
            modules=reliability_modules
        ),
        'frozen_teacher_parameters': count_unique_parameters(modules=teacher_modules),
    }


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cuda_memory_snapshot(device):
    if device.type != 'cuda':
        return None
    return int(torch.cuda.memory_allocated(device))


def benchmark_step(step, device, warmup, repeats, after_warmup=None):
    for _ in range(warmup):
        step()
    if after_warmup is not None:
        after_warmup()
    synchronize(device)
    before = cuda_memory_snapshot(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeats):
        step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = None
    incremental_peak = None
    if device.type == 'cuda':
        peak = int(torch.cuda.max_memory_allocated(device))
        incremental_peak = max(0, peak - before)
    return {
        'repeats': repeats,
        'total_seconds': elapsed,
        'mean_milliseconds': elapsed * 1000.0 / repeats,
        'peak_allocated_bytes': peak,
        'incremental_peak_bytes': incremental_peak,
    }


def configure_variant(train_cfg, variant, exp_name):
    is_p4 = variant in {'constant', 'learned'}
    train_cfg.exp_name = exp_name
    train_cfg.use_alw = False
    train_cfg.use_budgeted_aux = is_p4
    train_cfg.alw_q_type = 'align'
    train_cfg.alw_warmup_epoch = 10
    train_cfg.budget_warmup_epoch = 10
    train_cfg.budget_warmup_mode = 'allocation'
    train_cfg.use_interventional_reliability = is_p4
    train_cfg.reliability_task_warmup_epoch = 10
    train_cfg.reliability_task_corrupt_scale = 0.0
    train_cfg.reliability_allocation_control = (
        variant if variant in {'constant', 'learned'} else 'learned'
    )
    train_cfg.use_dpg = False
    train_cfg.use_css = False


def loss_for_backward(model, prediction, labels, losses, train_cfg):
    prediction_loss = torch.mean((prediction - labels) * (prediction - labels))
    if model.current_budgeted_aux is not None:
        return prediction_loss + model.current_budgeted_aux['loss']
    loss_v, loss_a, loss_nce = losses
    return (
        prediction_loss
        + train_cfg.delta_va * (loss_v + loss_a)
        + train_cfg.delta_nce * loss_nce
    )


def benchmark_variant(
    variant,
    exp_name,
    config,
    dataset_cfg,
    tva_train,
    batch,
    warmup,
    repeats,
    skip_training_step,
):
    train_cfg = dataset_cfg.downStream.TVAtrain
    configure_variant(train_cfg, variant, exp_name)
    checkpoint = os.path.join(
        dataset_cfg.path.model_path,
        str(config.seed),
        exp_name,
        'TVA_fusion_model.pt',
    )
    encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(config.seed))
    require_files(
        [
            checkpoint,
            os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
            os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
        ],
        '%s efficiency audit' % variant,
    )

    device = config.DEVICE
    model = tva_train.TVA_fusion(config).to(device)
    model.load_froze()
    model.load_model(checkpoint)
    parameters = parameter_summary(model, train_cfg.use_interventional_reliability)
    model_memory = cuda_memory_snapshot(device)

    text = batch['raw_text']
    vision = batch['vision'].clone().detach().to(device).float()
    audio = batch['audio'].clone().detach().to(device).float()
    labels = batch['labels']['M'].clone().detach().view(-1).to(device).float()
    batch_samples = int(labels.numel())

    model.eval()

    def inference_step():
        with torch.inference_mode():
            model(text, vision, audio, mode='test')

    inference = benchmark_step(inference_step, device, warmup, repeats)
    inference['samples_per_second'] = (
        batch_samples * 1000.0 / inference['mean_milliseconds']
    )

    training = None
    if not skip_training_step:
        model.train()

        def training_step():
            model.zero_grad(set_to_none=True)
            prediction, losses = model(
                text, vision, audio, mode='train', epoch=train_cfg.epoch
            )
            loss = loss_for_backward(model, prediction, labels, losses, train_cfg)
            loss.backward()

        training = benchmark_step(
            training_step,
            device,
            warmup,
            repeats,
            after_warmup=lambda: model.zero_grad(set_to_none=True),
        )
        training['samples_per_second'] = (
            batch_samples * 1000.0 / training['mean_milliseconds']
        )
        model.zero_grad(set_to_none=True)

    result = {
        'variant': variant,
        'exp_name': exp_name,
        'checkpoint': checkpoint,
        'checkpoint_bytes': int(os.path.getsize(checkpoint)),
        'batch_samples': batch_samples,
        'model_allocated_bytes': model_memory,
        'parameters': parameters,
        'inference': inference,
        'training_forward_backward': training,
    }
    del model, vision, audio, labels
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit parameter, latency, and peak-memory cost without training.'
    )
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='MOSEI')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--variants', default='baseline,constant,learned')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--skip-training-step', action='store_true')
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--baseline-exp-name', default=DEFAULT_EXPERIMENTS['baseline'])
    parser.add_argument('--constant-exp-name', default=DEFAULT_EXPERIMENTS['constant'])
    parser.add_argument('--learned-exp-name', default=DEFAULT_EXPERIMENTS['learned'])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative.')
    if args.repeats < 1:
        raise ValueError('--repeats must be positive.')
    variants = parse_variants(args.variants)
    experiment_names = {
        'baseline': args.baseline_exp_name,
        'constant': args.constant_exp_name,
        'learned': args.learned_exp_name,
    }

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    dataset_dir, config, data_loader, _, tva_train, _, _ = load_dataset_modules(
        args.dataset
    )
    dataset_cfg = getattr(config, args.dataset)
    config.seed = args.seed
    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    require_files([raw_data_path], '%s efficiency audit data' % args.dataset)
    dataset_cfg.path.raw_data_path = raw_data_path
    loader_cls = getattr(data_loader, DATASETS[args.dataset]['loader'])
    loader = loader_cls(
        args.split,
        raw_data_path,
        batch_size=dataset_cfg.downStream.batch_size,
        shuffle=False,
    )
    batch = next(iter(loader))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = config.DEVICE
    hardware = {
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cuda_device_name': (
            torch.cuda.get_device_name(device) if device.type == 'cuda' else None
        ),
        'configured_batch_size': int(dataset_cfg.downStream.batch_size),
        'warmup': args.warmup,
        'repeats': args.repeats,
        'training_step_included': not args.skip_training_step,
    }
    print(
        'Efficiency audit only: no optimizer step, no checkpoint write, no training run.'
    )
    print('hardware:', json.dumps(hardware, ensure_ascii=False, sort_keys=True))

    results = []
    for variant in variants:
        result = benchmark_variant(
            variant,
            experiment_names[variant],
            config,
            dataset_cfg,
            tva_train,
            batch,
            args.warmup,
            args.repeats,
            args.skip_training_step,
        )
        results.append(result)
        print('efficiency result:', json.dumps(result, ensure_ascii=False, sort_keys=True))

    payload = {
        'dataset': args.dataset,
        'seed': args.seed,
        'split': args.split,
        'hardware': hardware,
        'results': results,
    }
    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
        print('efficiency json saved at:', output_path)

    sys.path.remove(dataset_dir)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError, StopIteration) as exc:
        print('Error: %s' % exc, file=sys.stderr)
        sys.exit(1)
