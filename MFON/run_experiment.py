import argparse
import importlib
import os
import sys


DATASETS = {
    'SIMS': {
        'loader': 'SIMSDataloader',
    },
    'MOSI': {
        'loader': 'MOSIDataloader',
    },
    'MOSEI': {
        'loader': 'MOSEIDataloader',
    },
}


def clear_dataset_modules():
    for name in list(sys.modules):
        if name in {'config', 'data_loader', 'utils', 'train', 'models'}:
            del sys.modules[name]
        elif name.startswith(('train.', 'models.')):
            del sys.modules[name]


def load_dataset_modules(dataset):
    clear_dataset_modules()
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset)
    sys.path.insert(0, dataset_dir)
    try:
        config = importlib.import_module('config')
        data_loader = importlib.import_module('data_loader')
        utils = importlib.import_module('utils')
        tva_train = importlib.import_module('train.TVA_train')
        audio_train = importlib.import_module('train.Atrain')
        vision_train = importlib.import_module('train.Vtrain')
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            raise RuntimeError(
                'PyTorch is not available in the current Python environment. '
                'Create/activate the project environment first, e.g. '
                '`conda env create -f environment.yaml && conda activate pytorch`.'
            ) from exc
        raise
    return dataset_dir, config, data_loader, utils, tva_train, audio_train, vision_train


def resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def require_files(paths, purpose):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        formatted = '\n  - '.join(missing)
        raise FileNotFoundError(f'Missing files for {purpose}:\n  - {formatted}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run MFON/DAMFON experiments.')
    parser.add_argument('--dataset', choices=DATASETS.keys(), required=True)
    parser.add_argument(
        '--stage',
        choices=['train-audio', 'train-vision', 'train-unimodal', 'train-fusion', 'test-audio', 'test-vision', 'test-fusion'],
        default='train-fusion',
    )
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--epochs', type=int, default=None, help='Override fusion training epochs.')
    parser.add_argument('--exp-name', default=None, help='Checkpoint subdirectory name under seed path.')
    weighting = parser.add_mutually_exclusive_group()
    weighting.add_argument('--use-alw', action='store_true', help='Enable legacy adaptive loss weighting.')
    weighting.add_argument(
        '--use-budgeted-aux',
        action='store_true',
        help='Enable per-sample auxiliary losses with a fixed batch supervision budget.',
    )
    parser.add_argument('--q-type', choices=['align', 'norm', 'conf'], default='align')
    parser.add_argument('--warmup-epoch', type=int, default=10)
    parser.add_argument(
        '--budget-warmup-mode',
        choices=['scale', 'allocation'],
        default='scale',
        help=(
            'scale reproduces the original growing mean budget; allocation '
            'keeps the mean budget fixed and warms only sample redistribution.'
        ),
    )
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument(
        '--use-interventional-reliability',
        action='store_true',
        help='Learn modality reliability from ordered clean/corrupt feature triplets.',
    )
    parser.add_argument('--reliability-hidden-dim', type=int, default=64)
    parser.add_argument('--reliability-max-severity', type=float, default=1.0)
    parser.add_argument('--reliability-corrupt-prob', type=float, default=0.5)
    parser.add_argument('--reliability-margin', type=float, default=0.2)
    parser.add_argument('--reliability-loss-weight', type=float, default=0.1)
    parser.add_argument('--reliability-invariance-weight', type=float, default=0.1)
    parser.add_argument('--reliability-task-warmup-epoch', type=int, default=10)
    parser.add_argument(
        '--reliability-task-corrupt-scale',
        type=float,
        default=1.0,
        help=(
            'Scale task-path corruption in [0, 1] without disabling '
            'clean/corrupt supervision for the reliability heads.'
        ),
    )
    parser.add_argument(
        '--reliability-allocation-control',
        choices=['learned', 'constant', 'permuted', 'reversed', 'oracle'],
        default='learned',
        help='Equal-budget control applied only to reliability-based auxiliary allocation.',
    )
    parser.add_argument('--use-dpg', action='store_true', help='Enable dynamic prompt gating.')
    parser.add_argument('--dpg-hidden-dim', type=int, default=256)
    parser.add_argument('--use-css', action='store_true', help='Enable curriculum sample filtering.')
    parser.add_argument('--css-epoch', type=int, default=20)
    parser.add_argument('--css-min-ratio', type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    dataset_dir, config, data_loader, utils, tva_train, audio_train, vision_train = load_dataset_modules(args.dataset)
    dataset_cfg = getattr(config, args.dataset)
    train_cfg = dataset_cfg.downStream.TVAtrain

    config.seed = args.seed
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError('--epochs must be at least 1.')
        train_cfg.epoch = args.epochs
    train_cfg.use_alw = args.use_alw
    train_cfg.use_budgeted_aux = args.use_budgeted_aux
    train_cfg.alw_q_type = args.q_type
    train_cfg.alw_warmup_epoch = args.warmup_epoch
    train_cfg.budget_warmup_epoch = args.warmup_epoch
    train_cfg.budget_warmup_mode = args.budget_warmup_mode
    train_cfg.alw_temperature = args.temperature
    train_cfg.use_interventional_reliability = args.use_interventional_reliability
    train_cfg.reliability_hidden_dim = args.reliability_hidden_dim
    train_cfg.reliability_max_severity = args.reliability_max_severity
    train_cfg.reliability_corrupt_prob = args.reliability_corrupt_prob
    train_cfg.reliability_margin = args.reliability_margin
    train_cfg.reliability_loss_weight = args.reliability_loss_weight
    train_cfg.reliability_invariance_weight = args.reliability_invariance_weight
    train_cfg.reliability_task_warmup_epoch = args.reliability_task_warmup_epoch
    if not 0.0 <= args.reliability_task_corrupt_scale <= 1.0:
        raise ValueError('--reliability-task-corrupt-scale must be in [0, 1].')
    train_cfg.reliability_task_corrupt_scale = args.reliability_task_corrupt_scale
    train_cfg.reliability_allocation_control = args.reliability_allocation_control
    if args.use_interventional_reliability:
        if args.dataset != 'MOSI':
            raise ValueError('The interventional-reliability P1 pilot currently supports MOSI only.')
        if not args.use_budgeted_aux:
            raise ValueError('--use-interventional-reliability requires --use-budgeted-aux.')
    elif args.reliability_allocation_control != 'learned':
        raise ValueError(
            '--reliability-allocation-control requires '
            '--use-interventional-reliability.'
        )
    train_cfg.use_dpg = args.use_dpg
    train_cfg.dpg_hidden_dim = args.dpg_hidden_dim
    train_cfg.use_css = args.use_css
    train_cfg.css_epoch = args.css_epoch
    train_cfg.css_min_ratio = args.css_min_ratio
    if args.exp_name is None:
        flags = []
        if args.use_alw:
            flags.append(f'alw_{args.q_type}')
        if args.use_budgeted_aux:
            flags.append(f'budgeted_aux_{args.q_type}')
        if args.use_interventional_reliability:
            flags.append('interventional_reliability')
        if args.use_dpg:
            flags.append('dpg')
        if args.use_css:
            flags.append('css')
        args.exp_name = 'baseline' if not flags else '_'.join(flags)
    train_cfg.exp_name = args.exp_name

    loader_cls = getattr(data_loader, DATASETS[args.dataset]['loader'])
    batch_size = dataset_cfg.downStream.batch_size
    metrics = utils.Metrics()
    raw_data_path = resolve_repo_path(dataset_cfg.path.raw_data_path)
    require_files([raw_data_path], f'{args.dataset} data')
    dataset_cfg.path.raw_data_path = raw_data_path

    print(
        f'Running {args.dataset} {args.stage} | seed={args.seed} | '
        f'exp_name={args.exp_name} | '
        f'epochs={train_cfg.epoch} | '
        f'use_alw={args.use_alw} | use_budgeted_aux={args.use_budgeted_aux} | '
        f'q_type={args.q_type} | warmup={args.warmup_epoch} | '
        f'budget_warmup_mode={args.budget_warmup_mode} | '
        f'interventional_reliability={args.use_interventional_reliability} | '
        f'task_corrupt_scale={args.reliability_task_corrupt_scale} | '
        f'allocation_control={args.reliability_allocation_control} | '
        f'use_dpg={args.use_dpg} | use_css={args.use_css}'
    )
    if args.stage in {'train-audio', 'train-vision', 'train-unimodal', 'train-fusion'}:
        encoder_dir = os.path.join(dataset_cfg.path.encoder_path, str(args.seed))
    if args.stage == 'train-audio':
        train_data = loader_cls('train', dataset_cfg.path.raw_data_path, batch_size=batch_size)
        valid_data = loader_cls('valid', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        audio_train.Atrain(config, metrics, args.seed, train_data, valid_data)
    elif args.stage == 'train-vision':
        train_data = loader_cls('train', dataset_cfg.path.raw_data_path, batch_size=batch_size)
        valid_data = loader_cls('valid', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        vision_train.Vtrain(config, metrics, args.seed, train_data, valid_data)
    elif args.stage == 'train-unimodal':
        train_data = loader_cls('train', dataset_cfg.path.raw_data_path, batch_size=batch_size)
        valid_data = loader_cls('valid', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        audio_train.Atrain(config, metrics, args.seed, train_data, valid_data)
        vision_train.Vtrain(config, metrics, args.seed, train_data, valid_data)
    elif args.stage == 'train-fusion':
        require_files(
            [
                os.path.join(encoder_dir, 'best_loss_audio_encoder.pt'),
                os.path.join(encoder_dir, 'best_loss_vision_encoder.pt'),
            ],
            f'{args.dataset} frozen unimodal encoders',
        )
        train_data = loader_cls('train', dataset_cfg.path.raw_data_path, batch_size=batch_size)
        valid_data = loader_cls('valid', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        tva_train.TVA_train_fusion(config, metrics, args.seed, train_data, valid_data)
    elif args.stage == 'test-audio':
        test_data = loader_cls('test', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        audio_train.Atest(config, metrics, test_data)
    elif args.stage == 'test-vision':
        test_data = loader_cls('test', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        vision_train.Vtest(config, metrics, test_data)
    else:
        require_files(
            [os.path.join(dataset_cfg.path.model_path, str(args.seed), args.exp_name, 'TVA_fusion_model.pt')],
            f'{args.dataset} fusion checkpoint',
        )
        test_data = loader_cls('test', dataset_cfg.path.raw_data_path, batch_size=batch_size, shuffle=False)
        tva_train.TVA_test_fusion(config, metrics, test_data)

    sys.path.remove(dataset_dir)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
