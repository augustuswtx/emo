import argparse
import hashlib
import os
import subprocess
import sys


FILES = {
    'MOSI': {
        'id': '1U_RpJB_PV-JRgCs694UHczLr2h5YJOma',
        'path': 'data/MOSI/unaligned_50.pkl',
        'sha256': '78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524',
    },
    'MOSEI': {
        'id': '13yKFqrS6v95QzH29h-zSg7ZuJdn6G5yA',
        'path': 'data/MOSEI/unaligned_50.pkl',
        'sha256': 'ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f',
    },
    'SIMS': {
        'id': '1l_Nb9h3BRa3S-N76YcSQ3ixakr0gn6Qr',
        'path': 'data/SIMS/unaligned_39.pkl',
        'sha256': 'c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f',
    },
}


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(dataset):
    item = FILES[dataset]
    output = item['path']
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if os.path.exists(output):
        digest = sha256sum(output)
        if digest == item['sha256']:
            print(f'{dataset}: already downloaded and verified at {output}')
            return
        print(f'{dataset}: existing file hash mismatch, re-downloading {output}')
    cmd = [
        sys.executable,
        '-m',
        'gdown',
        item['id'],
        '-O',
        output,
        '--continue',
    ]
    subprocess.check_call(cmd)
    digest = sha256sum(output)
    if digest != item['sha256']:
        raise RuntimeError(f'{dataset}: SHA-256 mismatch: got {digest}, expected {item["sha256"]}')
    print(f'{dataset}: downloaded and verified at {output}')


def main():
    parser = argparse.ArgumentParser(description='Download MMSA processed data used by MFON.')
    parser.add_argument('--dataset', choices=[*FILES.keys(), 'all'], default='all')
    args = parser.parse_args()
    datasets = FILES.keys() if args.dataset == 'all' else [args.dataset]
    for dataset in datasets:
        download(dataset)


if __name__ == '__main__':
    main()
