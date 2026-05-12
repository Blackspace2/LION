#!/usr/bin/env python
import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Replace non-finite BN running buffers in a checkpoint.')
    parser.add_argument('--input', required=True, help='Checkpoint to repair')
    parser.add_argument('--reference', required=True, help='Clean checkpoint used as BN buffer source')
    parser.add_argument('--output', required=True, help='Repaired checkpoint path')
    return parser.parse_args()


def is_bn_running_buffer(name):
    return 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name


def main():
    args = parse_args()
    ckpt = torch.load(args.input, map_location='cpu')
    ref = torch.load(args.reference, map_location='cpu')
    state = ckpt['model_state']
    ref_state = ref['model_state']

    replaced = []
    remaining_bad = []
    for name, value in state.items():
        if not torch.is_tensor(value) or not is_bn_running_buffer(name):
            continue
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            if name not in ref_state:
                remaining_bad.append(name)
                continue
            state[name] = ref_state[name].clone()
            replaced.append(name)

    for name, value in state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value) and not torch.isfinite(value).all():
            remaining_bad.append(name)

    if remaining_bad:
        raise RuntimeError(f'Remaining non-finite tensors: {remaining_bad[:20]}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(ckpt, args.output, _use_new_zipfile_serialization=False)
    print(f'Replaced {len(replaced)} BN running buffers')
    print(args.output)


if __name__ == '__main__':
    main()
