#!/usr/bin/env python3
import _init_path  # noqa: F401
import torch

from pcdet.models.backbones_3d.lion_improve import (
    GEOMETRY_ORDER_NAMES,
    build_geometry_order_from_coords,
    reverse_order_within_batches,
)


def assert_permutation(order, num_nodes, name):
    expected = torch.arange(num_nodes, device=order.device, dtype=torch.long)
    if order.dtype != torch.long:
        raise AssertionError(f'{name}: expected long order, got {order.dtype}')
    if order.numel() != num_nodes or not torch.equal(torch.sort(order).values, expected):
        raise AssertionError(f'{name}: invalid permutation {order.tolist()}')


def run_geometry_orders(device='cpu'):
    coords = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 4],
            [0, 0, 1, 4],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    sparse_shape = [5, 6, 9]
    window_shape = [4, 3, 2]
    for name in sorted(GEOMETRY_ORDER_NAMES):
        order = build_geometry_order_from_coords(
            coords=coords,
            sparse_shape=sparse_shape,
            window_shape=window_shape,
            shift=False,
            order_name=name,
        )
        assert_permutation(order, coords.shape[0], name)

    left = build_geometry_order_from_coords(coords, sparse_shape, window_shape, False, 'bev_h')
    right = build_geometry_order_from_coords(coords, sparse_shape, window_shape, False, 'bev_h_t')
    if torch.equal(left, right):
        raise AssertionError('bev_h_t should differ from bev_h on asymmetric coordinates')


def run_reverse_order(device='cpu'):
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [1, 0, 0, 0], [1, 0, 0, 1]],
        device=device,
        dtype=torch.long,
    )
    order = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.long)
    rev = reverse_order_within_batches(order, coords, batch_size=2)
    expected = torch.tensor([2, 1, 0, 4, 3], device=device, dtype=torch.long)
    if not torch.equal(rev, expected):
        raise AssertionError(f'reverse expected {expected.tolist()}, got {rev.tolist()}')


def main():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        run_geometry_orders(device)
        run_reverse_order(device)
        print(f'lion_improve_serialization device={device} pass')
    print('lion_improve_serialization_tests=pass')


if __name__ == '__main__':
    main()
