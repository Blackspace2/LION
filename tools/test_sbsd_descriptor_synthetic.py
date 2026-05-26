#!/usr/bin/env python3

import torch
import spconv.pytorch as spconv

from pcdet.models.backbones_3d.sbsd import SBSD


def make_sbsd(
    dim=8,
    use_bandwidth=False,
    use_spectral=True,
    use_density=True,
    use_spectral_norm=False,
    point_cloud_range=None,
):
    if point_cloud_range is None:
        point_cloud_range = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
    return SBSD(
        dim=dim,
        cfg={
            "ENABLED": True,
            "USE_BANDWIDTH": use_bandwidth,
            "USE_SPECTRAL": use_spectral,
            "USE_DENSITY": use_density,
            "USE_SPECTRAL_NORM": use_spectral_norm,
            "SIGMA": 1.0,
        },
        point_cloud_range=point_cloud_range,
    ).cuda().eval()


def make_tensor(features, indices, spatial_shape=(4, 4, 4), batch_size=1):
    return spconv.SparseConvTensor(
        features=features.cuda(),
        indices=indices.cuda(),
        spatial_shape=list(spatial_shape),
        batch_size=batch_size,
    )


def assert_close(name, actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch: actual={actual} expected={expected} atol={atol}")


def test_empty_descriptor():
    sbsd = make_sbsd()
    x = make_tensor(
        features=torch.zeros((0, 8), dtype=torch.float32),
        indices=torch.zeros((0, 4), dtype=torch.int32),
    )
    d = sbsd._compute_descriptor(x)
    assert d.shape == (0, 7), d.shape


def test_isolated_voxel_zero_descriptor():
    sbsd = make_sbsd()
    x = make_tensor(
        features=torch.ones((1, 8), dtype=torch.float32),
        indices=torch.tensor([[0, 1, 1, 1]], dtype=torch.int32),
    )
    d = sbsd._compute_descriptor(x)
    assert d.shape == (1, 7), d.shape
    assert_close("isolated_descriptor", d, torch.zeros_like(d))


def test_symmetric_neighbors_have_zero_center_offset():
    sbsd = make_sbsd()
    x = make_tensor(
        features=torch.ones((3, 8), dtype=torch.float32),
        indices=torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 2],
            ],
            dtype=torch.int32,
        ),
    )
    d = sbsd._compute_descriptor(x)
    center = d[0]
    left = d[1]
    right = d[2]
    if not torch.isfinite(d).all():
        raise AssertionError("descriptor contains non-finite values")
    if not center[3].item() > 0.0:
        raise AssertionError(f"expected positive rho at center, got {center[3].item()}")
    assert abs(center[4].item()) < 1e-6, center[4].item()
    assert abs(center[5].item()) < 1e-6, center[5].item()
    assert abs(center[6].item()) < 1e-6, center[6].item()
    if not left[1].item() > 0.0:
        raise AssertionError(f"expected positive e1 at left edge, got {left[1].item()}")
    if not right[1].item() > 0.0:
        raise AssertionError(f"expected positive e1 at right edge, got {right[1].item()}")


def test_multibatch_no_leakage():
    sbsd = make_sbsd()
    x = make_tensor(
        features=torch.ones((6, 8), dtype=torch.float32),
        indices=torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 2],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 2],
            ],
            dtype=torch.int32,
        ),
        batch_size=2,
    )
    d = sbsd._compute_descriptor(x)
    assert_close("multibatch_center", d[0], d[3])
    assert_close("multibatch_left", d[1], d[4])
    assert_close("multibatch_right", d[2], d[5])


def test_proxy_bandwidth_order_sensitivity():
    sbsd = make_sbsd(
        use_bandwidth=True,
        use_spectral=False,
        use_density=False,
        point_cloud_range=[0.0, 0.0, 0.0, 8.0, 4.0, 4.0],
    )
    ordered_indices = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 1, 3],
            [0, 1, 1, 4],
            [0, 1, 1, 5],
        ],
        dtype=torch.int32,
    )
    shuffled_indices = ordered_indices[torch.tensor([0, 3, 1, 4, 2], dtype=torch.long)]
    features = torch.ones((5, 8), dtype=torch.float32)

    ordered = make_tensor(features=features, indices=ordered_indices, spatial_shape=(4, 4, 8))
    shuffled = make_tensor(features=features, indices=shuffled_indices, spatial_shape=(4, 4, 8))

    ordered_bbar = sbsd._compute_descriptor(ordered)[:, 0]
    shuffled_bbar = sbsd._compute_descriptor(shuffled)[:, 0]

    ordered_mean = float(ordered_bbar.mean().item())
    shuffled_mean = float(shuffled_bbar.mean().item())
    if not shuffled_mean > ordered_mean + 1e-4:
        raise AssertionError(
            f"expected shuffled proxy bandwidth to increase: ordered={ordered_mean} shuffled={shuffled_mean}"
        )


def test_spectral_block_layernorm_zero_centers_active_rows():
    sbsd = make_sbsd(use_bandwidth=False, use_spectral=True, use_density=True, use_spectral_norm=True)
    x = make_tensor(
        features=torch.ones((3, 8), dtype=torch.float32),
        indices=torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 2],
            ],
            dtype=torch.int32,
        ),
    )
    d = sbsd._compute_descriptor(x)
    spectral_block = d[:, 1:]
    active_rows = spectral_block.abs().sum(dim=-1) > 0
    if not active_rows.any():
        raise AssertionError("expected at least one active spectral row")
    centered = spectral_block[active_rows].mean(dim=-1)
    assert_close("spectral_norm_centering", centered, torch.zeros_like(centered), atol=1e-5)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this SBSD synthetic test")

    test_empty_descriptor()
    test_isolated_voxel_zero_descriptor()
    test_symmetric_neighbors_have_zero_center_offset()
    test_multibatch_no_leakage()
    test_proxy_bandwidth_order_sensitivity()
    test_spectral_block_layernorm_zero_centers_active_rows()
    print("sbsd_synthetic_tests=pass")


if __name__ == "__main__":
    main()
