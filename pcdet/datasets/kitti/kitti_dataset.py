import copy
import contextlib
import io as pyio
import os
import pickle
from pathlib import Path

import numpy as np
from skimage import io as skio

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..augmentor import database_sampler
from ..dataset import DatasetTemplate


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self._ground_segmenter = None
        self.refresh_ground_prior_state()
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self._ground_segmenter = None
        self.refresh_ground_prior_state()

    def refresh_ground_prior_state(self):
        self.ground_prior_cfg = self.dataset_cfg.get('GROUND_PRIOR', None)
        self.ground_prior_enabled = self.ground_prior_cfg is not None and self.ground_prior_cfg.get('ENABLED', False)
        self.ground_defect_guidance_enabled = self.ground_prior_enabled and self.ground_prior_cfg.get(
            'GROUND_DEFECT_GUIDANCE_ENABLED', False
        )
        self.ground_context_film_enabled = self.ground_prior_enabled and self.ground_prior_cfg.get(
            'GROUND_CONTEXT_FILM_ENABLED', False
        )
        self.bev_occupancy_guidance_enabled = self.ground_prior_enabled and self.ground_prior_cfg.get(
            'BEV_OCCUPANCY_GUIDANCE_ENABLED', False
        )
        self.patchwork_guidance_enabled = self.ground_prior_enabled and self.ground_prior_cfg.get(
            'PATCHWORK_GUIDANCE_ENABLED', False
        )
        enabled_modes = sum([
            int(self.ground_defect_guidance_enabled),
            int(self.ground_context_film_enabled),
            int(self.bev_occupancy_guidance_enabled),
            int(self.patchwork_guidance_enabled),
        ])
        if enabled_modes > 1:
            raise ValueError(
                'GROUND_DEFECT_GUIDANCE_ENABLED, GROUND_CONTEXT_FILM_ENABLED, '
                'BEV_OCCUPANCY_GUIDANCE_ENABLED, and PATCHWORK_GUIDANCE_ENABLED '
                'cannot enable more than one at the same time'
            )
        self.custom_ground_feature_enabled = (
            self.ground_defect_guidance_enabled or
            self.ground_context_film_enabled or
            self.bev_occupancy_guidance_enabled or
            self.patchwork_guidance_enabled
        )
        self.legacy_ground_prior_enabled = (
            self.ground_prior_enabled and
            not self.ground_defect_guidance_enabled and
            not self.ground_context_film_enabled and
            not self.bev_occupancy_guidance_enabled and
            not self.patchwork_guidance_enabled
        )
        if self.patchwork_guidance_enabled or self.bev_occupancy_guidance_enabled:
            self.point_feature_names = (
                'is_ground',
                'patch_center_z',
                'patch_normal_z',
                'patch_flatness',
                'patch_elevation',
                'patch_point_count',
                'patch_zone_id',
                'patch_ring_id',
                'patch_sector_id',
                'point_patch_id',
            )
        elif self.ground_context_film_enabled:
            self.point_feature_names = (
                'is_ground',
                'delta_z_to_ground',
                'ground_valid',
            )
        else:
            self.point_feature_names = (
                'is_ground',
                'delta_z_to_ground',
                'local_ground_height',
            )

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = skio.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(skio.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = skio.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @contextlib.contextmanager
    def suppress_native_stdio(self):
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                with contextlib.redirect_stdout(pyio.StringIO()), contextlib.redirect_stderr(pyio.StringIO()):
                    yield
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    def get_ground_segmenter(self):
        if self._ground_segmenter is not None:
            return self._ground_segmenter

        if self.patchwork_guidance_enabled or self.bev_occupancy_guidance_enabled:
            try:
                import pypatchworkpp
            except ImportError as exc:
                raise ImportError(
                    'PATCHWORK_GUIDANCE_ENABLED or BEV_OCCUPANCY_GUIDANCE_ENABLED but pypatchworkpp is not available'
                ) from exc

            params = pypatchworkpp.Parameters()
            with self.suppress_native_stdio():
                self._ground_segmenter = pypatchworkpp.patchworkpp(params)
        else:
            try:
                import linefit_bind
            except ImportError as exc:
                raise ImportError('GROUND_PRIOR enabled but linefit_bind is not available') from exc

            config_path = self.ground_prior_cfg.get('LINEFIT_CONFIG', None)
            with self.suppress_native_stdio():
                if config_path is not None:
                    self._ground_segmenter = linefit_bind.ground_seg(str(Path(config_path).expanduser()))
                else:
                    self._ground_segmenter = linefit_bind.ground_seg()
        return self._ground_segmenter

    def infer_ground_point_labels(self, points):
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        ground_segmenter = self.get_ground_segmenter()
        with self.suppress_native_stdio():
            labels = np.asarray(ground_segmenter.run(points[:, :3]), dtype=np.uint8)
        if labels.shape[0] != points.shape[0]:
            raise RuntimeError(
                f'linefit returned mismatched labels: {labels.shape[0]} vs points {points.shape[0]}'
            )
        return labels.astype(np.float32)

    def infer_patchwork_outputs(self, points):
        if points.shape[0] == 0:
            return {
                'point_ground_mask': np.zeros((0,), dtype=np.float32),
                'point_patch_id': np.zeros((0,), dtype=np.int32),
                'patch_infos': np.zeros((0, 13), dtype=np.float32),
            }

        ground_segmenter = self.get_ground_segmenter()
        with self.suppress_native_stdio():
            ground_segmenter.estimateGround(points[:, :4])

        point_ground_mask = np.asarray(ground_segmenter.getPointGroundMask(), dtype=np.float32)
        point_patch_id = np.asarray(ground_segmenter.getPointPatchIds(), dtype=np.int32)
        patch_centers = np.asarray(ground_segmenter.getCenters(), dtype=np.float32)
        patch_normals = np.asarray(ground_segmenter.getNormals(), dtype=np.float32)
        patch_flatness = np.asarray(ground_segmenter.getPatchFlatness(), dtype=np.float32).reshape(-1, 1)
        patch_elevation = np.asarray(ground_segmenter.getPatchElevation(), dtype=np.float32).reshape(-1, 1)
        patch_point_count = np.asarray(ground_segmenter.getPatchPointCount(), dtype=np.float32).reshape(-1, 1)
        patch_zone_id = np.asarray(ground_segmenter.getPatchZoneIds(), dtype=np.float32).reshape(-1, 1)
        patch_ring_id = np.asarray(ground_segmenter.getPatchRingIds(), dtype=np.float32).reshape(-1, 1)
        patch_sector_id = np.asarray(ground_segmenter.getPatchSectorIds(), dtype=np.float32).reshape(-1, 1)

        if point_ground_mask.shape[0] != points.shape[0]:
            raise RuntimeError(
                f'Patchwork++ returned mismatched ground mask: {point_ground_mask.shape[0]} vs {points.shape[0]}'
            )
        if point_patch_id.shape[0] != points.shape[0]:
            raise RuntimeError(
                f'Patchwork++ returned mismatched patch ids: {point_patch_id.shape[0]} vs {points.shape[0]}'
            )
        patch_count = patch_centers.shape[0]
        if patch_count != patch_normals.shape[0]:
            raise RuntimeError(
                f'Patchwork++ patch center/normal count mismatch: {patch_count} vs {patch_normals.shape[0]}'
            )
        if patch_count == 0:
            patch_infos = np.zeros((0, 13), dtype=np.float32)
        else:
            patch_ids = np.arange(patch_count, dtype=np.float32).reshape(-1, 1)
            patch_infos = np.concatenate(
                [
                    patch_ids,
                    patch_centers.astype(np.float32),
                    patch_normals.astype(np.float32),
                    patch_flatness,
                    patch_elevation,
                    patch_point_count,
                    patch_zone_id,
                    patch_ring_id,
                    patch_sector_id,
                ],
                axis=1
            )

        return {
            'point_ground_mask': point_ground_mask,
            'point_patch_id': point_patch_id,
            'patch_infos': patch_infos.astype(np.float32),
        }

    def build_ground_prior_cell_stats(self, points, point_ground_labels):
        if point_ground_labels is None:
            raise ValueError('point_ground_labels is required to build coarse observed ground prior')

        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        total_counts = np.zeros((map_h, map_w), dtype=np.float32)
        ground_counts = np.zeros((map_h, map_w), dtype=np.float32)
        if points.shape[0] == 0:
            return total_counts, ground_counts

        mask = common_utils.mask_points_by_range_v2(points[:, :3], self.point_cloud_range)
        if not np.any(mask):
            return total_counts, ground_counts

        points = points[mask]
        point_ground_labels = point_ground_labels[mask]
        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]

        x_idx = np.floor((points[:, 0] - pc_range_min[0]) / voxel_size[0]).astype(np.int32)
        y_idx = np.floor((points[:, 1] - pc_range_min[1]) / voxel_size[1]).astype(np.int32)

        valid = (
            (x_idx >= 0) & (x_idx < map_w) &
            (y_idx >= 0) & (y_idx < map_h)
        )
        if not np.any(valid):
            return total_counts, ground_counts

        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        point_ground_labels = point_ground_labels[valid]

        np.add.at(total_counts, (y_idx, x_idx), 1.0)
        np.add.at(ground_counts, (y_idx, x_idx), point_ground_labels)
        return total_counts, ground_counts

    @staticmethod
    def build_ground_boundary_map(ground_ratio_map, ground_valid_mask_map, ratio_threshold=0.5):
        valid = ground_valid_mask_map.astype(bool)
        majority_ground = ground_ratio_map >= ratio_threshold
        boundary_map = np.logical_and(valid, np.logical_and(ground_ratio_map > 0.0, ground_ratio_map < 1.0))

        vertical_diff = (
            valid[:-1, :] & valid[1:, :] &
            (majority_ground[:-1, :] != majority_ground[1:, :])
        )
        boundary_map[:-1, :] |= vertical_diff
        boundary_map[1:, :] |= vertical_diff

        horizontal_diff = (
            valid[:, :-1] & valid[:, 1:] &
            (majority_ground[:, :-1] != majority_ground[:, 1:])
        )
        boundary_map[:, :-1] |= horizontal_diff
        boundary_map[:, 1:] |= horizontal_diff
        return boundary_map.astype(np.float32)

    def build_coarse_observed_ground_prior(self, points, point_ground_labels):
        total_counts, ground_counts = self.build_ground_prior_cell_stats(points, point_ground_labels)
        valid_cells = total_counts > 0
        ground_ratio_map = np.zeros_like(total_counts, dtype=np.float32)
        ground_ratio_map[valid_cells] = ground_counts[valid_cells] / total_counts[valid_cells]

        ground_valid_mask_map = valid_cells.astype(np.float32)
        ground_density_map = np.log1p(total_counts)
        density_max = float(ground_density_map.max())
        if density_max > 0:
            ground_density_map /= density_max

        boundary_threshold = float(self.ground_prior_cfg.get('BOUNDARY_GROUND_RATIO_THRESHOLD', 0.5))
        ground_boundary_map = self.build_ground_boundary_map(
            ground_ratio_map=ground_ratio_map,
            ground_valid_mask_map=ground_valid_mask_map,
            ratio_threshold=boundary_threshold
        )

        coarse_observed_ground_prior = np.stack(
            [ground_ratio_map, ground_valid_mask_map, ground_density_map, ground_boundary_map],
            axis=0
        ).astype(np.float32)
        prior_dict = {
            'coarse_observed_ground_prior': coarse_observed_ground_prior,
            'ground_ratio_map': ground_ratio_map,
            'ground_valid_mask_map': ground_valid_mask_map,
            'ground_density_map': ground_density_map,
            'ground_boundary_map': ground_boundary_map,
        }
        if self.ground_prior_cfg.get('EXPORT_LEGACY_GROUND_BEV_MAP', True):
            prior_dict['ground_bev_map'] = ground_ratio_map
        return prior_dict

    def get_ground_feature_index_map(self):
        feature_names = list(self.point_feature_encoder.src_feature_list)
        missing_features = [name for name in self.point_feature_names if name not in feature_names]
        if missing_features:
            raise KeyError(
                'Ground-aware feature pipeline requires POINT_FEATURE_ENCODING.src_feature_list to include '
                f'{missing_features}, but got {feature_names}'
            )
        return {name: feature_names.index(name) for name in self.point_feature_names}

    @staticmethod
    def fit_local_ground_plane(points_xyz):
        if points_xyz.shape[0] < 3:
            return None, None

        design = np.concatenate(
            [points_xyz[:, :2].astype(np.float64), np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
            axis=1
        )
        target = points_xyz[:, 2].astype(np.float64)
        try:
            plane_params, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            return None, None

        if rank < 3:
            return None, None

        predicted = design @ plane_params
        residual = predicted - target
        rmse = float(np.sqrt(np.mean(residual ** 2))) if residual.size > 0 else None
        return plane_params.astype(np.float32), rmse

    def build_local_ground_surface_features(self, points, point_ground_labels):
        local_ground_height = np.zeros(points.shape[0], dtype=np.float32)
        delta_z_to_ground = np.zeros(points.shape[0], dtype=np.float32)
        ground_valid = np.zeros(points.shape[0], dtype=np.float32)

        if points.shape[0] == 0:
            return local_ground_height, delta_z_to_ground, ground_valid

        mask = common_utils.mask_points_by_range_v2(points[:, :3], self.point_cloud_range)
        if not np.any(mask):
            return local_ground_height, delta_z_to_ground, ground_valid

        min_ground_points = int(self.ground_prior_cfg.get('GROUND_VALID_MIN_GROUND_POINTS', 5))
        max_rmse = float(self.ground_prior_cfg.get('GROUND_VALID_MAX_RMSE', 0.10))
        max_support_dist = float(self.ground_prior_cfg.get('GROUND_VALID_MAX_SUPPORT_XY_DIST', 0.40))
        neighborhood_radius = int(self.ground_prior_cfg.get('GROUND_VALID_NEIGHBOR_RADIUS', 1))

        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]

        point_indices = np.where(mask)[0]
        points_in_range = points[point_indices]
        labels_in_range = point_ground_labels[point_indices] > 0.5

        x_idx = np.floor((points_in_range[:, 0] - pc_range_min[0]) / voxel_size[0]).astype(np.int32)
        y_idx = np.floor((points_in_range[:, 1] - pc_range_min[1]) / voxel_size[1]).astype(np.int32)
        valid_idx = (
            (x_idx >= 0) & (x_idx < map_w) &
            (y_idx >= 0) & (y_idx < map_h)
        )
        if not np.any(valid_idx):
            return local_ground_height, delta_z_to_ground, ground_valid

        point_indices = point_indices[valid_idx]
        points_in_range = points_in_range[valid_idx]
        labels_in_range = labels_in_range[valid_idx]
        x_idx = x_idx[valid_idx]
        y_idx = y_idx[valid_idx]

        occupied_cells = {}
        ground_cells = {}
        for local_idx, (cell_y, cell_x) in enumerate(zip(y_idx.tolist(), x_idx.tolist())):
            cell_key = (cell_y, cell_x)
            occupied_cells.setdefault(cell_key, []).append(local_idx)
            if labels_in_range[local_idx]:
                ground_cells.setdefault(cell_key, []).append(local_idx)

        for cell_key, local_point_indices in occupied_cells.items():
            cell_y, cell_x = cell_key
            support_local_indices = []
            for off_y in range(-neighborhood_radius, neighborhood_radius + 1):
                neighbor_y = cell_y + off_y
                if neighbor_y < 0 or neighbor_y >= map_h:
                    continue
                for off_x in range(-neighborhood_radius, neighborhood_radius + 1):
                    neighbor_x = cell_x + off_x
                    if neighbor_x < 0 or neighbor_x >= map_w:
                        continue
                    support_local_indices.extend(ground_cells.get((neighbor_y, neighbor_x), ()))

            if len(support_local_indices) < min_ground_points:
                continue

            support_local_indices = np.asarray(support_local_indices, dtype=np.int32)
            support_points_xyz = points_in_range[support_local_indices, :3]
            plane_params, rmse = self.fit_local_ground_plane(support_points_xyz)
            if plane_params is None or rmse is None or rmse > max_rmse:
                continue

            query_points = points_in_range[local_point_indices, :3]
            query_xy = query_points[:, :2].astype(np.float32)
            support_xy = support_points_xyz[:, :2].astype(np.float32)
            pairwise_sq_dist = (
                (query_xy[:, None, 0] - support_xy[None, :, 0]) ** 2 +
                (query_xy[:, None, 1] - support_xy[None, :, 1]) ** 2
            )
            min_support_dist = np.sqrt(pairwise_sq_dist.min(axis=1))
            reliable_mask = min_support_dist <= max_support_dist
            if not np.any(reliable_mask):
                continue

            reliable_global_indices = point_indices[np.asarray(local_point_indices, dtype=np.int32)[reliable_mask]]
            reliable_points = points[reliable_global_indices, :3]
            z_ground = (
                plane_params[0] * reliable_points[:, 0] +
                plane_params[1] * reliable_points[:, 1] +
                plane_params[2]
            )
            local_ground_height[reliable_global_indices] = z_ground.astype(np.float32)
            delta_z_to_ground[reliable_global_indices] = (
                reliable_points[:, 2] - z_ground
            ).astype(np.float32)
            ground_valid[reliable_global_indices] = 1.0

        return local_ground_height, delta_z_to_ground, ground_valid

    def attach_ground_point_features(self, data_dict):
        if data_dict.get('ground_point_feature_indices', None) is not None:
            return data_dict

        points = data_dict.get('points', None)
        if points is None:
            return data_dict

        ground_feature_indices = self.get_ground_feature_index_map()
        point_ground_labels = self.infer_ground_point_labels(points)
        local_ground_height, delta_z_to_ground, ground_valid = self.build_local_ground_surface_features(
            points=points,
            point_ground_labels=point_ground_labels
        )

        if self.ground_context_film_enabled:
            appended_features = np.stack(
                [
                    point_ground_labels.astype(np.float32),
                    delta_z_to_ground.astype(np.float32),
                    ground_valid.astype(np.float32),
                ],
                axis=1
            )
        else:
            appended_features = np.stack(
                [
                    point_ground_labels.astype(np.float32),
                    delta_z_to_ground.astype(np.float32),
                    local_ground_height.astype(np.float32),
                ],
                axis=1
            )
        data_dict['points'] = np.concatenate([points, appended_features], axis=1)
        data_dict['ground_point_feature_indices'] = ground_feature_indices
        return data_dict

    def attach_patchworkpp_point_features(self, data_dict):
        if data_dict.get('patch_infos', None) is not None:
            return data_dict

        points = data_dict.get('points', None)
        if points is None:
            return data_dict

        patch_outputs = self.infer_patchwork_outputs(points)
        point_ground_mask = patch_outputs['point_ground_mask']
        point_patch_id = patch_outputs['point_patch_id']
        patch_infos = patch_outputs['patch_infos']

        num_points = points.shape[0]
        patch_center_z = np.zeros(num_points, dtype=np.float32)
        patch_normal_z = np.zeros(num_points, dtype=np.float32)
        patch_flatness = np.zeros(num_points, dtype=np.float32)
        patch_elevation = np.zeros(num_points, dtype=np.float32)
        patch_point_count = np.zeros(num_points, dtype=np.float32)
        patch_zone_id = np.full(num_points, -1.0, dtype=np.float32)
        patch_ring_id = np.full(num_points, -1.0, dtype=np.float32)
        patch_sector_id = np.full(num_points, -1.0, dtype=np.float32)

        valid_patch = (
            point_patch_id >= 0
        ) & (
            point_patch_id < patch_infos.shape[0]
        )
        if np.any(valid_patch):
            point_patch_id_valid = point_patch_id[valid_patch]
            patch_center_z[valid_patch] = patch_infos[point_patch_id_valid, 3]
            patch_normal_z[valid_patch] = patch_infos[point_patch_id_valid, 6]
            patch_flatness[valid_patch] = patch_infos[point_patch_id_valid, 7]
            patch_elevation[valid_patch] = patch_infos[point_patch_id_valid, 8]
            patch_point_count[valid_patch] = patch_infos[point_patch_id_valid, 9]
            patch_zone_id[valid_patch] = patch_infos[point_patch_id_valid, 10]
            patch_ring_id[valid_patch] = patch_infos[point_patch_id_valid, 11]
            patch_sector_id[valid_patch] = patch_infos[point_patch_id_valid, 12]

        appended_features = np.stack(
            [
                point_ground_mask.astype(np.float32),
                patch_center_z,
                patch_normal_z,
                patch_flatness,
                patch_elevation,
                patch_point_count,
                patch_zone_id,
                patch_ring_id,
                patch_sector_id,
                point_patch_id.astype(np.float32),
            ],
            axis=1
        )
        data_dict['points'] = np.concatenate([points, appended_features], axis=1)
        data_dict['patch_infos'] = patch_infos.astype(np.float32)
        return data_dict

    def attach_custom_ground_features(self, data_dict):
        if self.patchwork_guidance_enabled or self.bev_occupancy_guidance_enabled:
            return self.attach_patchworkpp_point_features(data_dict)
        return self.attach_ground_point_features(data_dict)

    def build_object_footprint_mask(self, gt_boxes):
        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        footprint_mask = np.zeros((map_h, map_w), dtype=np.float32)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            return footprint_mask

        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]
        x_centers = pc_range_min[0] + (np.arange(map_w, dtype=np.float32) + 0.5) * voxel_size[0]
        y_centers = pc_range_min[1] + (np.arange(map_h, dtype=np.float32) + 0.5) * voxel_size[1]
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)

        for box in gt_boxes[:, :7]:
            dx = float(box[3]) * 0.5
            dy = float(box[4]) * 0.5
            if dx <= 0 or dy <= 0:
                continue

            rel_x = grid_x - float(box[0])
            rel_y = grid_y - float(box[1])
            cos_heading = float(np.cos(box[6]))
            sin_heading = float(np.sin(box[6]))
            local_x = rel_x * cos_heading + rel_y * sin_heading
            local_y = -rel_x * sin_heading + rel_y * cos_heading
            in_box = (np.abs(local_x) <= dx) & (np.abs(local_y) <= dy)
            footprint_mask[in_box] = 1.0

        return footprint_mask

    def build_class_footprint_masks(self, gt_boxes):
        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        class_masks = np.zeros((len(self.class_names), map_h, map_w), dtype=np.float32)
        if gt_boxes is None or gt_boxes.shape[0] == 0 or gt_boxes.shape[1] < 8:
            return class_masks

        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]
        x_centers = pc_range_min[0] + (np.arange(map_w, dtype=np.float32) + 0.5) * voxel_size[0]
        y_centers = pc_range_min[1] + (np.arange(map_h, dtype=np.float32) + 0.5) * voxel_size[1]
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)

        for box in gt_boxes:
            class_idx = int(round(float(box[7]))) - 1
            if class_idx < 0 or class_idx >= len(self.class_names):
                continue

            dx = float(box[3]) * 0.5
            dy = float(box[4]) * 0.5
            if dx <= 0 or dy <= 0:
                continue

            rel_x = grid_x - float(box[0])
            rel_y = grid_y - float(box[1])
            cos_heading = float(np.cos(box[6]))
            sin_heading = float(np.sin(box[6]))
            local_x = rel_x * cos_heading + rel_y * sin_heading
            local_y = -rel_x * sin_heading + rel_y * cos_heading
            in_box = (np.abs(local_x) <= dx) & (np.abs(local_y) <= dy)
            class_masks[class_idx, in_box] = 1.0

        return class_masks

    def build_patch_bev_context_maps(self, data_dict):
        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        context_map = np.zeros((5, map_h, map_w), dtype=np.float32)
        valid_mask = np.zeros((map_h, map_w), dtype=np.float32)

        points = data_dict.get('points', None)
        if points is None or points.shape[0] == 0:
            return context_map, valid_mask

        feature_offset = 4
        feature_names = {name: feature_offset + idx for idx, name in enumerate(self.point_feature_names)}
        mask = common_utils.mask_points_by_range_v2(points[:, :3], self.point_cloud_range)
        if not np.any(mask):
            return context_map, valid_mask

        points_in_range = points[mask]
        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]
        x_idx = np.floor((points_in_range[:, 0] - pc_range_min[0]) / voxel_size[0]).astype(np.int32)
        y_idx = np.floor((points_in_range[:, 1] - pc_range_min[1]) / voxel_size[1]).astype(np.int32)
        valid = (
            (x_idx >= 0) & (x_idx < map_w) &
            (y_idx >= 0) & (y_idx < map_h)
        )
        if not np.any(valid):
            return context_map, valid_mask

        points_in_range = points_in_range[valid]
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]

        total_counts = np.zeros((map_h, map_w), dtype=np.float32)
        ground_counts = np.zeros((map_h, map_w), dtype=np.float32)
        center_z_sum = np.zeros((map_h, map_w), dtype=np.float32)
        normal_z_sum = np.zeros((map_h, map_w), dtype=np.float32)
        flatness_sum = np.zeros((map_h, map_w), dtype=np.float32)
        elevation_sum = np.zeros((map_h, map_w), dtype=np.float32)

        np.add.at(total_counts, (y_idx, x_idx), 1.0)
        np.add.at(ground_counts, (y_idx, x_idx), points_in_range[:, feature_names['is_ground']])
        np.add.at(center_z_sum, (y_idx, x_idx), points_in_range[:, feature_names['patch_center_z']])
        np.add.at(normal_z_sum, (y_idx, x_idx), points_in_range[:, feature_names['patch_normal_z']])
        np.add.at(flatness_sum, (y_idx, x_idx), points_in_range[:, feature_names['patch_flatness']])
        np.add.at(elevation_sum, (y_idx, x_idx), points_in_range[:, feature_names['patch_elevation']])

        valid_cells = total_counts > 0
        valid_mask[valid_cells] = 1.0
        denom = np.clip(total_counts, a_min=1.0, a_max=None)

        ground_ratio_map = ground_counts / denom
        center_z_map = center_z_sum / denom
        normal_z_map = normal_z_sum / denom
        flatness_map = flatness_sum / denom
        elevation_map = elevation_sum / denom

        center_z_norm = float(self.ground_prior_cfg.get('PATCH_CENTER_Z_NORM', 3.0))
        elevation_norm = float(self.ground_prior_cfg.get('PATCH_ELEVATION_NORM', 3.0))
        flatness_norm = float(self.ground_prior_cfg.get('PATCH_FLATNESS_NORM', 1.0))
        if center_z_norm > 0:
            center_z_map = np.clip(center_z_map / center_z_norm, -1.0, 1.0)
        normal_z_map = np.clip(normal_z_map, -1.0, 1.0)
        if flatness_norm > 0:
            flatness_map = np.clip(flatness_map / flatness_norm, 0.0, 1.0)
        if elevation_norm > 0:
            elevation_map = np.clip(elevation_map / elevation_norm, -1.0, 1.0)

        context_map = np.stack(
            [ground_ratio_map, center_z_map, normal_z_map, flatness_map, elevation_map],
            axis=0
        ).astype(np.float32)
        return context_map, valid_mask

    def build_soft_occupancy_supervision(self, gt_boxes):
        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        occupancy_target = np.zeros((map_h, map_w), dtype=np.float32)
        occupancy_weight = np.zeros((map_h, map_w), dtype=np.float32)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            return occupancy_target, occupancy_weight

        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]
        x_centers = pc_range_min[0] + (np.arange(map_w, dtype=np.float32) + 0.5) * voxel_size[0]
        y_centers = pc_range_min[1] + (np.arange(map_h, dtype=np.float32) + 0.5) * voxel_size[1]
        grid_x, grid_y = np.meshgrid(x_centers, y_centers)

        softness_ratio = float(self.ground_prior_cfg.get('SOFT_OCCUPANCY_EDGE_RATIO', 0.15))
        min_softness_cells = float(self.ground_prior_cfg.get('SOFT_OCCUPANCY_MIN_EDGE_CELLS', 1.0))
        min_target_value = float(self.ground_prior_cfg.get('SOFT_OCCUPANCY_MIN_TARGET_VALUE', 1e-4))
        min_softness_x = voxel_size[0] * min_softness_cells
        min_softness_y = voxel_size[1] * min_softness_cells

        for box in gt_boxes[:, :7]:
            half_dx = float(box[3]) * 0.5
            half_dy = float(box[4]) * 0.5
            if half_dx <= 0 or half_dy <= 0:
                continue

            rel_x = grid_x - float(box[0])
            rel_y = grid_y - float(box[1])
            cos_heading = float(np.cos(box[6]))
            sin_heading = float(np.sin(box[6]))
            local_x = rel_x * cos_heading + rel_y * sin_heading
            local_y = -rel_x * sin_heading + rel_y * cos_heading

            softness_x = max(half_dx * softness_ratio, min_softness_x)
            softness_y = max(half_dy * softness_ratio, min_softness_y)
            soft_x_arg = np.clip((np.abs(local_x) - half_dx) / max(softness_x, 1e-6), -60.0, 60.0)
            soft_y_arg = np.clip((np.abs(local_y) - half_dy) / max(softness_y, 1e-6), -60.0, 60.0)
            soft_x = 1.0 / (1.0 + np.exp(soft_x_arg))
            soft_y = 1.0 / (1.0 + np.exp(soft_y_arg))
            occupancy = (soft_x * soft_y).astype(np.float32)
            occupancy[occupancy < min_target_value] = 0.0

            occupancy_sum = float(occupancy.sum())
            if occupancy_sum <= 0:
                continue
            occupancy_target = np.maximum(occupancy_target, occupancy)
            occupancy_weight = occupancy_weight + (occupancy / occupancy_sum)

        return occupancy_target.astype(np.float32), occupancy_weight.astype(np.float32)

    def build_bev_occupancy_guidance_supervision(self, data_dict):
        if not self.bev_occupancy_guidance_enabled or data_dict.get('points', None) is None:
            return data_dict

        context_map, valid_mask = self.build_patch_bev_context_maps(data_dict)
        data_dict['bev_occupancy_context_map'] = context_map.astype(np.float32)
        data_dict['bev_occupancy_context_valid_mask'] = valid_mask.astype(np.float32)

        gt_boxes = data_dict.get('gt_boxes', None)
        occupancy_target, occupancy_weight = self.build_soft_occupancy_supervision(gt_boxes)
        data_dict['bev_occupancy_target_map'] = occupancy_target.astype(np.float32)
        data_dict['bev_occupancy_positive_weight_map'] = occupancy_weight.astype(np.float32)
        data_dict['bev_occupancy_class_mask'] = self.build_class_footprint_masks(gt_boxes).astype(np.float32)
        return data_dict

    def build_ground_defect_supervision(self, data_dict):
        if not self.ground_defect_guidance_enabled or data_dict.get('points', None) is None:
            return data_dict

        ground_feature_indices = data_dict.get('ground_point_feature_indices', None)
        if ground_feature_indices is None:
            raise KeyError('ground_point_feature_indices is missing before building ground defect supervision')

        points = data_dict['points']
        map_h = int(self.grid_size[1])
        map_w = int(self.grid_size[0])
        voxel_size = np.asarray(self.voxel_size, dtype=np.float32)
        pc_range_min = self.point_cloud_range[:3]
        mask = common_utils.mask_points_by_range_v2(points[:, :3], self.point_cloud_range)

        ground_ratio_map = np.zeros((map_h, map_w), dtype=np.float32)
        ground_height_map = np.zeros((map_h, map_w), dtype=np.float32)
        height_residual_map = np.zeros((map_h, map_w), dtype=np.float32)
        near_non_ground_map = np.zeros((map_h, map_w), dtype=np.float32)
        valid_ground_mask = np.zeros((map_h, map_w), dtype=np.float32)
        boundary_valid_mask = np.zeros((map_h, map_w), dtype=np.float32)

        if np.any(mask):
            points_in_range = points[mask]
            x_idx = np.floor((points_in_range[:, 0] - pc_range_min[0]) / voxel_size[0]).astype(np.int32)
            y_idx = np.floor((points_in_range[:, 1] - pc_range_min[1]) / voxel_size[1]).astype(np.int32)
            valid = (
                (x_idx >= 0) & (x_idx < map_w) &
                (y_idx >= 0) & (y_idx < map_h)
            )
            if np.any(valid):
                points_in_range = points_in_range[valid]
                x_idx = x_idx[valid]
                y_idx = y_idx[valid]

                total_counts = np.zeros((map_h, map_w), dtype=np.float32)
                ground_counts = np.zeros((map_h, map_w), dtype=np.float32)
                ground_height_sum = np.zeros((map_h, map_w), dtype=np.float32)
                residual_sum = np.zeros((map_h, map_w), dtype=np.float32)
                residual_count = np.zeros((map_h, map_w), dtype=np.float32)
                near_non_ground_counts = np.zeros((map_h, map_w), dtype=np.float32)

                is_ground = points_in_range[:, ground_feature_indices['is_ground']] > 0.5
                delta_z = points_in_range[:, ground_feature_indices['delta_z_to_ground']]
                local_ground_height = points_in_range[:, ground_feature_indices['local_ground_height']]

                np.add.at(total_counts, (y_idx, x_idx), 1.0)
                np.add.at(ground_counts, (y_idx, x_idx), is_ground.astype(np.float32))
                np.add.at(
                    ground_height_sum,
                    (y_idx[is_ground], x_idx[is_ground]),
                    local_ground_height[is_ground]
                )

                valid_cells = total_counts > 0
                valid_ground_cells = ground_counts > 0
                ground_ratio_map[valid_cells] = ground_counts[valid_cells] / total_counts[valid_cells]
                valid_ground_mask = valid_ground_cells.astype(np.float32)
                ground_height_map[valid_ground_cells] = (
                    ground_height_sum[valid_ground_cells] / ground_counts[valid_ground_cells]
                )
                boundary_valid_mask = valid_cells.astype(np.float32)

                non_ground = np.logical_not(is_ground) & valid_ground_cells[y_idx, x_idx]
                if np.any(non_ground):
                    delta_clip = float(self.ground_prior_cfg.get('DELTA_Z_CLIP', 3.0))
                    near_threshold = float(self.ground_prior_cfg.get('NEAR_GROUND_DELTA_Z', 0.5))
                    clipped_delta = np.clip(delta_z[non_ground], 0.0, delta_clip)
                    np.add.at(residual_sum, (y_idx[non_ground], x_idx[non_ground]), clipped_delta)
                    np.add.at(residual_count, (y_idx[non_ground], x_idx[non_ground]), 1.0)

                    near_non_ground = np.logical_and(
                        delta_z[non_ground] > 0.0,
                        delta_z[non_ground] <= near_threshold
                    )
                    np.add.at(
                        near_non_ground_counts,
                        (y_idx[non_ground][near_non_ground], x_idx[non_ground][near_non_ground]),
                        1.0
                    )

                residual_cells = residual_count > 0
                height_residual_map[residual_cells] = residual_sum[residual_cells] / residual_count[residual_cells]

                density_norm = float(np.log1p(max(1.0, near_non_ground_counts.max())))
                if density_norm > 0:
                    near_non_ground_map = np.log1p(near_non_ground_counts) / density_norm

        ground_height_norm = float(self.ground_prior_cfg.get('GROUND_HEIGHT_NORM', 3.0))
        delta_z_norm = float(self.ground_prior_cfg.get('DELTA_Z_NORM', 2.0))
        if ground_height_norm > 0:
            ground_height_map = np.clip(ground_height_map / ground_height_norm, -1.0, 1.0)
        if delta_z_norm > 0:
            height_residual_map = np.clip(height_residual_map / delta_z_norm, 0.0, 1.0)

        ground_boundary_map = self.build_ground_boundary_map(
            ground_ratio_map=ground_ratio_map,
            ground_valid_mask_map=boundary_valid_mask,
            ratio_threshold=float(self.ground_prior_cfg.get('BOUNDARY_GROUND_RATIO_THRESHOLD', 0.5))
        )

        data_dict['ground_defect_bev_map'] = np.stack(
            [
                ground_ratio_map,
                ground_height_map,
                height_residual_map,
                near_non_ground_map,
                ground_boundary_map,
            ],
            axis=0
        ).astype(np.float32)
        data_dict['ground_defect_valid_mask'] = valid_ground_mask.astype(np.float32)
        data_dict['ground_defect_footprint_mask'] = self.build_object_footprint_mask(
            data_dict.get('gt_boxes', None)
        ).astype(np.float32)
        return data_dict

    def apply_training_augmentor_with_ground_features(self, data_dict):
        assert self.data_augmentor is not None
        augmentor_dict = {
            **data_dict,
            'gt_boxes_mask': np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        }
        ground_features_attached = False
        has_gt_sampler = any(
            isinstance(cur_augmentor, database_sampler.DataBaseSampler)
            for cur_augmentor in self.data_augmentor.data_augmentor_queue
        )

        if self.custom_ground_feature_enabled and not has_gt_sampler:
            augmentor_dict = self.attach_custom_ground_features(augmentor_dict)
            ground_features_attached = True

        for cur_augmentor in self.data_augmentor.data_augmentor_queue:
            augmentor_dict = cur_augmentor(data_dict=augmentor_dict)
            if (
                self.custom_ground_feature_enabled and
                not ground_features_attached and
                isinstance(cur_augmentor, database_sampler.DataBaseSampler)
            ):
                augmentor_dict = self.attach_custom_ground_features(augmentor_dict)
                ground_features_attached = True

        if self.custom_ground_feature_enabled and not ground_features_attached:
            augmentor_dict = self.attach_custom_ground_features(augmentor_dict)

        augmentor_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            augmentor_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'road_plane' in augmentor_dict:
            augmentor_dict.pop('road_plane')
        if 'gt_boxes_mask' in augmentor_dict:
            gt_boxes_mask = augmentor_dict.pop('gt_boxes_mask')
            augmentor_dict['gt_boxes'] = augmentor_dict['gt_boxes'][gt_boxes_mask]
            augmentor_dict['gt_names'] = augmentor_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in augmentor_dict:
                augmentor_dict['gt_boxes2d'] = augmentor_dict['gt_boxes2d'][gt_boxes_mask]

        return augmentor_dict

    def prepare_data(self, data_dict):
        if not self.custom_ground_feature_enabled:
            return super().prepare_data(data_dict)

        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            calib = data_dict.get('calib', None)
            data_dict = self.apply_training_augmentor_with_ground_features(data_dict)
            if calib is not None:
                data_dict['calib'] = calib
        elif data_dict.get('points', None) is not None:
            data_dict = self.attach_custom_ground_features(data_dict)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes
            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if self.ground_defect_guidance_enabled:
            data_dict = self.build_ground_defect_supervision(data_dict)
        if self.bev_occupancy_guidance_enabled:
            data_dict = self.build_bev_occupancy_guidance_supervision(data_dict)

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        data_dict.pop('ground_point_feature_indices', None)
        return data_dict

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points
            if self.legacy_ground_prior_enabled:
                input_dict['point_ground_labels'] = self.infer_ground_point_labels(points)

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)
        if self.legacy_ground_prior_enabled and 'coarse_observed_ground_prior' not in data_dict:
            point_ground_labels = data_dict.pop('point_ground_labels', None)
            if point_ground_labels is not None:
                data_dict.update(
                    self.build_coarse_observed_ground_prior(
                        points=data_dict['points'],
                        point_ground_labels=point_ground_labels
                    )
                )

        data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = Path('./')  #(Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
