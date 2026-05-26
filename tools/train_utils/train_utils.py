import glob
import os
import shutil

import torch
import tqdm
import time
import contextlib

from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils

try:
    import torch.cuda.amp
except:
    # Make sure the torch version is latest enough to support mixed precision training
    pass


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in self._state_dict(model).items()
        }

    @staticmethod
    def _state_dict(model):
        module = model.module if hasattr(model, 'module') else model
        return module.state_dict()

    @torch.no_grad()
    def update(self, model):
        model_state = self._state_dict(model)
        for name, tensor in model_state.items():
            if name not in self.shadow:
                self.shadow[name] = tensor.detach().clone()
                continue
            if torch.is_floating_point(tensor):
                self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(tensor.detach())

    def state_dict(self):
        return {
            name: tensor.detach().clone()
            for name, tensor in self.shadow.items()
        }



def _set_frozen_batchnorm_eval(model):
    module = model.module if hasattr(model, 'module') else model
    if not getattr(module, '_freeze_bn_when_frozen', False):
        return

    for submodule in module.modules():
        if not isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            continue
        params = list(submodule.parameters(recurse=False))
        if len(params) > 0 and all(not param.requires_grad for param in params):
            submodule.eval()


def _snapshot_bn_running_buffers(model):
    module = model.module if hasattr(model, 'module') else model
    return {
        name: buf.detach().clone()
        for name, buf in module.named_buffers()
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name
    }


def _restore_bn_running_buffers(model, snapshot):
    module = model.module if hasattr(model, 'module') else model
    buffers = dict(module.named_buffers())
    for name, saved in snapshot.items():
        if name in buffers:
            buffers[name].copy_(saved)


def _collect_named_grad_norms(model):
    module = model.module if hasattr(model, 'module') else model
    model_cfg = getattr(module, 'model_cfg', None)
    if model_cfg is None:
        return {}

    keywords = []
    backbone_cfg = getattr(model_cfg, 'BACKBONE_3D', None)
    if backbone_cfg is not None:
        guided_cfg = backbone_cfg.get('GROUND_GUIDED_DIFFUSION', None)
        if guided_cfg is not None and guided_cfg.get('ENABLED', False):
            keywords.extend(
                list(
                    guided_cfg.get(
                        'GRAD_LOG_KEYWORDS',
                        ['response_proj', 'prior_alpha_logit', 'prior_trust_logit', 'diffusion_feature_scale_logit']
                    )
                )
            )
        patchwork_cfg = backbone_cfg.get('PATCHWORK_GUIDANCE', None)
        if patchwork_cfg is not None and patchwork_cfg.get('ENABLED', False):
            keywords.extend(
                list(
                    patchwork_cfg.get(
                        'GRAD_LOG_KEYWORDS',
                        [
                            'backbone_3d.patch_token_encoder',
                            'backbone_3d.patch_context_encoder',
                            'backbone_3d.patch_topology_embedding',
                            'backbone_3d.patch_guidance_pre',
                            'backbone_3d.patch_guidance_logits',
                            'backbone_3d.patch_guidance_gates',
                            'backbone_3d.patch_guidance_residuals',
                            'backbone_3d.patch_guidance_residual_scale',
                        ]
                    )
                )
            )

    map_to_bev_cfg = getattr(model_cfg, 'MAP_TO_BEV', None)
    if map_to_bev_cfg is not None:
        map_to_bev_name = map_to_bev_cfg.get('NAME', None)
        if map_to_bev_name == 'GroundDefectHeightCompression':
            keywords.extend(
                list(
                    map_to_bev_cfg.get(
                        'GRAD_LOG_KEYWORDS',
                        [
                            'map_to_bev_module.defect_encoder',
                            'map_to_bev_module.defect_head',
                            'map_to_bev_module.gate_head',
                            'map_to_bev_module.residual_head',
                            'map_to_bev_module.residual_scale',
                        ]
                    )
                )
            )
        elif map_to_bev_name == 'BEVOccupancyGuidanceHeightCompression':
            keywords.extend(
                list(
                    map_to_bev_cfg.get(
                        'GRAD_LOG_KEYWORDS',
                        [
                            'map_to_bev_module.context_encoder',
                            'map_to_bev_module.occupancy_head',
                            'map_to_bev_module.inject_head',
                            'map_to_bev_module.gate_head',
                            'map_to_bev_module.fusion_head',
                        ]
                    )
                )
            )

    if len(keywords) == 0:
        return {}

    grad_norms = {}
    total_sq_norm = 0.0
    total_matches = 0
    for keyword in keywords:
        sq_norm = None
        for name, param in module.named_parameters():
            if keyword not in name or param.grad is None:
                continue
            grad_value = param.grad.detach().float().norm().pow(2)
            sq_norm = grad_value if sq_norm is None else sq_norm + grad_value
        if sq_norm is not None:
            norm_value = sq_norm.sqrt()
            grad_norms[f'grad_norm/{keyword}'] = float(norm_value.item())
            total_sq_norm += float(sq_norm.item())
            total_matches += 1

    if total_matches > 0:
        grad_norms['grad_norm/auxiliary_total'] = total_sq_norm ** 0.5
    return grad_norms


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None,
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False, fp16=False,
                    model_ema=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        loss_disp = common_utils.AverageMeter()
        # just for centerhead
        hm_loss_disp = common_utils.AverageMeter()
        loc_loss_disp = common_utils.AverageMeter()
        rcnn_cls_loss_disp = common_utils.AverageMeter()
        rcnn_reg_loss_disp = common_utils.AverageMeter()


    amp_ctx = contextlib.nullcontext()
    if fp16:
        scaler = torch.cuda.amp.grad_scaler.GradScaler(init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**12))
        amp_ctx = torch.cuda.amp.autocast()


    end = time.time()
    consecutive_nonfinite_skips = 0
    max_consecutive_nonfinite_skips = int(os.environ.get('LION_MAX_CONSECUTIVE_NONFINITE_SKIP', '20'))
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        batch['cur_epoch'] = int(cur_epoch) if cur_epoch is not None else -1
        batch['total_epochs'] = int(total_epochs) if total_epochs is not None else 0

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        _set_frozen_batchnorm_eval(model)
        optimizer.zero_grad()

        skipped_nonfinite = False
        did_optimizer_step = False
        total_norm = torch.zeros((), device='cuda' if torch.cuda.is_available() else 'cpu')
        bn_running_snapshot = _snapshot_bn_running_buffers(model)
        with amp_ctx:
            try:
                loss, tb_dict, disp_dict = model_func(model, batch)
            except torch.cuda.OutOfMemoryError as e:
                skipped_nonfinite = True
                _restore_bn_running_buffers(model, bn_running_snapshot)
                optimizer.zero_grad()
                loss = torch.zeros((), device='cuda' if torch.cuda.is_available() else 'cpu')
                tb_dict = {}
                disp_dict = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if rank == 0 and logger is not None:
                    logger.warning(
                        f'Skip CUDA OOM batch at epoch={cur_epoch}, iter={cur_it}, '
                        f'acc_iter={accumulated_iter}: {e}'
                    )
            if skipped_nonfinite:
                pass
            elif not torch.isfinite(loss).all():
                skipped_nonfinite = True
                _restore_bn_running_buffers(model, bn_running_snapshot)
                if rank == 0 and logger is not None:
                    logger.warning(
                        f'Skip non-finite loss at epoch={cur_epoch}, iter={cur_it}, '
                        f'acc_iter={accumulated_iter}, loss={loss.item()}'
                    )
                optimizer.zero_grad()
                loss = loss.detach()
                tb_dict = {}
                disp_dict = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif fp16:
                assert loss.dtype is torch.float32
                scaler.scale(loss).backward()
                # unscale gradient for clip gradient
                scaler.unscale_(optimizer)
                total_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                tb_dict.update(_collect_named_grad_norms(model))
                if torch.isfinite(total_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    did_optimizer_step = True
                else:
                    skipped_nonfinite = True
                    optimizer.zero_grad()
                    scaler.update()
                    if rank == 0 and logger is not None:
                        logger.warning(
                            f'Skip non-finite grad norm at epoch={cur_epoch}, iter={cur_it}, '
                            f'acc_iter={accumulated_iter}, norm={total_norm.item()}'
                        )
            else:
                loss.backward()
                total_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                tb_dict.update(_collect_named_grad_norms(model))
                if torch.isfinite(total_norm):
                    optimizer.step()
                    did_optimizer_step = True
                else:
                    skipped_nonfinite = True
                    optimizer.zero_grad()
                    if rank == 0 and logger is not None:
                        logger.warning(
                            f'Skip non-finite grad norm at epoch={cur_epoch}, iter={cur_it}, '
                            f'acc_iter={accumulated_iter}, norm={total_norm.item()}'
                        )

        accumulated_iter += 1
        if did_optimizer_step and model_ema is not None:
            model_ema.update(model)
        # assert not torch.isnan(loss)
        if skipped_nonfinite:
            consecutive_nonfinite_skips += 1
            if consecutive_nonfinite_skips >= max_consecutive_nonfinite_skips:
                raise RuntimeError(
                    f'Abort training after {consecutive_nonfinite_skips} consecutive non-finite/OOM skips '
                    f'at epoch={cur_epoch}, iter={cur_it}. Resume from an earlier clean checkpoint with a '
                    f'lower LR, smaller batch size, or fp32.'
                )
        else:
            consecutive_nonfinite_skips = 0

        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            if not skipped_nonfinite:
                loss_disp.update(loss.item())
            
            # for centerhead
            if 'hm_loss_head_0' in list(tb_dict.keys()) and 'loc_loss_head_0' in list(tb_dict.keys()):
                hm_loss_disp.update(tb_dict['hm_loss_head_0'])
                loc_loss_disp.update(tb_dict['loc_loss_head_0'])
                disp_dict.update({
                'loss_hm': f'{hm_loss_disp.avg:.4f}', 'loss_loc': f'{loc_loss_disp.avg:.4f}'})
            if 'rcnn_loss_reg' in list(tb_dict.keys()) and 'rcnn_loss_cls' in list(tb_dict.keys()):
                rcnn_cls_loss_disp.update(tb_dict['rcnn_loss_cls'])
                rcnn_reg_loss_disp.update(tb_dict['rcnn_loss_reg'])
                disp_dict.update({
                'loss_rcnn_cls': f'{rcnn_cls_loss_disp.avg:.4f}', 'loss_rcnn_reg': f'{rcnn_reg_loss_disp.avg:.4f}'})
            disp_dict.update({
                'loss': loss_disp.avg if loss_disp.count > 0 else loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                'norm': total_norm.item(), 'skipped_nonfinite': int(skipped_nonfinite)
            })

            if use_logger_to_record:
                if (accumulated_iter % logger_iter_interval == 0 and cur_it != start_it) or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    disp_str = ', '.join([f'{key}={val}' for key, val in disp_dict.items() if key != 'lr'])
                    disp_str += f', lr={disp_dict["lr"]}'
                    batch_size = batch.get('batch_size', None)
                    logger.info(f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, '
                                f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)}, '
                                f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}, '
                                f'{disp_str}')
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        if shutil.which('gpustat') is not None:
                            gpu_info = os.popen('gpustat').read()
                            logger.info(gpu_info)
                    
                    loss_disp.reset()  # WHY
                    hm_loss_disp.reset()
                    loc_loss_disp.reset()
                    rcnn_cls_loss_disp.reset()
                    rcnn_reg_loss_disp.reset()
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                if not skipped_nonfinite:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('train/grad_norm', total_norm.item(), accumulated_iter)
                tb_log.add_scalar('train/nonfinite_skip', int(skipped_nonfinite), accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                if not skipped_nonfinite:
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # save intermediate ckpt every {ckpt_save_time_interval} seconds
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, stop_epoch, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None,
                show_gpu_stat=False, fp16=False, cfg=None, model_ema=None, save_ema_as_model=False):
    accumulated_iter = start_iter

    augment_disable_flag = False
    with tqdm.trange(start_epoch, stop_epoch, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            hook_config = cfg.get('HOOK', None) 
            if hook_config is not None:
                DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
                if DisableAugmentationHook is not None:
                    num_last_epochs = cfg.HOOK.DisableAugmentationHook.NUM_LAST_EPOCHS
                    if (total_epochs - num_last_epochs) <= cur_epoch and not augment_disable_flag:
                        from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
                        from pathlib import Path
                        DISABLE_AUG_LIST = cfg.HOOK.DisableAugmentationHook.DISABLE_AUG_LIST
                        dataset_cfg=cfg.DATA_CONFIG
                        # This hook turns off some data augmentation strategies. 
                        logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                        dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                        class_names=cfg.CLASS_NAMES
                        root_path = Path(dataset_cfg.DATA_PATH)
                        new_data_augmentor = DataAugmentor(root_path, dataset_cfg.DATA_AUGMENTOR, class_names, logger=logger)
                        dataloader_iter._dataset.data_augmentor = new_data_augmentor
                        augment_disable_flag = True


            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == stop_epoch),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,

                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                fp16=fp16,
                model_ema=model_ema
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                if model_ema is not None:
                    ema_ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d_ema' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(
                            model,
                            optimizer,
                            trained_epoch,
                            accumulated_iter,
                            model_state_override=model_ema.state_dict()
                        ),
                        filename=ema_ckpt_name,
                    )
                    if save_ema_as_model:
                        raw_ckpt_name = ckpt_save_dir / ('raw_checkpoint_epoch_%d' % trained_epoch)
                        os.replace(str(ckpt_name) + '.pth', str(raw_ckpt_name) + '.pth')
                        os.replace(str(ema_ckpt_name) + '.pth', str(ckpt_name) + '.pth')


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, model_state_override=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model_state_override is not None:
        model_state = model_state_to_cpu(model_state_override)
    elif model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
