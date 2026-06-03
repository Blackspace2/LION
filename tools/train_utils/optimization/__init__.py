from functools import partial

import torch.nn as nn
import torch.nn.parameter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def _cfg_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _matches_group(name, group_cfg):
    prefixes = _cfg_list(group_cfg.get('PREFIXES', None))
    keywords = _cfg_list(group_cfg.get('KEYWORDS', None))
    exclude_prefixes = _cfg_list(group_cfg.get('EXCLUDE_PREFIXES', None))
    exclude_keywords = _cfg_list(group_cfg.get('EXCLUDE_KEYWORDS', None))

    if any(name.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if any(keyword in name for keyword in exclude_keywords):
        return False

    has_positive_rule = bool(prefixes) or bool(keywords)
    if not has_positive_rule:
        raise ValueError('OPTIMIZATION.PARAM_GROUPS entries require PREFIXES or KEYWORDS')
    return any(name.startswith(prefix) for prefix in prefixes) or any(keyword in name for keyword in keywords)


def _build_named_param_groups(model, optim_cfg):
    group_cfgs = list(optim_cfg.get('PARAM_GROUPS', []))
    if len(group_cfgs) == 0:
        return None

    named_params = list(model.named_parameters())
    assigned_ids = set()
    param_groups = []
    base_lr = float(optim_cfg.LR)
    base_wd = float(optim_cfg.WEIGHT_DECAY)

    for idx, group_cfg in enumerate(group_cfgs):
        group_name = group_cfg.get('NAME', f'group_{idx}')
        matched = []
        matched_names = []
        for name, param in named_params:
            if id(param) in assigned_ids:
                continue
            if _matches_group(name, group_cfg):
                matched.append(param)
                matched_names.append(name)
                assigned_ids.add(id(param))

        if len(matched) == 0:
            if bool(group_cfg.get('ALLOW_EMPTY', False)):
                continue
            raise ValueError(f'OPTIMIZATION.PARAM_GROUPS[{group_name}] matched no parameters')

        lr_mult = float(group_cfg.get('LR_MULT', 1.0))
        weight_decay = float(group_cfg.get('WEIGHT_DECAY', base_wd))
        param_groups.append({
            'params': matched,
            'lr': base_lr * lr_mult,
            'weight_decay': weight_decay,
            'name': group_name,
            'lr_mult': lr_mult,
            'param_count': sum(param.numel() for param in matched),
            'sample_names': matched_names[:8],
        })

    default_params = []
    default_names = []
    for name, param in named_params:
        if id(param) not in assigned_ids:
            default_params.append(param)
            default_names.append(name)

    if len(default_params) > 0:
        param_groups.insert(0, {
            'params': default_params,
            'lr': base_lr,
            'weight_decay': base_wd,
            'name': 'default',
            'lr_mult': 1.0,
            'param_count': sum(param.numel() for param in default_params),
            'sample_names': default_names[:8],
        })
    return param_groups


def build_optimizer(model, optim_cfg):
    named_param_groups = _build_named_param_groups(model, optim_cfg)
    if optim_cfg.OPTIMIZER == 'adam':
        params = named_param_groups if named_param_groups is not None else model.parameters()
        optimizer = optim.Adam(params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'adamw':
        params = named_param_groups if named_param_groups is not None else model.parameters()
        optimizer = optim.AdamW(params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        params = named_param_groups if named_param_groups is not None else model.parameters()
        optimizer = optim.SGD(
            params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        if named_param_groups is not None:
            raise ValueError(
                'OPTIMIZATION.PARAM_GROUPS is only supported for standard adam/adamw/sgd. '
                'adam_onecycle filters trainable params at optimizer creation and is not suitable '
                'for staged PaSS image unfreezing.'
            )
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        def named_children(m, prefix):
            if num_children(m[1]):
                mm_list = list()
                for mm in m[1].named_children():
                    mm_list.extend(named_children(mm, prefix + f'.{m[0]}' if prefix != '' else f'{m[0]}'))
                return mm_list
            else:
                mm_list = list()
                for n, _ in m[1].named_parameters():
                    mm_list.append(prefix + f'.{m[0]}.{n}' if prefix != '' else f'{m[0]}.{n}')
                return mm_list

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))

        modules = named_children(('', model), '')

        params = model.named_parameters()
        other_params = list()

        for p in params:
            if p[0] not in modules:
                name = p[0].split('.')
                m = model
                for n in name:
                    if n.isnumeric():
                        m = m[int(n)]
                    else:
                        m = getattr(m, n)
                p = p[1]
                if isinstance(m, torch.nn.parameter.Parameter) and hasattr(m, '_no_weight_decay'):
                    setattr(p, '_no_weight_decay', m._no_weight_decay)
                other_params.append(p)

        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), params=iter(other_params), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    scheduler_type = getattr(optim_cfg, 'SCHEDULER', '')

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    def constant_lr(_):
        return 1.0

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif scheduler_type in ['constant', 'none']:
        lr_scheduler = lr_sched.LambdaLR(optimizer, constant_lr, last_epoch=last_epoch)
    elif scheduler_type == 'cosine':
        lr_scheduler = lr_sched.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=optim_cfg.LR_CLIP,
            last_epoch=last_epoch,
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
