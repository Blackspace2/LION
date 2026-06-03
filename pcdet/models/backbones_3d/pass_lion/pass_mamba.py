import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .delta_b_mod import DeltaBMod

try:
    from timm.models.layers import DropPath
except ImportError:
    class DropPath(nn.Identity):
        pass

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError:
    selective_scan_fn, selective_scan_ref = None, None


def _inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class PaSSMamba(nn.Module):
    """Bidirectional Mamba split path with optional PaSS delta/B modulation."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank='auto',
        dt_min=0.001,
        dt_max=0.1,
        dt_init='random',
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        c_img=64,
        rank=4,
        b_residual_clip=1.0,
        use_ref_scan=False,
        device=None,
        dtype=None,
        init_layer_scale=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else int(dt_rank)
        self.use_ref_scan = bool(use_ref_scan)
        self.c_img = int(c_img)
        self.init_layer_scale = init_layer_scale

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_b = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.A_log = nn.Parameter(self._make_a_log(device=device))
        self.A_log._no_weight_decay = True
        self.A_b_log = nn.Parameter(self._make_a_log(device=device))
        self.A_b_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.D_b._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((self.d_model,), **factory_kwargs))

        self.delta_b_mod = DeltaBMod(
            c_img=self.c_img,
            d_inner=self.d_inner,
            d_state=self.d_state,
            rank=rank,
            b_residual_clip=b_residual_clip,
        )
        if device is not None or dtype is not None:
            self.delta_b_mod.to(**factory_kwargs)
        self._reset_dt_parameters(dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

    def _make_a_log(self, device=None):
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        a = a.view(1, self.d_state).repeat(self.d_inner, 1).contiguous()
        return torch.log(a)

    def _reset_dt_parameters(self, dt_min, dt_max, dt_init, dt_scale, dt_init_floor):
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        for layer in (self.dt_proj, self.dt_proj_b):
            if dt_init == 'constant':
                nn.init.constant_(layer.weight, dt_init_std)
            elif dt_init == 'random':
                nn.init.uniform_(layer.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError('Unsupported dt_init')
            dt = torch.exp(
                torch.rand(self.d_inner, device=layer.weight.device, dtype=layer.weight.dtype)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            with torch.no_grad():
                layer.bias.copy_(_inverse_softplus(dt))
            layer.bias._no_reinit = True

    def _causal_conv(self, x, conv):
        if causal_conv1d_fn is not None and x.is_cuda:
            return causal_conv1d_fn(
                x,
                conv.weight.squeeze(1),
                conv.bias,
                activation='silu',
            )
        out = F.conv1d(x, conv.weight, conv.bias, padding=self.d_conv - 1, groups=self.d_inner)
        out = out[..., :x.shape[-1]]
        return F.silu(out)

    def _scan(self, conv_out, delta, a_log, b_mod, c_raw, d_skip, z):
        a = -torch.exp(a_log.float())
        if (not self.use_ref_scan) and selective_scan_fn is not None and conv_out.is_cuda:
            b_scan = b_mod.unsqueeze(1).contiguous() if b_mod.dim() == 3 else b_mod.contiguous()
            c_scan = c_raw.unsqueeze(1).expand(-1, b_scan.shape[1], -1, -1).contiguous()
            return selective_scan_fn(
                conv_out,
                delta,
                a,
                b_scan,
                c_scan,
                d_skip.float(),
                z=z,
                delta_bias=None,
                delta_softplus=False,
            )
        if selective_scan_ref is None:
            return self._scan_ref_fallback(conv_out, delta, a, b_mod, c_raw, d_skip.float(), z)
        return selective_scan_ref(
            conv_out,
            delta,
            a,
            b_mod,
            c_raw,
            d_skip.float(),
            z=z,
            delta_bias=None,
            delta_softplus=False,
        )

    @staticmethod
    def _scan_ref_fallback(u, delta, a, b_mod, c_raw, d_skip, z):
        dtype_in = u.dtype
        u_f = u.float()
        delta_f = delta.float()
        a_f = a.float()
        b_f = b_mod.float()
        c_f = c_raw.float()
        batch, dim, seqlen = u.shape
        d_state = a.shape[1]
        state = torch.zeros((batch, dim, d_state), dtype=torch.float32, device=u.device)
        outputs = []
        for idx in range(seqlen):
            delta_a = torch.exp(delta_f[:, :, idx].unsqueeze(-1) * a_f.unsqueeze(0))
            if b_f.dim() == 3:
                delta_b_u = delta_f[:, :, idx].unsqueeze(-1) * b_f[:, :, idx].unsqueeze(1) * u_f[:, :, idx].unsqueeze(-1)
            else:
                delta_b_u = delta_f[:, :, idx].unsqueeze(-1) * b_f[:, :, :, idx] * u_f[:, :, idx].unsqueeze(-1)
            state = state * delta_a + delta_b_u
            y = torch.einsum('bdn,bn->bd', state, c_f[:, :, idx])
            y = y + d_skip.float().unsqueeze(0) * u_f[:, :, idx]
            outputs.append(y)
        out = torch.stack(outputs, dim=-1)
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype=dtype_in)

    def _prepare_pass_inputs(self, hidden_states, v_img, fov, pass_enabled, rho):
        batch, seqlen, _ = hidden_states.shape
        rho_is_zero = DeltaBMod._rho_is_zero(rho)
        if pass_enabled and (not rho_is_zero) and (v_img is None or fov is None):
            raise ValueError('v_img and fov are required when PaSS is enabled with rho > 0')
        if v_img is None:
            v_img = hidden_states.new_zeros((batch, seqlen, self.c_img))
        if fov is None:
            fov = torch.zeros((batch, seqlen), dtype=torch.bool, device=hidden_states.device)
        return v_img, fov

    def _forward_one_direction(self, xz, conv, x_proj, dt_proj, a_log, d_skip, v_img, fov, rho, pass_enabled):
        batch, _, seqlen = xz.shape
        x, z = xz.chunk(2, dim=1)
        conv_out = self._causal_conv(x, conv)
        x_dbl = x_proj(conv_out.transpose(1, 2).reshape(batch * seqlen, self.d_inner))
        dt_raw = x_dbl[:, :self.dt_rank]
        b_raw = x_dbl[:, self.dt_rank:self.dt_rank + self.d_state].view(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        c_raw = x_dbl[:, -self.d_state:].view(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        delta = F.linear(dt_raw, dt_proj.weight, dt_proj.bias)
        delta = F.softplus(delta.view(batch, seqlen, self.d_inner).transpose(1, 2).contiguous())
        delta_mod, b_mod = self.delta_b_mod(delta, b_raw, v_img, fov, rho=rho, pass_enabled=pass_enabled)
        return self._scan(conv_out, delta_mod, a_log, b_mod, c_raw, d_skip, z)

    def forward(self, hidden_states, v_img=None, fov=None, rho=1.0, pass_enabled=True, mappings=None):
        """Run bidirectional split Mamba.

        Args:
            hidden_states: (B, L, d_model)
            v_img: optional (B, L, c_img)
            fov: optional (B, L)
            rho: PaSS warmup scalar.
            pass_enabled: toggles delta/B image modulation.
            mappings: accepted for compatibility with mamba_ssm.Block.
        """
        del mappings
        batch, seqlen, dim = hidden_states.shape
        if dim != self.d_model:
            raise ValueError('hidden_states last dimension must equal d_model')
        v_img, fov = self._prepare_pass_inputs(hidden_states, v_img, fov, pass_enabled, rho)
        self.delta_b_mod.reset_debug_stats()

        xz = self.in_proj(hidden_states).transpose(1, 2).contiguous()
        out = self._forward_one_direction(
            xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D, v_img, fov, rho, pass_enabled
        )
        out_b = self._forward_one_direction(
            xz.flip(-1),
            self.conv1d_b,
            self.x_proj_b,
            self.dt_proj_b,
            self.A_b_log,
            self.D_b,
            v_img.flip(1),
            fov.flip(1),
            rho,
            pass_enabled,
        )
        out = (out + out_b.flip(-1)).transpose(1, 2).contiguous()
        out = self.out_proj(out)
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


def _checkpoint(function, *args):
    try:
        return cp.checkpoint(function, *args, use_reentrant=False)
    except TypeError:
        return cp.checkpoint(function, *args)


class PaSSMambaBlock(nn.Module):
    """mamba_ssm.Block-compatible wrapper around PaSSMamba."""

    supports_pass_fusion = True

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path=0.0,
        post_norm=True,
        with_cp=False,
        c_img=64,
        rank=4,
        b_residual_clip=1.0,
        use_ref_scan=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.mamba = PaSSMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            c_img=c_img,
            rank=rank,
            b_residual_clip=b_residual_clip,
            use_ref_scan=use_ref_scan,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.post_norm = bool(post_norm)
        self.with_cp = bool(with_cp)

    def _run_mamba(self, x, v_img, fov, rho, pass_enabled):
        if self.with_cp and self.training:
            if v_img is None or fov is None:
                return _checkpoint(
                    lambda hidden: self.mamba(
                        hidden,
                        v_img=None,
                        fov=None,
                        rho=rho,
                        pass_enabled=pass_enabled,
                    ),
                    x,
                )
            return _checkpoint(
                lambda hidden, img, mask: self.mamba(
                    hidden,
                    v_img=img,
                    fov=mask,
                    rho=rho,
                    pass_enabled=pass_enabled,
                ),
                x,
                v_img,
                fov,
            )
        return self.mamba(x, v_img=v_img, fov=fov, rho=rho, pass_enabled=pass_enabled)

    def forward(self, x, v_img=None, fov=None, rho=0.0, pass_enabled=False, mappings=None):
        del mappings
        if self.post_norm:
            x_out = self._run_mamba(x, v_img=v_img, fov=fov, rho=rho, pass_enabled=pass_enabled)
            x = x + self.drop_path(self.norm(x_out))
        else:
            x_norm = self.norm(x)
            x_out = self._run_mamba(x_norm, v_img=v_img, fov=fov, rho=rho, pass_enabled=pass_enabled)
            x = x + self.drop_path(x_out)
        return x
