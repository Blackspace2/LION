import torch
import torch.nn as nn


class DeltaBMod(nn.Module):
    """Image-conditioned modulation for Mamba delta and B parameters."""

    def __init__(self, c_img=64, d_inner=128, d_state=16, rank=4, delta_scale=0.5, b_residual_clip=1.0):
        super().__init__()
        self.c_img = int(c_img)
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.rank = int(rank)
        self.b_residual_clip = None if b_residual_clip is None else float(b_residual_clip)
        self.debug_stats = []

        self.img_norm = nn.LayerNorm(self.c_img)
        self.w_delta = nn.Linear(self.c_img, self.d_inner)
        self.w_lr = nn.Linear(self.c_img, self.rank)
        self.u = nn.Parameter(torch.empty(self.d_inner, self.rank))
        self.v = nn.Parameter(torch.empty(self.d_state, self.rank))
        self.scale = nn.Parameter(torch.tensor(float(delta_scale)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.w_delta.weight)
        nn.init.zeros_(self.w_delta.bias)
        nn.init.normal_(self.w_lr.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w_lr.bias)
        nn.init.normal_(self.u, mean=0.0, std=0.01)
        nn.init.zeros_(self.v)

    @staticmethod
    def _rho_is_zero(rho):
        if isinstance(rho, torch.Tensor):
            return bool(rho.detach().float().abs().max().item() == 0.0)
        return float(rho) == 0.0

    def reset_debug_stats(self):
        self.debug_stats = []

    def _soft_clamp_delta_b(self, delta_b):
        if self.b_residual_clip is None or self.b_residual_clip <= 0:
            return delta_b
        clip = torch.as_tensor(self.b_residual_clip, dtype=delta_b.dtype, device=delta_b.device)
        norm = torch.linalg.vector_norm(delta_b.float(), dim=(-2, -1), keepdim=True).to(dtype=delta_b.dtype)
        ratio = norm / clip
        safe_ratio = torch.where(norm > 0, ratio, torch.ones_like(ratio))
        scale = torch.tanh(safe_ratio) / safe_ratio
        scale = torch.where(norm > 0, scale, torch.ones_like(scale))
        return delta_b * scale

    def _record_debug_stats(self, delta_b_raw, delta_b, b_raw, rho_t):
        with torch.no_grad():
            raw_f = delta_b_raw.detach().float()
            clipped_f = delta_b.detach().float()
            b_f = b_raw.detach().float()
            b_norm = (b_f.pow(2).sum() * self.d_inner).sqrt().clamp_min(1e-12)
            raw_norm = raw_f.norm()
            clipped_norm = clipped_f.norm()
            rho_abs = rho_t.detach().float().abs().max()

            raw_token_norm = torch.linalg.vector_norm(raw_f, dim=(-2, -1))
            clipped_token_norm = torch.linalg.vector_norm(clipped_f, dim=(-2, -1))
            b_token_norm = torch.linalg.vector_norm(b_f.transpose(1, 2), dim=-1) * (float(self.d_inner) ** 0.5)
            b_token_norm = b_token_norm.clamp_min(1e-12)

            self.debug_stats.append({
                'delta_b_raw_norm': float(raw_norm.item()),
                'delta_b_norm': float(clipped_norm.item()),
                'rho_delta_b_norm': float((rho_abs * clipped_norm).item()),
                'b_norm': float(b_norm.item()),
                'delta_b_to_b_ratio': float((clipped_norm / b_norm).item()),
                'rho_delta_b_to_b_ratio': float((rho_abs * clipped_norm / b_norm).item()),
                'delta_b_raw_to_b_ratio': float((raw_norm / b_norm).item()),
                'delta_b_token_norm_max': float(clipped_token_norm.max().item()) if clipped_token_norm.numel() else 0.0,
                'delta_b_raw_token_norm_max': float(raw_token_norm.max().item()) if raw_token_norm.numel() else 0.0,
                'delta_b_token_to_b_max': float((clipped_token_norm / b_token_norm).max().item()) if clipped_token_norm.numel() else 0.0,
                'b_residual_clip': 0.0 if self.b_residual_clip is None else float(self.b_residual_clip),
            })

    def forward(self, delta, b_raw, v_img=None, fov=None, rho=1.0, pass_enabled=True):
        """Apply PaSS modulation.

        Args:
            delta: (B, d_inner, L), already softplus-transformed.
            b_raw: (B, d_state, L), shared-over-channel B from x_proj.
            v_img: (B, L, c_img), image vector aligned to token order.
            fov: (B, L), bool/float mask. FOV-out tokens are identity.
            rho: scalar warmup in [0, 1].
            pass_enabled: if false, returns the shared-B identity branch.
        """
        if delta.dim() != 3:
            raise ValueError('delta must have shape (B, d_inner, L)')
        if b_raw.dim() != 3:
            raise ValueError('b_raw must have shape (B, d_state, L)')
        if delta.shape[1] != self.d_inner:
            raise ValueError('delta d_inner mismatch')
        if b_raw.shape[1] != self.d_state:
            raise ValueError('b_raw d_state mismatch')

        if (not pass_enabled) or self._rho_is_zero(rho):
            self.reset_debug_stats()
            return delta, b_raw
        if v_img is None or fov is None:
            raise ValueError('v_img and fov are required when PaSS is enabled with rho > 0')
        if v_img.dim() != 3:
            raise ValueError('v_img must have shape (B, L, c_img)')
        if fov.dim() != 2:
            raise ValueError('fov must have shape (B, L)')
        if v_img.shape[0] != delta.shape[0] or v_img.shape[1] != delta.shape[2]:
            raise ValueError('v_img must align with delta batch and sequence dimensions')
        if fov.shape != v_img.shape[:2]:
            raise ValueError('fov must align with v_img batch and sequence dimensions')

        rho_t = torch.as_tensor(rho, dtype=delta.dtype, device=delta.device)
        fov_t = fov.to(dtype=delta.dtype, device=delta.device)
        v_norm = self.img_norm(v_img.to(dtype=delta.dtype))

        g_delta = self.w_delta(v_norm)
        delta_mul = torch.exp(rho_t * self.scale.to(delta.dtype) * torch.tanh(g_delta) * fov_t.unsqueeze(-1))
        delta_mod = delta * delta_mul.permute(0, 2, 1).contiguous()

        alpha = self.w_lr(v_norm)
        delta_b = torch.einsum('dr,blr,nr->bldn', self.u.to(delta.dtype), alpha, self.v.to(delta.dtype))
        delta_b = delta_b * fov_t[:, :, None, None]
        delta_b_raw = delta_b
        delta_b = self._soft_clamp_delta_b(delta_b)
        self._record_debug_stats(delta_b_raw, delta_b, b_raw, rho_t)
        b_mod = b_raw.unsqueeze(1).expand(-1, self.d_inner, -1, -1) + rho_t * delta_b.permute(0, 2, 3, 1)
        return delta_mod, b_mod.contiguous()
