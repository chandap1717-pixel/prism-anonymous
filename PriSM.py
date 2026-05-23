import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pscan import pscan


@dataclass
class PriSMConfig:
    num_nodes: int
    seq_len: int
    pred_len: int
    in_dim: int
    d_model: int
    n_layers: int

    ada_emb_d: int
    steps_per_day: int
    tod_emb_d: int
    days_per_week: int
    dow_emb_d: int
    cau_emb_d: int = 8
    deg_emb_d: int = 8

    dt_rank: Union[int, str] = "auto"
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 3
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    dropout: float = 0.1

    adj: Optional[torch.Tensor] = None
    B_tau: Optional[torch.Tensor] = None
    A_c: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class PriSMEmbedding(nn.Module):
    """Input fusion embedding with physical and lag-aware prior features."""

    def __init__(self, config: PriSMConfig):
        super().__init__()
        self.config = config
        self.in_embedding = nn.Linear(1, config.in_dim)

        self.tod_embedding = (
            nn.Embedding(config.steps_per_day, config.tod_emb_d)
            if config.tod_emb_d > 0
            else None
        )
        self.dow_embedding = (
            nn.Embedding(config.days_per_week, config.dow_emb_d)
            if config.dow_emb_d > 0
            else None
        )

        self.causal_proj = None
        if config.B_tau is not None and config.cau_emb_d > 0:
            B_tau = config.B_tau.float()
            if B_tau.dim() != 3:
                raise ValueError("B_tau must have shape [T_max, N, N].")
            with torch.no_grad():
                B_agg = B_tau.max(dim=0).values
                incoming = B_agg.sum(dim=0)
                outgoing = B_agg.sum(dim=1)
                causal_feat = torch.stack([incoming, outgoing], dim=-1)
                causal_feat = (causal_feat - causal_feat.mean(dim=0, keepdim=True)) / (
                    causal_feat.std(dim=0, keepdim=True) + 1e-6
                )
            self.register_buffer("causal_feat", causal_feat)
            self.causal_proj = nn.Linear(2, config.cau_emb_d)

        self.degree_proj = None
        if config.adj is not None and config.deg_emb_d > 0:
            adj = config.adj.float()
            if adj.dim() != 2:
                raise ValueError("adj must have shape [N, N].")
            with torch.no_grad():
                degree = adj.sum(dim=-1)
                degree = (degree - degree.mean()) / (degree.std() + 1e-6)
                degree = degree.unsqueeze(-1)
            self.register_buffer("degree_feat", degree)
            self.degree_proj = nn.Linear(1, config.deg_emb_d)

        self.adaptive_embedding = None
        if config.ada_emb_d > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(1, config.seq_len, config.num_nodes, config.d_model)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        input_dim = config.in_dim
        if self.tod_embedding is not None:
            input_dim += config.tod_emb_d
        if self.dow_embedding is not None:
            input_dim += config.dow_emb_d
        if self.causal_proj is not None:
            input_dim += config.cau_emb_d
        if self.degree_proj is not None:
            input_dim += config.deg_emb_d

        self.proj_out = nn.Linear(input_dim, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, C = x.shape

        emb_list = [self.in_embedding(x[..., :1])]

        if self.tod_embedding is not None:
            if C >= 2:
                tod_idx = (x[..., 1] * self.config.steps_per_day).long()
                tod_idx = tod_idx.clamp(0, self.config.steps_per_day - 1)
                tod_emb = self.tod_embedding(tod_idx)
            else:
                tod_emb = torch.zeros(B, L, N, self.config.tod_emb_d, device=x.device, dtype=x.dtype)
            emb_list.append(tod_emb)

        if self.dow_embedding is not None:
            if C >= 3:
                dow_idx = (x[..., 2] * self.config.days_per_week).long()
                dow_idx = dow_idx.clamp(0, self.config.days_per_week - 1)
                dow_emb = self.dow_embedding(dow_idx)
            else:
                dow_emb = torch.zeros(B, L, N, self.config.dow_emb_d, device=x.device, dtype=x.dtype)
            emb_list.append(dow_emb)

        if self.causal_proj is not None:
            causal_emb = self.causal_proj(self.causal_feat.to(x.device))
            causal_emb = causal_emb.view(1, 1, N, -1).expand(B, L, N, -1)
            emb_list.append(causal_emb)

        if self.degree_proj is not None:
            degree_emb = self.degree_proj(self.degree_feat.to(x.device))
            degree_emb = degree_emb.view(1, 1, N, -1).expand(B, L, N, -1)
            emb_list.append(degree_emb)

        h = self.proj_out(torch.cat(emb_list, dim=-1))

        if self.adaptive_embedding is not None:
            h = h + self.adaptive_embedding.expand(B, -1, -1, -1)

        return h


class SSMLayer(nn.Module):
    """Selective state-space layer used by temporal and spatial branches."""

    def __init__(self, config: PriSMConfig):
        super().__init__()
        self.config = config
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(config.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)


class AdaBiMambaBlock(nn.Module):
    """Temporal-spatial SSM block with lag-aware context modulation."""

    def __init__(self, config: PriSMConfig):
        super().__init__()
        self.config = config
        self.d_inner = config.d_inner
        self.d_model = config.d_model

        self.context_encoder = nn.Sequential(nn.SiLU(), nn.Linear(config.d_model, 6 * config.d_model))

        self.t_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.t_in_proj = nn.Linear(config.d_model, 2 * config.d_inner)
        self.t_conv = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv // 2,
            groups=config.d_inner,
        )
        self.t_ssm = SSMLayer(config)
        self.t_out_proj = nn.Linear(config.d_inner, config.d_model)

        self.s_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.s_in_proj = nn.Linear(config.d_model, 2 * config.d_inner)
        self.s_ssm_fwd = SSMLayer(config)
        self.s_ssm_bwd = SSMLayer(config)
        self.s_out_proj = nn.Linear(config.d_inner, config.d_model)

        self.register_buffer("perm_idx", torch.arange(config.num_nodes).long())
        self.register_buffer("rev_perm_idx", torch.arange(config.num_nodes).long())
        self.perm_initialized = False

        self.out_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(config.dropout)

    def _run_ssm(self, x: torch.Tensor, ssm_layer: SSMLayer) -> torch.Tensor:
        deltaBC = ssm_layer.x_proj(x)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )

        delta = F.softplus(ssm_layer.dt_proj(delta))
        A = -torch.exp(ssm_layer.A_log.float())
        D = ssm_layer.D.float()

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        hs = pscan(deltaA, deltaB * x.unsqueeze(-1))
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def _update_permutation(self, device: torch.device) -> None:
        if self.perm_initialized:
            return

        physical_adj = self.config.adj
        if physical_adj is not None:
            degree = physical_adj.to(device).sum(dim=1)
            indices = torch.argsort(degree, descending=True)
            self.perm_idx = indices.long().to(device)
            self.rev_perm_idx = torch.argsort(indices).long().to(device)

        self.perm_initialized = True

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        shift = shift.unsqueeze(1).unsqueeze(1)
        scale = scale.unsqueeze(1).unsqueeze(1)
        return x * (1.0 + scale) + shift

    def _extract_context(self, x: torch.Tensor) -> torch.Tensor:
        A_c = self.config.A_c
        if A_c is None:
            return x.mean(dim=[1, 2])

        A_c = A_c.to(x.device)
        x_context = torch.einsum("mn,blnd->blmd", A_c, x)
        return x_context.mean(dim=[1, 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, D = x.shape
        self._update_permutation(x.device)

        ctx_vec = self._extract_context(x)
        shift_t, scale_t, shift_s, scale_s, shift_o, scale_o = self.context_encoder(ctx_vec).chunk(6, dim=1)

        x_t_in = self.modulate(self.t_norm(x), shift_t, scale_t)
        x_t_flat = x_t_in.permute(0, 2, 1, 3).reshape(B * N, L, D)
        xz_t = self.t_in_proj(x_t_flat)
        x_t_branch, z_t = xz_t.chunk(2, dim=-1)
        x_t_branch = self.t_conv(x_t_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_t_branch = F.silu(x_t_branch)
        y_t = self._run_ssm(x_t_branch, self.t_ssm) * F.silu(z_t)
        out_t = self.t_out_proj(y_t).view(B, N, L, D).permute(0, 2, 1, 3)

        x_s_in = self.modulate(self.s_norm(x), shift_s, scale_s)
        x_s_flat = x_s_in.reshape(B * L, N, D)
        x_s_perm = x_s_flat[:, self.perm_idx, :]
        xz_s = self.s_in_proj(x_s_perm)
        x_s_branch, z_s = xz_s.chunk(2, dim=-1)
        x_s_branch = F.silu(x_s_branch)

        y_s_fwd = self._run_ssm(x_s_branch, self.s_ssm_fwd)
        y_s_bwd = self._run_ssm(torch.flip(x_s_branch, dims=[1]), self.s_ssm_bwd)
        y_s = 0.5 * (y_s_fwd + torch.flip(y_s_bwd, dims=[1])) * F.silu(z_s)
        y_s = y_s[:, self.rev_perm_idx, :]
        out_s = self.s_out_proj(y_s).view(B, L, N, D)

        out = out_t + out_s
        out = self.modulate(self.out_norm(out), shift_o, scale_o)
        return self.dropout(out)


class PriSM(nn.Module):
    """Prior-Separated Selective State-Space Model."""

    def __init__(self, config: PriSMConfig):
        super().__init__()
        self.config = config
        self.embedding = PriSMEmbedding(config)
        self.layers = nn.ModuleList([AdaBiMambaBlock(config) for _ in range(config.n_layers)])
        self.out_proj = nn.Sequential(
            nn.Linear(config.d_model * config.seq_len, config.d_model * config.seq_len),
            nn.GELU(),
            nn.Linear(config.d_model * config.seq_len, config.pred_len),
        )

    def forward(self, x: torch.Tensor, return_loss_components: bool = False):
        x_curr = self.embedding(x)

        for layer in self.layers:
            x_curr = x_curr + layer(x_curr)

        x_flat = rearrange(x_curr, "b l n d -> b n (l d)")
        out = self.out_proj(x_flat).transpose(1, 2).unsqueeze(-1)

        if return_loss_components:
            return out, None, None, None, None
        return out
