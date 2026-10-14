import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm


class CenterNorm(nn.Module):
    r"""CenterNorm normalizing across the feature dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape))
        self.scale = (
            normalized_shape / (normalized_shape - 1.0)
            if normalized_shape > 1
            else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        x = self.scale * (x - u)
        x = self.weight * x + self.bias
        return x


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=bias,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attentions = self.mha(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        attn_out = self.proj_drop(attn_out)
        return attn_out, attentions


class SimpleGatedSDPAttention(nn.Module):
    """
    SDP Attention with headwise/elementwise gating and optional q/k RMSNorm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        gate: str = "headwise",
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=qkv_bias)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.gate_type = gate
        self.gate_activation = gate_activation

        if self.gate_type != "none":
            out_dim = self.num_heads if self.gate_type == "headwise" else (self.num_heads * self.head_dim)
            self.gate_proj = nn.Linear(hidden_size, out_dim, bias=False)

    def _compute_gate(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "none":
            return attn_out

        gate_logits = self.gate_proj(x)
        if self.gate_activation == "sigmoid":
            gate_scores = torch.sigmoid(gate_logits)
        elif self.gate_activation == "silu":
            gate_scores = F.silu(gate_logits)
        else:
            gate_scores = gate_logits

        b, seq = gate_scores.shape[0], gate_scores.shape[1]
        if self.gate_type == "headwise":
            gate = gate_scores.unsqueeze(-1)
        else:
            gate = gate_scores.view(b, seq, self.num_heads, self.head_dim)

        return attn_out * gate

    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(attn_scores.dtype)
        attn_out = torch.matmul(attn_probs, v)

        attn_out = attn_out.transpose(1, 2)
        gated = self._compute_gate(hidden_states, attn_out)

        gated = gated.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        output = self.out_proj(gated)

        attn_weights = attn_probs if output_attentions else None
        return output, attn_weights


def g_k(p: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Compute g_k(p) = x_(k) * (1 - x_(k) + x_(k+1))
    where x_(k) and x_(k+1) are k-th and (k+1)-th largest elements per row.
    """
    sorted_p, _ = torch.sort(p, dim=-1, descending=True)
    x_k = sorted_p[..., k - 1]
    x_k1 = sorted_p[..., k] if k < p.size(-1) else torch.zeros_like(x_k)
    return x_k * (1 - x_k + x_k1)


def jasmin_loss(
    attn_maps: Union[torch.Tensor, List[torch.Tensor]],
    k: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute JaSMin regularization loss on attention maps.
    """
    if isinstance(attn_maps, torch.Tensor):
        attn_maps = [attn_maps]

    losses = []
    for P in attn_maps:
        if torch.isnan(P).any() or torch.isinf(P).any():
            continue
        P = torch.clamp(P, min=1e-6, max=1.0)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)
        g1 = g_k(P, k=1)
        if k == 0:
            loss = torch.log(g1 + 1e-6)
        else:
            gk = g_k(P, k=k)
            loss = torch.log(g1 + 1e-6) - torch.log(gk + 1e-6)

        loss = loss.max(dim=-1).values.mean(dim=1)
        losses.append(loss.mean())

    if len(losses) == 0:
        dev = attn_maps[0].device if isinstance(attn_maps, list) else torch.device("cpu")
        return torch.tensor(0.0, device=dev, requires_grad=True)

    losses = torch.stack(losses)
    return losses.mean() if reduction == "mean" else losses.sum()


class Lie_Trotter(nn.Module):
    """
    Parallel sublayers derivative calculation: F(x) + G(x).
    Supports standard MultiheadSelfAttention and SimpleGatedSDPAttention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        gated_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.norm_attn = CenterNorm(dim)
        self.norm_mlp = CenterNorm(dim)

        if gated_attention:
            self.attn = SimpleGatedSDPAttention(
                hidden_size=dim,
                num_heads=num_heads,
                qkv_bias=False,
                gate="headwise",
            )
        else:
            self.attn = MultiheadSelfAttention(
                dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop
            )

        self.mlp = MLPBlock(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=mlp_drop)

        self.alpha = nn.Parameter(torch.tensor(0.5)[None], requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5)[None], requires_grad=True)

    def forward(self, x: torch.Tensor, t: Optional[float] = None) -> torch.Tensor:
        g, self.attentions = self.attn(self.norm_attn(x))
        f = self.mlp(self.norm_mlp(x))
        return self.alpha * g + self.beta * f


class ODEViTVelocityNet(nn.Module):
    """
    Computes latent velocity field dz/dt = v_phi(z_t, a_t, t).
    Supports standard 2-token and Control-Affine (1 + d_a tokens) architecture.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        depth: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gated_attention: bool = False,
        control_affine: bool = False,
        use_delta_z: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.control_affine = control_affine
        self.use_delta_z = use_delta_z

        if self.control_affine:
            # Control query tokens: one learned embedding per action dimension
            self.ctrl_tokens = nn.Parameter(torch.randn(action_dim, dim) * 0.02)
            # Learned type embeddings for state and control query tokens
            self.type_emb_z = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.type_emb_c = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        else:
            self.act_proj = (
                nn.Linear(action_dim, dim) if action_dim != dim else nn.Identity()
            )
            if self.use_delta_z:
                self.type_emb_z = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
                self.type_emb_a = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
                self.type_emb_dz = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            Lie_Trotter(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                proj_drop=dropout,
                mlp_drop=dropout,
                gated_attention=gated_attention,
            )
            for _ in range(depth)
        ])
        self.out_norm = CenterNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, t: Optional[float] = None, dz: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.control_affine:
            B = z.shape[0]
            z_tok = z.unsqueeze(1) + self.type_emb_z  # (B, 1, D)
            ctrl_tokens = self.ctrl_tokens.unsqueeze(0).expand(B, -1, -1) + self.type_emb_c  # (B, d_a, D)
            tokens = torch.cat([z_tok, ctrl_tokens], dim=1)  # (B, 1 + d_a, D)

            for block in self.blocks:
                tokens = tokens + block(tokens, t)

            h = self.out_norm(tokens)
            drift = self.out_proj(h[:, 0])  # f(z): (B, D)
            gcols = self.out_proj(h[:, 1:])  # g(z): (B, d_a, D)

            v = drift + torch.einsum('ba,bad->bd', a, gcols)  # zdot = f(z) + g(z)*u
            return v
        else:
            if self.use_delta_z:
                z_tok = z.unsqueeze(1) + self.type_emb_z  # (B, 1, D)
                a_tok = self.act_proj(a).unsqueeze(1) + self.type_emb_a  # (B, 1, D)
                if dz is None:
                    dz = torch.zeros_like(z)
                dz_tok = dz.unsqueeze(1) + self.type_emb_dz  # (B, 1, D)
                tokens = torch.cat([z_tok, a_tok, dz_tok], dim=1)  # (B, 3, D)
            else:
                z_tok = z.unsqueeze(1)  # (B, 1, D)
                a_tok = self.act_proj(a).unsqueeze(1)  # (B, 1, D)
                tokens = torch.cat([z_tok, a_tok], dim=1)  # (B, 2, D)

            for block in self.blocks:
                tokens = tokens + block(tokens, t)

            v_z = self.out_proj(self.out_norm(tokens[:, 0, :]))  # (B, D)
            return v_z


class ODEViTPredictor(nn.Module):
    """
    Continuous-Time Deterministic ODE-ViT Predictor with fixed-step Euler integrator, RK4, and JaSMin support.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_euler_steps_per_frame: int = 4,
        depth: int = 2,
        heads: int = 8,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
        gated_attention: bool = False,
        regularize_jasmin: bool = False,
        control_affine: bool = False,
        use_delta_z: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim or input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_euler_steps_per_frame = kwargs.get("num_euler_steps", num_euler_steps_per_frame)
        self.regularize_jasmin = regularize_jasmin
        self.control_affine = control_affine
        self.use_delta_z = use_delta_z

        mlp_ratio = (mlp_dim / self.hidden_dim) if mlp_dim else 4.0

        self.velocity_net = ODEViTVelocityNet(
            dim=self.hidden_dim,
            action_dim=self.action_dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gated_attention=gated_attention,
            control_affine=control_affine,
            use_delta_z=use_delta_z,
        )
        self.attention_trajectory = []
        
        # Spectral Initialization (Lipschitz constraint)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    u, s, vh = torch.linalg.svd(m.weight, full_matrices=False)
                    m.weight.data = m.weight.data / s[0]

    def velocity_field(
        self, z_t: torch.Tensor, a_t: torch.Tensor, t: Optional[float] = None, dz_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes dz/dt = v_phi(z_t, a_t, t, dz_t)."""
        v = self.velocity_net(z_t, a_t, t, dz=dz_t)
        if self.regularize_jasmin:
            for block in self.velocity_net.blocks:
                if hasattr(block, "attentions") and block.attentions is not None:
                    self.attention_trajectory.append(block.attentions)
        return v

    def get_jasmin_loss(self, k: int = 0) -> torch.Tensor:
        """Compute JaSMin regularization loss over recorded attention trajectory."""
        if len(self.attention_trajectory) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        loss = jasmin_loss(self.attention_trajectory, k=k)
        self.attention_trajectory = []
        return loss

    def step_euler(
        self, z_t: torch.Tensor, a_t: torch.Tensor, num_steps: Optional[int] = None, dz_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Advances physical time from t to t+1 using N-step fixed-step Euler integration per frame.
        """
        if num_steps is None:
            num_steps = self.num_euler_steps_per_frame

        dt = 1.0 / num_steps
        z = z_t
        for i in range(num_steps):
            t_curr = i * dt
            v = self.velocity_field(z, a_t, t=t_curr, dz_t=dz_t)
            z = z + dt * v
        return z

    def step_midpoint(
        self, z_t: torch.Tensor, a_t: torch.Tensor, num_steps: Optional[int] = None, dz_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Advances physical time from t to t+1 using N-step explicit Midpoint (RK2) integration per frame.
        """
        if num_steps is None:
            num_steps = getattr(self, "num_midpoint_steps_per_frame", 1)

        dt = 1.0 / num_steps
        z = z_t
        for i in range(num_steps):
            t_curr = i * dt
            k1 = self.velocity_field(z, a_t, t=t_curr, dz_t=dz_t)
            k2 = self.velocity_field(z + 0.5 * dt * k1, a_t, t=t_curr + 0.5 * dt, dz_t=dz_t)
            z = z + dt * k2
        return z

    def step_rk4(
        self, z_t: torch.Tensor, a_t: torch.Tensor, num_steps: Optional[int] = None, dz_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Advances physical time from t to t+1 using N-step classic 4th-order Runge-Kutta (RK4) integration per frame.
        """
        if num_steps is None:
            num_steps = getattr(self, "num_rk4_steps_per_frame", 1)

        dt = 1.0 / num_steps
        z = z_t
        for i in range(num_steps):
            t_curr = i * dt
            k1 = self.velocity_field(z, a_t, t=t_curr, dz_t=dz_t)
            k2 = self.velocity_field(z + 0.5 * dt * k1, a_t, t=t_curr + 0.5 * dt, dz_t=dz_t)
            k3 = self.velocity_field(z + 0.5 * dt * k2, a_t, t=t_curr + 0.5 * dt, dz_t=dz_t)
            k4 = self.velocity_field(z + dt * k3, a_t, t=t_curr + dt, dz_t=dz_t)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z

    def predict_subframe(
        self, z_t: torch.Tensor, a_t: torch.Tensor, dt: float, dz_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Subframe Query Utility:
        Queries continuous intermediate state z_{t + dt} for sub-interval dt in (0, 1).
        """
        was_3d = z_t.ndim == 3
        if was_3d:
            z_in = z_t.squeeze(1)
            a_in = a_t.squeeze(1) if a_t.ndim == 3 else a_t
            dz_in = dz_t.squeeze(1) if (dz_t is not None and dz_t.ndim == 3) else dz_t
        else:
            z_in = z_t
            a_in = a_t
            dz_in = dz_t

        num_steps = max(1, round(dt * self.num_euler_steps_per_frame))
        step_dt = dt / num_steps
        
        z_curr = z_in
        for i in range(num_steps):
            t_curr = i * step_dt
            v = self.velocity_field(z_curr, a_in, t=t_curr, dz_t=dz_in)
            z_curr = z_curr + step_dt * v

        if was_3d:
            z_curr = z_curr.unsqueeze(1)
        return z_curr

    def predict_trajectory(
        self, z_0: torch.Tensor, action_seq: torch.Tensor, num_steps: Optional[int] = None, solver: Optional[str] = None, dz_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sequentially predicts an H-step future trajectory given initial state z_0 and action sequence.
        Tracks incoming velocity/momentum dz_t = z_curr - z_prev at each step.
        """
        B, H, _ = action_seq.shape
        z_curr = z_0
        z_prev = (z_0 - dz_0) if dz_0 is not None else z_0
        preds = []
        solver_mode = solver or getattr(self, "solver", "euler")
        if solver_mode == "rk4":
            step_fn = self.step_rk4
        elif solver_mode in ("midpoint", "rk2"):
            step_fn = self.step_midpoint
        else:
            step_fn = self.step_euler

        for h in range(H):
            a_curr = action_seq[:, h]
            dz_curr = z_curr - z_prev
            z_next = step_fn(z_curr, a_curr, num_steps=num_steps, dz_t=dz_curr)
            preds.append(z_next)
            z_prev = z_curr
            z_curr = z_next
        return torch.stack(preds, dim=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_steps: Optional[int] = None, solver: Optional[str] = None) -> torch.Tensor:
        """
        JEPA Predictor forward interface:
        x: (B, T, D) sequence of latent state embeddings
        c: (B, T, D_act) sequence of action embeddings
        Returns: (B, T, D) predicted next state embeddings
        """
        self.attention_trajectory = []
        B, T, D = x.shape
        z_flat = rearrange(x, "b t d -> (b t) d")
        a_flat = rearrange(c, "b t d -> (b t) d")

        solver_mode = solver or getattr(self, "solver", "euler")
        if solver_mode == "rk4":
            z_next_flat = self.step_rk4(z_flat, a_flat, num_steps=num_steps)
        elif solver_mode in ("midpoint", "rk2"):
            z_next_flat = self.step_midpoint(z_flat, a_flat, num_steps=num_steps)
        else:
            z_next_flat = self.step_euler(z_flat, a_flat, num_steps=num_steps)
        pred = rearrange(z_next_flat, "(b t) d -> b t d", b=B, t=T)
        return pred
