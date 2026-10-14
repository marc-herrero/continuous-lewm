import torch
import torch.nn as nn
from typing import Optional
from cwm.models.ode_vit import ODEViTPredictor


class Embedder(nn.Module):
    """Action embedder projecting low-dim control actions into the latent dimension."""

    def __init__(
        self,
        input_dim: int = 2,
        smoothed_dim: int = 192,
        emb_dim: int = 192,
        mlp_scale: int = 4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class ODESegmentedModel(nn.Module):
    """
    Continuous-time ODE-ViT predictor (2.55M parameters).
    Learns a vector field dz/dt = f_theta(z, a) integrated across arbitrary horizons.
    """

    def __init__(
        self,
        input_dim: int = 192,
        action_dim: int = 2,
        hidden_dim: int = 192,
        depth: int = 5,
        heads: int = 3,
        mlp_dim: int = 768,
    ):
        super().__init__()
        self.action_encoder = Embedder(
            input_dim=action_dim, smoothed_dim=hidden_dim, emb_dim=hidden_dim
        )
        self.predictor = ODEViTPredictor(
            input_dim=input_dim,
            action_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.0,
            control_affine=False,
            use_delta_z=False,
        )

    def integrate(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        n_sub: int = 1,
        inject_noise_sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        Integrates latent state forward in time under given action sequence.
        z: (B, 192)
        actions: (B, Delta, 2)
        """
        act_emb = self.action_encoder(actions)  # (B, Delta, 192)
        delta = actions.shape[1]
        curr_z = z
        dt = 1.0 / n_sub
        for step in range(delta):
            if inject_noise_sigma > 0.0 and self.training:
                curr_z = curr_z + torch.randn_like(curr_z) * inject_noise_sigma

            a_t = act_emb[:, step]
            for s in range(n_sub):
                v = self.predictor.velocity_field(curr_z, a_t, t=s * dt)
                curr_z = curr_z + dt * v
        return curr_z

    def integrate_backward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        n_sub: int = 1,
    ) -> torch.Tensor:
        """
        Integrates latent state backward in time (-dt) to reconstruct past latents.
        z: (B, 192)
        actions: (B, Delta, 2)
        """
        act_emb = self.action_encoder(actions)
        delta = actions.shape[1]
        curr_z = z
        dt = 1.0 / n_sub
        # Reverse through time steps
        for step in reversed(range(delta)):
            a_t = act_emb[:, step]
            for s in range(n_sub):
                v = self.predictor.velocity_field(curr_z, a_t, t=(n_sub - s) * dt)
                curr_z = curr_z - dt * v
        return curr_z


class DiscreteMatchedModel(nn.Module):
    """
    Matched discrete transformer/MLP predictor (2.55M parameters).
    Evaluates fixed discrete transition map z_{t+1} = z_t + F(z_t, a_t) composed iteratively.
    """

    def __init__(
        self,
        input_dim: int = 192,
        action_dim: int = 2,
        hidden_dim: int = 192,
        depth: int = 5,
        heads: int = 3,
        mlp_dim: int = 768,
    ):
        super().__init__()
        self.action_encoder = Embedder(
            input_dim=action_dim, smoothed_dim=hidden_dim, emb_dim=hidden_dim
        )
        self.predictor = ODEViTPredictor(
            input_dim=input_dim,
            action_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.0,
            control_affine=False,
            use_delta_z=False,
        )

    def integrate(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        n_sub: int = 1,
        inject_noise_sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        Iterative composition of discrete map: z_{t+1} = z_t + F(z_t, a_t)
        """
        act_emb = self.action_encoder(actions)
        delta = actions.shape[1]
        curr_z = z
        for step in range(delta):
            a_t = act_emb[:, step]
            dz = self.predictor.velocity_field(curr_z, a_t, t=0.0)
            curr_z = curr_z + dz
        return curr_z
