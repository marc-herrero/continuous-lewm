import time
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging
from stable_worldmodel.solver.utils import prepare_init_action

class CollocationSolver(torch.nn.Module):
    """Collocation (Boundary Value Problem) Solver for ODE World Models.
    
    Optimizes both knot state variables z_1...z_K and action sequence a_0...a_{K-1}
    simultaneously with trapezoidal integration defect penalty.
    """

    def __init__(
        self,
        model: Any,
        n_steps: int = 50,
        batch_size: int | None = None,
        lr: float = 0.05,
        defect_weight: float = 10.0,
        action_reg: float = 1e-4,
        device: str | torch.device = 'cuda',
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lr = lr
        self.defect_weight = defect_weight
        self.action_reg = action_reg
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

        self._configured = False
        self._n_envs = None
        self._action_dim = None
        self._config = None

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        total_envs = len(next(iter(info_dict.values())))

        # Encode initial state z_0 and goal state z_goal
        # Extract batch of initial info and goal info
        with torch.no_grad():
            init_action = prepare_init_action(
                self.model, info_dict, init_action, self.horizon, n_envs=total_envs, action_dim=self.action_dim
            ).to(self.device, dtype=self.dtype)

            # Get z_0 and z_goal embeddings
            local_info = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in info_dict.items()}
            
            # Encode z_0
            init_obs = {k: v[:, 0] for k, v in local_info.items() if torch.is_tensor(v) and k not in ["action", "emb", "predicted_emb", "goal_emb"]}
            init_enc = self.model.encode(init_obs)
            z_0 = init_enc["emb"][:, 0] # (total_envs, D)
            
            # Encode z_goal
            goal_obs = {k: v[:, 0] for k, v in local_info.items() if torch.is_tensor(v) and k not in ["action", "emb", "predicted_emb", "goal_emb"]}
            goal_obs["pixels"] = goal_obs["goal"]
            for k in list(goal_obs.keys()):
                if k.startswith("goal_"):
                    goal_obs[k[len("goal_") :]] = goal_obs.pop(k)
            goal_obs.pop("action", None)
            goal_enc = self.model.encode(goal_obs)
            z_goal = goal_enc["emb"][:, 0] # (total_envs, D)

        K = self.horizon
        D = z_0.shape[-1]
        
        # Initialize knot states via linear interpolation between z_0 and z_goal
        # z_knots: (total_envs, K+1, D)
        alphas = torch.linspace(0, 1, K + 1, device=self.device, dtype=self.dtype).view(1, K + 1, 1)
        z_knots_init = (1 - alphas) * z_0.unsqueeze(1) + alphas * z_goal.unsqueeze(1)
        
        # Decision variables to optimize: z_knots[:, 1:] (since z_0 is fixed) and actions
        z_knots_opt = torch.nn.Parameter(z_knots_init[:, 1:].clone().detach())
        actions_opt = torch.nn.Parameter(init_action.clone().detach())

        optimizer = torch.optim.AdamW([z_knots_opt, actions_opt], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Construct full state trajectory (z_0, z_1 ... z_K)
            z_full = torch.cat([z_0.unsqueeze(1), z_knots_opt], dim=1) # (total_envs, K+1, D)

            # Compute velocities f(z_k, a_k) for all steps
            # Encode actions: actions_opt (total_envs, K, action_dim)
            act_emb = self.model.action_encoder(actions_opt) # (total_envs, K, A_emb)

            # Compute predicted next states using model.predict
            # defect penalty: z_{k+1} - model.predict(z_k, a_k)
            defects = []
            for k in range(K):
                z_k = z_full[:, k:k+1] # (total_envs, 1, D)
                z_k1 = z_full[:, k + 1] # (total_envs, D)
                a_k = act_emb[:, k:k+1] # (total_envs, 1, A_emb)

                # Step forward using model.predict
                z_k1_pred = self.model.predict(z_k, a_k)[:, -1] # (total_envs, D)

                # Defect: difference between predicted state from z_k and knot z_{k+1}
                defect = z_k1 - z_k1_pred
                defects.append(defect)

            defects_tensor = torch.stack(defects, dim=1) # (total_envs, K, D)
            defect_loss = defects_tensor.pow(2).sum(dim=-1).mean()

            # Goal distance loss: distance between final knot z_K and z_goal
            goal_loss = (z_knots_opt[:, -1] - z_goal).pow(2).sum(dim=-1).mean()

            # Action regularization loss
            reg_loss = actions_opt.pow(2).sum(dim=-1).mean()

            total_loss = goal_loss + self.defect_weight * defect_loss + self.action_reg * reg_loss
            total_loss.backward()

            optimizer.step()

        solve_time = time.time() - start_time
        print(f"CollocationSolver completed in {solve_time:.4f} seconds.")

        actions_ret = actions_opt.detach().cpu()

        return {
            'actions': actions_ret,
            'z_knots': torch.cat([z_0.unsqueeze(1), z_knots_opt.detach()], dim=1).cpu(),
            'solve_time': solve_time
        }
