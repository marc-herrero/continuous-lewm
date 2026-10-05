#!/usr/bin/env python3
"""
ensemble_kalman_solver.py
Ensemble Kalman Inversion (EKI) and Hybrid CEM-EKI Solvers for Latent World Model Planning.

Formulation:
- State to estimate: Action sequence a in R^{H * action_dim}
- Measurement: Goal latent embedding z_goal in R^{D}
- Forward model: World model rollout Phi(a) -> hat{z}_T in R^{D}
- Cross-covariance C^{az} = 1/(J-1) * sum (a_j - bar{a}) (hat{z}_j - bar{z})^T
- Output covariance C^{zz} = 1/(J-1) * sum (hat{z}_j - bar{z}) (hat{z}_j - bar{z})^T
- EKI Update: a_j <- a_j + C^{az} (C^{zz} + Gamma)^{-1} (z_goal - hat{z}_j)
- Self-annealing: Ensemble variance contracts naturally over iterations
- Covariance inflation: beta * (a_j - bar{a}) with beta ~ 1.05 prevents premature collapse
"""

import time
import math
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.solver.cem import prepare_init_action


class EnsembleKalmanInversionSolver:
    """
    Pure Ensemble Kalman Inversion (EKI) Solver for action planning.
    """
    def __init__(
        self,
        model: nn.Module,
        num_particles: int = 64,
        n_iterations: int = 10,
        gamma: float = 0.1,
        inflation: float = 1.05,
        stochastic: bool = False,
        device: torch.device = torch.device("cuda:0"),
        seed: int = 42,
        batch_size: int = 25,
        dtype: torch.dtype = torch.float32,
        action_clip: float = 3.0,
        **kwargs,
    ):
        self.model = model
        self.num_particles = num_particles
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.inflation = inflation
        self.stochastic = stochastic
        self.device = device
        self.base_seed = seed
        self.batch_size = batch_size
        self.dtype = dtype
        self.action_clip = action_clip

        self.horizon: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.action_space = None
        self.env_generators: Dict[int, torch.Generator] = {}
        self.callbacks = []

        # Diagnostics: track ensemble spread over iterations (empirical annealing curve)
        self.spread_history: List[float] = []
        self.iteration_spreads: List[List[float]] = [[] for _ in range(self.n_iterations)]

    def configure(self, action_space, n_envs: int, config, **kwargs):
        self.action_space = action_space
        if len(action_space.shape) > 1:
            self._action_dim = int(np.prod(action_space.shape[1:]))
        else:
            self._action_dim = int(np.prod(action_space.shape))
        action_block = getattr(config, "action_block", 1)
        self.action_dim = self._action_dim * action_block
        self.horizon = config.horizon

    def get_env_generator(self, env_id: int):
        if env_id not in self.env_generators:
            gen = torch.Generator(device=self.device).manual_seed(self.base_seed + int(env_id) * 10007)
            self.env_generators[env_id] = gen
        return self.env_generators[env_id]

    def __call__(self, info_dict: dict, init_action: Optional[torch.Tensor] = None) -> dict:
        return self.solve(info_dict, init_action=init_action)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: Optional[torch.Tensor] = None) -> dict:
        start_time = time.time()
        total_envs = len(next(iter(info_dict.values())))

        active_env_ids = info_dict.get("_env_idx")
        if active_env_ids is None:
            active_env_ids = list(range(total_envs))
        elif torch.is_tensor(active_env_ids):
            active_env_ids = active_env_ids.cpu().tolist()

        init_action = prepare_init_action(
            self.model, info_dict, init_action, self.horizon,
            n_envs=total_envs, action_dim=self.action_dim,
        )

        J = self.num_particles
        H = self.horizon
        d_a = self.action_dim
        d_flat = H * d_a

        final_actions = torch.zeros(total_envs, H, d_a, device=self.device, dtype=self.dtype)
        final_costs = []

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_env_ids = active_env_ids[start_idx:end_idx]

            # Initialize Gaussian prior for the ensemble
            # (current_bs, H, d_a)
            if init_action is not None and torch.is_tensor(init_action):
                batch_init = init_action[start_idx:end_idx].to(device=self.device, dtype=self.dtype)
            else:
                batch_init = torch.zeros(current_bs, H, d_a, device=self.device, dtype=self.dtype)

            # Sample initial ensemble A_0: (current_bs, J, H, d_a)
            cands_list = []
            for b_i in range(current_bs):
                gen = self.get_env_generator(batch_env_ids[b_i])
                noise = torch.randn(
                    1, J, H, d_a,
                    generator=gen, device=self.device, dtype=self.dtype,
                )
                cands_list.append(noise)
            ensemble = torch.cat(cands_list, dim=0) + batch_init.unsqueeze(1)
            # Particle 0 tracks the prior mean
            ensemble[:, 0] = batch_init

            # Prepare expanded environment info for batch rollouts
            expanded_infos = {}
            for k, v in info_dict.items():
                if k.startswith("_"): continue
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = self.dtype if v_batch.is_floating_point() else None
                    v_batch = (
                        v_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(current_bs, J, *v_batch.shape[1:])
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], J, axis=1)
                expanded_infos[k] = v_batch

            # Encode goal if not already present
            if "goal_emb" not in expanded_infos:
                goal = {k: v[:, 0] for k, v in expanded_infos.items() if torch.is_tensor(v)}
                goal["pixels"] = goal["goal"]
                for k in list(expanded_infos.keys()):
                    if k.startswith("goal_"):
                        goal[k[len("goal_"):]] = goal.pop(k)
                goal.pop("action", None)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    enc_goal = self.model.encode(goal)
                g_emb = enc_goal["emb"].detach()
                while g_emb.dim() > 2:
                    g_emb = g_emb.squeeze(1)
                expanded_infos["goal_emb"] = g_emb.unsqueeze(1).expand(current_bs, J, -1)

            # (current_bs, D) goal latent
            goal_latent = expanded_infos["goal_emb"][:, 0]
            if goal_latent.dim() > 2:
                goal_latent = goal_latent.reshape(current_bs, -1)
            D = goal_latent.shape[-1]
            I_D = torch.eye(D, device=self.device, dtype=self.dtype).unsqueeze(0).expand(current_bs, -1, -1)

            # EKI Iteration Loop
            for it in range(self.n_iterations):
                # Clamp actions
                if self.action_clip is not None:
                    ensemble = torch.clamp(ensemble, -self.action_clip, self.action_clip)

                # Flatten ensemble for covariance computation: (current_bs, J, d_flat)
                A_flat = ensemble.reshape(current_bs, J, d_flat)

                # Log ensemble spread (variance across particles)
                current_spread = A_flat.var(dim=1).mean().item()
                self.spread_history.append(current_spread)
                if it < len(self.iteration_spreads):
                    self.iteration_spreads[it].append(current_spread)

                # 1. World Model Rollout (Forward Map)
                rolled_info = self.model.rollout(expanded_infos, ensemble)
                # Predicted final state hat{z}_T: (current_bs, J, D)
                pred_z_seq = rolled_info.get("predicted_emb", rolled_info.get("emb"))
                pred_z = pred_z_seq[..., -1, :].to(dtype=self.dtype)

                if hasattr(self.model, "criterion"):
                    costs = self.model.criterion(rolled_info)
                else:
                    costs = torch.norm(pred_z - goal_latent.unsqueeze(1), dim=-1)

                # If last iteration, evaluate final cost and break
                if it == self.n_iterations - 1:
                    best_idx = torch.argmin(costs, dim=1) # (current_bs,)
                    best_actions = ensemble[torch.arange(current_bs, device=self.device), best_idx]
                    final_actions[start_idx:end_idx] = best_actions
                    final_costs.extend(costs[torch.arange(current_bs, device=self.device), best_idx].cpu().tolist())
                    break

                # 2. Means and Centered Fluctuations
                a_bar = A_flat.mean(dim=1, keepdim=True)        # (current_bs, 1, d_flat)
                z_bar = pred_z.mean(dim=1, keepdim=True)        # (current_bs, 1, D)
                A_tilde = A_flat - a_bar                        # (current_bs, J, d_flat)
                Z_tilde = pred_z - z_bar                        # (current_bs, J, D)

                # 3. Output Covariance C^{zz}: (current_bs, D, D)
                C_zz = torch.bmm(Z_tilde.transpose(1, 2), Z_tilde) / (J - 1)

                # Regularization Gamma = gamma * I_D (or adaptive Levenberg-Marquardt scale)
                reg_scale = self.gamma * torch.clamp(C_zz.diagonal(dim1=1, dim2=2).mean(dim=-1, keepdim=True).unsqueeze(-1), min=1.0)
                M = C_zz + reg_scale * I_D

                # 4. Innovation: (current_bs, J, D)
                R = goal_latent.unsqueeze(1) - pred_z
                if self.stochastic:
                    # Stochastic EKI adds perturbation eta ~ N(0, Gamma)
                    eta = torch.randn_like(R) * math.sqrt(self.gamma)
                    R = R + eta

                # Solve M * W = R^T -> W is (current_bs, D, J)
                W = torch.linalg.solve(M, R.transpose(1, 2))

                # 5. Particle Update: Delta A = 1/(J-1) * (W^T * Z_tilde^T)^T ...
                # Delta A^T = 1/(J-1) * A_tilde^T * Z_tilde * W -> (current_bs, d_flat, J)
                Delta_A_T = torch.bmm(A_tilde.transpose(1, 2), torch.bmm(Z_tilde, W)) / (J - 1)
                Delta_A = Delta_A_T.transpose(1, 2)            # (current_bs, J, d_flat)

                # Apply update
                A_flat_new = A_flat + Delta_A

                # 6. Covariance Inflation (prevents premature ensemble collapse)
                if self.inflation > 1.0:
                    a_bar_new = A_flat_new.mean(dim=1, keepdim=True)
                    A_flat_new = a_bar_new + self.inflation * (A_flat_new - a_bar_new)

                ensemble = A_flat_new.reshape(current_bs, J, H, d_a)

        elapsed = time.time() - start_time
        outputs = {
            "actions": final_actions.detach().cpu(),
            "mean": [final_actions.detach().cpu()],
            "costs": final_costs,
            "elapsed_s": elapsed,
        }
        return outputs


class HybridCEMEKI_Solver:
    """
    Hybrid CEM-EKI Solver:
    - Phase 1 (Mode Selection): 2 CEM iterations to identify the dominant basin of attraction
    - Phase 2 (Gauss-Newton Refinement): 6-8 EKI iterations with self-annealing Gauss-Newton updates
    """
    def __init__(
        self,
        model: nn.Module,
        num_particles: int = 64,
        n_cem_steps: int = 2,
        n_eki_steps: int = 6,
        topk: Optional[int] = None,
        gamma: float = 0.1,
        inflation: float = 1.05,
        device: torch.device = torch.device("cuda:0"),
        seed: int = 42,
        batch_size: int = 25,
        dtype: torch.dtype = torch.float32,
        action_clip: float = 3.0,
        inflation_decay: bool = False,
        use_mean: bool = False,
        **kwargs,
    ):
        self.model = model
        self.num_particles = num_particles
        self.n_cem_steps = n_cem_steps
        self.n_eki_steps = n_eki_steps
        self.topk = topk or max(8, num_particles // 4)
        self.gamma = gamma
        self.inflation = inflation
        self.inflation_decay = inflation_decay
        self.use_mean = use_mean
        self.device = device
        self.base_seed = seed
        self.batch_size = batch_size
        self.dtype = dtype
        self.action_clip = action_clip

        self.horizon: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.action_space = None
        self.env_generators: Dict[int, torch.Generator] = {}
        self.callbacks = []

        self.spread_history: List[float] = []
        self.iteration_spreads: List[List[float]] = [[] for _ in range(self.n_cem_steps + self.n_eki_steps)]

    def configure(self, action_space, n_envs: int, config, **kwargs):
        self.action_space = action_space
        if len(action_space.shape) > 1:
            self._action_dim = int(np.prod(action_space.shape[1:]))
        else:
            self._action_dim = int(np.prod(action_space.shape))
        action_block = getattr(config, "action_block", 1)
        self.action_dim = self._action_dim * action_block
        self.horizon = config.horizon

    def get_env_generator(self, env_id: int):
        if env_id not in self.env_generators:
            gen = torch.Generator(device=self.device).manual_seed(self.base_seed + int(env_id) * 10007)
            self.env_generators[env_id] = gen
        return self.env_generators[env_id]

    def __call__(self, info_dict: dict, init_action: Optional[torch.Tensor] = None) -> dict:
        return self.solve(info_dict, init_action=init_action)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: Optional[torch.Tensor] = None) -> dict:
        start_time = time.time()
        total_envs = len(next(iter(info_dict.values())))

        active_env_ids = info_dict.get("_env_idx")
        if active_env_ids is None:
            active_env_ids = list(range(total_envs))
        elif torch.is_tensor(active_env_ids):
            active_env_ids = active_env_ids.cpu().tolist()

        init_action = prepare_init_action(
            self.model, info_dict, init_action, self.horizon,
            n_envs=total_envs, action_dim=self.action_dim,
        )

        J = self.num_particles
        H = self.horizon
        d_a = self.action_dim
        d_flat = H * d_a

        final_actions = torch.zeros(total_envs, H, d_a, device=self.device, dtype=self.dtype)
        final_costs = []

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_env_ids = active_env_ids[start_idx:end_idx]

            # Initial Gaussian parameters: (current_bs, H, d_a)
            if init_action is not None and torch.is_tensor(init_action):
                batch_mean = init_action[start_idx:end_idx].to(device=self.device, dtype=self.dtype)
            else:
                batch_mean = torch.zeros(current_bs, H, d_a, device=self.device, dtype=self.dtype)
            batch_var = torch.ones_like(batch_mean)

            expanded_infos = {}
            for k, v in info_dict.items():
                if k.startswith("_"): continue
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = self.dtype if v_batch.is_floating_point() else None
                    v_batch = (
                        v_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(current_bs, J, *v_batch.shape[1:])
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], J, axis=1)
                expanded_infos[k] = v_batch

            if "goal_emb" not in expanded_infos:
                goal = {k: v[:, 0] for k, v in expanded_infos.items() if torch.is_tensor(v)}
                goal["pixels"] = goal["goal"]
                for k in list(expanded_infos.keys()):
                    if k.startswith("goal_"):
                        goal[k[len("goal_"):]] = goal.pop(k)
                goal.pop("action", None)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    enc_goal = self.model.encode(goal)
                g_emb = enc_goal["emb"].detach()
                while g_emb.dim() > 2:
                    g_emb = g_emb.squeeze(1)
                expanded_infos["goal_emb"] = g_emb.unsqueeze(1).expand(current_bs, J, -1)

            goal_latent = expanded_infos["goal_emb"][:, 0]
            if goal_latent.dim() > 2:
                goal_latent = goal_latent.reshape(current_bs, -1)
            D = goal_latent.shape[-1]
            I_D = torch.eye(D, device=self.device, dtype=self.dtype).unsqueeze(0).expand(current_bs, -1, -1)

            # -------------------------------------------------------------
            # PHASE 1: CEM Mode Selection (n_cem_steps, e.g. 2)
            # -------------------------------------------------------------
            for step in range(self.n_cem_steps):
                cands_list = []
                for b_i in range(current_bs):
                    gen = self.get_env_generator(batch_env_ids[b_i])
                    c = torch.randn(
                        1, J, H, d_a,
                        generator=gen, device=self.device, dtype=self.dtype,
                    )
                    cands_list.append(c)
                candidates = torch.cat(cands_list, dim=0) * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                cand_flat = candidates.reshape(current_bs, J, d_flat)
                cand_spread = cand_flat.var(dim=1).mean().item()
                self.spread_history.append(cand_spread)
                if step < len(self.iteration_spreads):
                    self.iteration_spreads[step].append(cand_spread)

                rolled_info = self.model.rollout(expanded_infos, candidates)
                pred_z_seq = rolled_info.get("predicted_emb", rolled_info.get("emb"))
                pred_z = pred_z_seq[..., -1, :].to(dtype=self.dtype)
                if hasattr(self.model, "criterion"):
                    costs = self.model.criterion(rolled_info)
                else:
                    costs = torch.norm(pred_z - goal_latent.unsqueeze(1), dim=-1)

                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]

                batch_mean = topk_candidates.mean(dim=1)
                batch_var = torch.clamp(topk_candidates.std(dim=1), min=0.05)

            # -------------------------------------------------------------
            # PHASE 2: EKI Gauss-Newton Refinement (n_eki_steps, e.g. 6)
            # -------------------------------------------------------------
            # Initialize ensemble from the refined mode distribution
            cands_list = []
            for b_i in range(current_bs):
                gen = self.get_env_generator(batch_env_ids[b_i])
                c = torch.randn(
                    1, J, H, d_a,
                    generator=gen, device=self.device, dtype=self.dtype,
                )
                cands_list.append(c)
            ensemble = torch.cat(cands_list, dim=0) * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
            ensemble[:, 0] = batch_mean

            for it in range(self.n_eki_steps):
                if self.action_clip is not None:
                    ensemble = torch.clamp(ensemble, -self.action_clip, self.action_clip)

                A_flat = ensemble.reshape(current_bs, J, d_flat)

                # Record spread
                spread_val = A_flat.var(dim=1).mean().item()
                self.spread_history.append(spread_val)
                step_idx = self.n_cem_steps + it
                if step_idx < len(self.iteration_spreads):
                    self.iteration_spreads[step_idx].append(spread_val)

                rolled_info = self.model.rollout(expanded_infos, ensemble)
                pred_z_seq = rolled_info.get("predicted_emb", rolled_info.get("emb"))
                pred_z = pred_z_seq[..., -1, :].to(dtype=self.dtype)

                if hasattr(self.model, "criterion"):
                    costs = self.model.criterion(rolled_info)
                else:
                    costs = torch.norm(pred_z - goal_latent.unsqueeze(1), dim=-1)

                if it == self.n_eki_steps - 1:
                    if self.use_mean:
                        final_actions[start_idx:end_idx] = ensemble.mean(dim=1)
                        final_costs.extend(costs.mean(dim=1).cpu().tolist())
                    else:
                        best_idx = torch.argmin(costs, dim=1)
                        best_actions = ensemble[torch.arange(current_bs, device=self.device), best_idx]
                        final_actions[start_idx:end_idx] = best_actions
                        final_costs.extend(costs[torch.arange(current_bs, device=self.device), best_idx].cpu().tolist())
                    break

                a_bar = A_flat.mean(dim=1, keepdim=True)
                z_bar = pred_z.mean(dim=1, keepdim=True)
                A_tilde = A_flat - a_bar
                Z_tilde = pred_z - z_bar

                C_zz = torch.bmm(Z_tilde.transpose(1, 2), Z_tilde) / (J - 1)
                reg_scale = self.gamma * torch.clamp(C_zz.diagonal(dim1=1, dim2=2).mean(dim=-1, keepdim=True).unsqueeze(-1), min=1.0)
                M = C_zz + reg_scale * I_D

                R = goal_latent.unsqueeze(1) - pred_z
                W = torch.linalg.solve(M, R.transpose(1, 2))

                Delta_A_T = torch.bmm(A_tilde.transpose(1, 2), torch.bmm(Z_tilde, W)) / (J - 1)
                Delta_A = Delta_A_T.transpose(1, 2)

                A_flat_new = A_flat + Delta_A

                eff_inflation = self.inflation
                if self.inflation_decay and self.inflation > 1.0:
                    decay = max(0.0, 1.0 - (it + 1) / max(1, self.n_eki_steps))
                    eff_inflation = 1.0 + (self.inflation - 1.0) * decay

                if eff_inflation > 1.0:
                    a_bar_new = A_flat_new.mean(dim=1, keepdim=True)
                    A_flat_new = a_bar_new + eff_inflation * (A_flat_new - a_bar_new)

                ensemble = A_flat_new.reshape(current_bs, J, H, d_a)

        elapsed = time.time() - start_time
        outputs = {
            "actions": final_actions.detach().cpu(),
            "mean": [final_actions.detach().cpu()],
            "costs": final_costs,
            "elapsed_s": elapsed,
        }
        return outputs
