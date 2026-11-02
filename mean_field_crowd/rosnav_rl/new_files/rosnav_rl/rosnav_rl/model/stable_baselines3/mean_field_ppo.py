"""PPO with a one-second crowd-density auxiliary loss.

The forecast is conditioned on the previous command, the same input the policy
uses at inference:

    rho_hat_{t+10} = F(rho_t, rho_{t-1}, a_{t-1})

``a_{t-1}`` is ``[linear.x, angular.z]``. The target is the recorded density
ten steps later. Pairs that would cross an episode boundary, or run off the
end of the rollout, are left out of the loss. The response network is a
submodule of the policy, so this term and the navigation loss update it in the
same backward pass.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.action import (
    previous_command,
)
from rosnav_rl.model.stable_baselines3.policy.feature_extractors.mean_field.targets import (
    flatten_like_rollout,
    masked_density_loss,
    same_episode_forecast_mask,
)

# Gym dict keys. Kept as strings so this module does not import the space classes.
_DENSITY_KEY = "PedestrianDensitySpace"
_LAST_ACTION_KEY = "LastActionSpace"
_DENSITY_CURRENT = 0
_DENSITY_PREVIOUS = 1


class MeanFieldDictRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer that remembers the minibatch's flat indices."""

    def _get_samples(self, batch_inds: np.ndarray, env=None):
        self.last_batch_inds = batch_inds
        return super()._get_samples(batch_inds, env)


class MeanFieldPPO(PPO):
    """PPO plus the mean-squared one-second density forecast."""

    def __init__(self, *args, density_coef: float = 1.0, forecast_horizon: int = 10, **kwargs):
        self.density_coef = density_coef
        self.forecast_horizon = forecast_horizon
        kwargs.setdefault("rollout_buffer_class", MeanFieldDictRolloutBuffer)
        super().__init__(*args, **kwargs)

    def _cache_density_forecast(self) -> None:
        """Flatten density, previous command, target, and the same-episode mask."""
        buffer = self.rollout_buffer
        if buffer.generator_ready:
            raise RuntimeError("density targets must be cached before the rollout buffer is flattened")
        density = buffer.observations[_DENSITY_KEY]
        last_action = buffer.observations[_LAST_ACTION_KEY]
        valid = same_episode_forecast_mask(buffer.episode_starts, self.forecast_horizon)
        target = np.zeros(density.shape[:2] + density.shape[3:], dtype=np.float32)
        horizon = self.forecast_horizon
        if density.shape[0] > horizon:
            target[:-horizon] = density[horizon:, :, _DENSITY_CURRENT]
        self._density_rho_t = buffer.to_torch(flatten_like_rollout(density[:, :, _DENSITY_CURRENT]))
        self._density_rho_prev = buffer.to_torch(flatten_like_rollout(density[:, :, _DENSITY_PREVIOUS]))
        self._density_velocity = buffer.to_torch(flatten_like_rollout(previous_command(last_action))).float()
        self._density_target = buffer.to_torch(flatten_like_rollout(target))
        self._density_valid = buffer.to_torch(flatten_like_rollout(valid.astype(np.float32))).bool()

    def _density_loss(self, batch_inds: np.ndarray) -> th.Tensor:
        index = th.as_tensor(batch_inds, device=self.device)
        valid = self._density_valid[index]
        rho_t = self._density_rho_t[index].unsqueeze(1)
        rho_prev = self._density_rho_prev[index].unsqueeze(1)
        velocity = self._density_velocity[index]
        prediction = self.policy.features_extractor.response_net(rho_t, rho_prev, velocity)
        return masked_density_loss(prediction, self._density_target[index].unsqueeze(1), valid)

    def _density_rollout_stats(self) -> tuple[int, float, float]:
        """Valid same-episode pairs and the raw density mass on those pairs."""
        valid = self._density_valid
        count = int(valid.sum().item())
        if count == 0:
            return 0, 0.0, 0.0
        return (
            count,
            float(self._density_rho_t[valid].sum().item()),
            float(self._density_target[valid].sum().item()),
        )

    def _response_grad_norm(self) -> float:
        """L2 norm of the response-network gradient, before the shared PPO clip."""
        extractor = getattr(self.policy, "features_extractor", None)
        response_net = getattr(extractor, "response_net", None)
        if response_net is None:
            return 0.0
        squares = [
            parameter.grad.detach().float().pow(2).sum()
            for parameter in response_net.parameters()
            if parameter.grad is not None
        ]
        if not squares:
            return 0.0
        return float(th.sqrt(th.stack(squares).sum()).item())

    def train(self) -> None:
        """PPO update with the density forecast added to the minibatch loss."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, density_losses = [], [], []
        clip_fractions = []
        response_grad_norms = []
        self._cache_density_forecast()
        valid_pairs, current_sum, target_sum = self._density_rollout_stats()

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                density_loss = self._density_loss(self.rollout_buffer.last_batch_inds)
                density_losses.append(density_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.density_coef * density_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                response_grad_norms.append(self._response_grad_norm())
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        mean_density_loss = float(np.mean(density_losses)) if density_losses else 0.0
        mean_response_grad = float(np.mean(response_grad_norms)) if response_grad_norms else 0.0
        self.logger.record("train/density_loss", mean_density_loss)
        self.logger.record("train/density_valid_pairs", valid_pairs)
        self.logger.record("train/density_current_sum", current_sum)
        self.logger.record("train/density_target_sum", target_sum)
        self.logger.record("train/response_grad_norm", mean_response_grad)
        print(
            "[mean-field] "
            f"density_loss={mean_density_loss:.6f} "
            f"valid_pairs={valid_pairs} "
            f"current_sum={current_sum:.4f} "
            f"target_sum={target_sum:.4f} "
            f"response_grad_norm={mean_response_grad:.6f}",
            flush=True,
        )
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
