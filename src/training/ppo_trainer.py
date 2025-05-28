from collections import defaultdict

import numpy as np
import torch


class PPOTrainer:
    def __init__(
        self, policy_network, value_network=None, memory_buffer=None, config=None
    ):
        """
        Proximal Policy Optimization trainer for agent learning

        Args:
            policy_network: Neural network for policy decisions
            value_network: Neural network for value estimation (if separate from policy)
            memory_buffer: Buffer for storing experiences
            config: Training configuration parameters
        """
        self.policy_network = policy_network
        self.value_network = value_network if value_network else policy_network
        self.memory_buffer = memory_buffer

        # Default config if none provided
        if config is None:
            config = {
                "learning_rate": 0.0003,
                "clip_ratio": 0.2,
                "gamma": 0.99,
                "lam": 0.95,
                "batch_size": 64,
                "update_epochs": 10,
                "ppo_epochs": 4,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "target_kl": 0.01,
                "shared_weights": True,
            }

        self.config = config
        self.clip_ratio = config["clip_ratio"]
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.batch_size = config["batch_size"]
        self.update_epochs = config["update_epochs"]
        self.value_coef = config["value_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.target_kl = config["target_kl"]

        # Initialize optimizer - shared or separate
        if config.get("shared_weights", True) and hasattr(
            policy_network, "get_all_parameters"
        ):
            self.optimizer = torch.optim.Adam(
                policy_network.get_all_parameters(), lr=config["learning_rate"]
            )
        else:
            params = list(policy_network.parameters())
            if value_network and value_network is not policy_network:
                params.extend(list(value_network.parameters()))
            self.optimizer = torch.optim.Adam(params, lr=config["learning_rate"])

        # Training metrics
        self.metrics = defaultdict(list)

    def train(self):
        """Execute a training update using data from memory buffer"""
        if self.memory_buffer.size() < self.batch_size:
            return {"error": "Not enough samples for training"}

        for _ in range(self.update_epochs):
            for batch in self.memory_buffer.sample(self.batch_size):
                metrics = self._update_policy(batch)
                for k, v in metrics.items():
                    self.metrics[k].append(v)

        # Return average metrics
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def _update_policy(self, batch):
        """Update policy and value networks using a batch of experiences"""
        (
            states,
            actions,
            old_log_probs,
            values,
            rewards,
            next_states,
            dones,
            advantages,
            returns,
        ) = batch

        # Move data to device if necessary
        device = next(self.policy_network.parameters()).device
        states = self._to_tensor(states, device)
        actions = self._to_tensor(actions, device)
        old_log_probs = self._to_tensor(old_log_probs, device)
        advantages = self._to_tensor(advantages, device)
        returns = self._to_tensor(returns, device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track metrics
        metrics = {}

        # PPO update loop
        for _ in range(self.config["ppo_epochs"]):
            # Get new log probs and entropy
            logits = self.policy_network(states)
            new_log_probs, entropy = self._get_log_probs_and_entropy(logits, actions)

            # Calculate value predictions
            new_values = self.value_network(states).squeeze(-1)

            # Calculate policy loss (clipped surrogate objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = (
                torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Calculate value loss
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.max_grad_norm
                )
                if self.value_network is not self.policy_network:
                    torch.nn.utils.clip_grad_norm_(
                        self.value_network.parameters(), self.max_grad_norm
                    )

            self.optimizer.step()

            # Calculate approximate KL divergence for early stopping
            approx_kl = ((old_log_probs - new_log_probs) ** 2).mean().item()
            if approx_kl > self.target_kl:
                break

            # Update metrics
            metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.mean().item(),
                "approx_kl": approx_kl,
                "clip_fraction": (torch.abs(ratio - 1.0) > self.clip_ratio)
                .float()
                .mean()
                .item(),
            }

        return metrics

    def _compute_advantages_and_returns(self, states, rewards, next_states, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        # Convert to tensors if needed
        device = next(self.policy_network.parameters()).device
        states = self._to_tensor(states, device)
        next_states = self._to_tensor(next_states, device)
        rewards = self._to_tensor(rewards, device)
        dones = self._to_tensor(dones, device)

        # Get value predictions
        with torch.no_grad():
            values = self.value_network(states).squeeze(-1)
            next_values = self.value_network(next_states).squeeze(-1)

        # Calculate GAE advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = (
                rewards[t] + self.gamma * next_values[t] * next_non_terminal - values[t]
            )
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return advantages.detach(), returns.detach()

    def _get_log_probs_and_entropy(self, logits, actions):
        """Calculate log probabilities and entropy from logits"""
        # For categorical actions
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy

    def _to_tensor(self, data, device):
        """Convert data to tensor if it's not already"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        return torch.tensor(data, dtype=torch.float32, device=device)

    def process_batch_for_training(self, batch):
        """Process a batch of experiences into the format needed for training"""
        states, actions, rewards, next_states, dones = batch

        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(
            states, rewards, next_states, dones
        )

        # Get old log probs and values
        with torch.no_grad():
            logits = self.policy_network(states)
            old_log_probs, _ = self._get_log_probs_and_entropy(logits, actions)
            values = self.value_network(states).squeeze(-1)

        return (
            states,
            actions,
            old_log_probs,
            values,
            rewards,
            next_states,
            dones,
            advantages,
            returns,
        )
