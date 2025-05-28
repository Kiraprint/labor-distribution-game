from collections import defaultdict

import numpy as np
import torch


class SACTrainer:
    def __init__(
        self,
        policy_network,
        q_network1,
        q_network2,
        target_q_network1=None,
        target_q_network2=None,
        memory_buffer=None,
        config=None,
    ):
        """
        Soft Actor-Critic trainer for continuous or discrete action spaces

        Args:
            policy_network: Actor network for policy decisions
            q_network1: First critic Q-network
            q_network2: Second critic Q-network
            target_q_network1: Target network for first critic
            target_q_network2: Target network for second critic
            memory_buffer: Buffer for storing experiences
            config: Training configuration parameters
        """
        self.policy_network = policy_network
        self.q_network1 = q_network1
        self.q_network2 = q_network2

        # If target networks not provided, create copies
        self.target_q_network1 = (
            target_q_network1
            if target_q_network1
            else self._create_target_network(q_network1)
        )
        self.target_q_network2 = (
            target_q_network2
            if target_q_network2
            else self._create_target_network(q_network2)
        )

        self.memory_buffer = memory_buffer

        # Default config if none provided
        if config is None:
            config = {
                "policy_lr": 0.0003,
                "q_lr": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,  # Target network update rate
                "batch_size": 64,
                "alpha": 0.2,  # Entropy coefficient
                "auto_entropy_tuning": True,
                "shared_weights": True,
            }

        self.config = config
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.alpha = config["alpha"]
        self.auto_entropy_tuning = config["auto_entropy_tuning"]

        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy_network.parameters(), lr=config["policy_lr"]
        )

        if config.get("shared_weights", True):
            # Shared parameters for both Q-networks
            self.q_optimizer = torch.optim.Adam(
                list(q_network1.parameters()), lr=config["q_lr"]
            )
        else:
            # Separate optimizers for each Q-network
            self.q1_optimizer = torch.optim.Adam(
                q_network1.parameters(), lr=config["q_lr"]
            )
            self.q2_optimizer = torch.optim.Adam(
                q_network2.parameters(), lr=config["q_lr"]
            )

        # Setup automatic entropy tuning if enabled
        if self.auto_entropy_tuning:
            self.target_entropy = -np.prod(policy_network.action_dim)
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=next(policy_network.parameters()).device
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config["policy_lr"]
            )

        # Training metrics
        self.metrics = defaultdict(list)

    def _create_target_network(self, network):
        """Create a copy of a network for target network"""
        target_net = type(network)(*network.init_args, **network.init_kwargs)
        target_net.load_state_dict(network.state_dict())
        return target_net

    def train(self):
        """Execute a training update using data from memory buffer"""
        if self.memory_buffer.size() < self.batch_size:
            return {"error": "Not enough samples for training"}

        metrics = {}

        for batch in self.memory_buffer.sample(self.batch_size):
            # Update Q-networks and policy
            q_metrics = self._update_critics(batch)
            policy_metrics = self._update_policy(batch)

            # Update target networks
            self._update_target_networks()

            # Update alpha if auto-tuning
            if self.auto_entropy_tuning:
                alpha_metrics = self._update_alpha(batch)
                metrics.update(alpha_metrics)

            metrics.update(q_metrics)
            metrics.update(policy_metrics)

            # Store metrics
            for k, v in metrics.items():
                self.metrics[k].append(v)

        # Return average metrics
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def _update_critics(self, batch):
        """Update Q-networks"""
        states, actions, rewards, next_states, dones = batch

        # Move data to device if necessary
        device = next(self.policy_network.parameters()).device
        states = self._to_tensor(states, device)
        actions = self._to_tensor(actions, device)
        rewards = self._to_tensor(rewards, device)
        next_states = self._to_tensor(next_states, device)
        dones = self._to_tensor(dones, device)

        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_logits = self.policy_network(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-10)

            # Calculate entropy term
            entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

            # Calculate target Q-values
            next_q1 = self.target_q_network1(next_states)
            next_q2 = self.target_q_network2(next_states)

            # Use minimum Q-value for robustness
            next_q = torch.min(next_q1, next_q2)

            # Calculate expected Q-value
            expected_q = rewards + (1 - dones) * self.gamma * (
                next_q + self.alpha * entropy
            )

        # Calculate current Q-values
        current_q1 = self.q_network1(states).gather(1, actions.long())
        current_q2 = self.q_network2(states).gather(1, actions.long())

        # Calculate Q-network losses
        q1_loss = torch.nn.functional.mse_loss(current_q1, expected_q)
        q2_loss = torch.nn.functional.mse_loss(current_q2, expected_q)
        q_loss = q1_loss + q2_loss

        # Update Q-networks
        if hasattr(self, "q_optimizer"):
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
        else:
            # Update each network separately
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_value": current_q1.mean().item(),
        }

    def _update_policy(self, batch):
        """Update policy network"""
        states, _, _, _, _ = batch

        # Move data to device if necessary
        device = next(self.policy_network.parameters()).device
        states = self._to_tensor(states, device)

        # Get policy distribution
        logits = self.policy_network(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)

        # Calculate Q-values for actions
        q1 = self.q_network1(states)
        q2 = self.q_network2(states)
        q = torch.min(q1, q2)

        # Calculate entropy
        entropy = -torch.sum(probs * log_probs, dim=1)

        # Calculate expected value
        inside_term = self.alpha * log_probs - q
        policy_loss = torch.sum(probs * inside_term, dim=1).mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {"policy_loss": policy_loss.item(), "entropy": entropy.mean().item()}

    def _update_alpha(self, batch):
        """Update entropy coefficient alpha if auto-tuning is enabled"""
        if not self.auto_entropy_tuning:
            return {}

        states, _, _, _, _ = batch

        # Move data to device if necessary
        device = next(self.policy_network.parameters()).device
        states = self._to_tensor(states, device)

        # Get policy distribution
        with torch.no_grad():
            logits = self.policy_network(states)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=1)

        # Calculate alpha loss
        alpha = self.log_alpha.exp()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = alpha.item()

        return {"alpha": self.alpha, "alpha_loss": alpha_loss.item()}

    def _update_target_networks(self):
        """Soft update of target networks"""
        self._soft_update(self.q_network1, self.target_q_network1)
        self._soft_update(self.q_network2, self.target_q_network2)

    def _soft_update(self, source, target):
        """Soft update: target = tau * source + (1 - tau) * target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def _to_tensor(self, data, device):
        """Convert data to tensor if it's not already"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        return torch.tensor(data, dtype=torch.float32, device=device)
