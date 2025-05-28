import random
from collections import defaultdict

import numpy as np
import torch

from agents.agent_base import AgentBase
from models.policy_network import PolicyNetwork
from models.transformer import HistoryTransformer


class Level1Agent(AgentBase):
    def __init__(self, agent_id, resource_distribution_strategy, config=None):
        self.agent_id = agent_id
        self.resource_distribution_strategy = resource_distribution_strategy
        self.news = None
        self.history = []
        self.max_history_length = 50

        # Default config if none provided
        if config is None:
            config = {
                "embedding_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "policy_hidden_dim": [128, 64],
                "news_action_dim": 32,
                "distribution_action_dim": 16,
                "learning_rate": 0.0003,
            }

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize transformer for processing history
        self.history_encoder = HistoryTransformer(
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        # Policy networks for news generation and distribution decisions
        self.news_policy = PolicyNetwork(
            input_dim=config["embedding_dim"],
            hidden_dim=config["policy_hidden_dim"],
            output_dim=config["news_action_dim"],
        ).to(self.device)

        self.distribution_policy = PolicyNetwork(
            input_dim=config["embedding_dim"],
            hidden_dim=config["policy_hidden_dim"],
            output_dim=config["distribution_action_dim"],
        ).to(self.device)

        # Value network for critic
        self.value_network = PolicyNetwork(
            input_dim=config["embedding_dim"],
            hidden_dim=config["policy_hidden_dim"],
            output_dim=1,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.history_encoder.parameters())
            + list(self.news_policy.parameters())
            + list(self.distribution_policy.parameters())
            + list(self.value_network.parameters()),
            lr=config["learning_rate"],
        )

        # For experience collection
        self.experiences = []

        # Action and state memory
        self.last_observation = None
        self.last_action = None
        self.last_action_log_prob = None
        self.last_value = None

    def generate_news(self, observation=None):
        """Generate news using either rule-based or learned policy"""
        if observation is not None:
            # Use learned policy
            with torch.no_grad():
                history_embedding = self._process_history(observation)
                news_logits = self.news_policy(history_embedding)
                news_action, log_prob = self._sample_action(news_logits)

                # Store for learning
                self.last_observation = observation
                self.last_action = news_action
                self.last_action_log_prob = log_prob
                self.last_value = self.value_network(history_embedding).item()

                # Convert action to news
                self.news = self._convert_action_to_news(news_action)
        else:
            # Use rule-based approach (for compatibility)
            self.news = (
                f"Agent {self.agent_id} has updated its resource distribution strategy."
            )

        return self.news

    def announce_distribution(self, observation=None):
        """Announce distribution strategy using either rule-based or learned policy"""
        if observation is not None:
            # Use learned policy
            with torch.no_grad():
                history_embedding = self._process_history(observation)
                dist_logits = self.distribution_policy(history_embedding)
                dist_action, log_prob = self._sample_action(dist_logits)

                # Store for learning
                self.last_observation = observation
                self.last_action = dist_action
                self.last_action_log_prob = log_prob
                self.last_value = self.value_network(history_embedding).item()

                # Convert action to distribution strategy
                strategy = self._convert_action_to_strategy(dist_action)
                self.resource_distribution_strategy = strategy

        # Return in the expected format (for compatibility)
        return {
            "agent_id": self.agent_id,
            "strategy": self.resource_distribution_strategy,
        }

    def update_strategy(self, new_strategy):
        """Update the resource distribution strategy"""
        self.resource_distribution_strategy = new_strategy

        # Add to history
        self._add_to_history({"type": "strategy_update", "strategy": new_strategy})

    def receive_news(self, news):
        """Process received news"""
        print(f"Agent {self.agent_id} received news: {news}")

        # Add to history
        self._add_to_history({"type": "received_news", "content": news})

    def update(self, reward, next_observation, done):
        """Update policy based on received reward"""
        if self.last_observation is None:
            return

        # Calculate advantage (simplified)
        with torch.no_grad():
            next_value = (
                0.0
                if done
                else self.value_network(self._process_history(next_observation)).item()
            )

        advantage = reward + 0.99 * next_value - self.last_value

        # Store experience
        experience = {
            "observation": self.last_observation,
            "action": self.last_action,
            "log_prob": self.last_action_log_prob,
            "value": self.last_value,
            "reward": reward,
            "advantage": advantage,
            "return": reward + 0.99 * next_value,
        }

        self.experiences.append(experience)

        # Update history with reward information
        self._add_to_history({"type": "reward", "value": reward})

        # Check if we have enough experiences to update
        if len(self.experiences) >= 32:  # batch size
            self._update_policy()
            self.experiences = []

    def _update_policy(self):
        """Update policy networks using PPO"""
        # Prepare batch data
        observations = [exp["observation"] for exp in self.experiences]
        actions = torch.tensor(
            [exp["action"] for exp in self.experiences],
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            [exp["log_prob"] for exp in self.experiences],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.tensor(
            [exp["advantage"] for exp in self.experiences],
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.tensor(
            [exp["return"] for exp in self.experiences],
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Process history for each observation
        history_embeddings = torch.stack(
            [self._process_history(obs) for obs in observations]
        )

        # Get current policy outputs
        policy_logits = self.news_policy(
            history_embeddings
        )  # Assuming we're updating news policy
        values = self.value_network(history_embeddings).squeeze(-1)

        # Calculate new log probs
        action_probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions.long())

        # PPO clip objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_ratio = 0.2  # PPO clip parameter
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        entropy_coef = 0.01

        # Total loss
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.history_encoder.parameters())
            + list(self.news_policy.parameters())
            + list(self.distribution_policy.parameters())
            + list(self.value_network.parameters()),
            max_norm=0.5,
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

    def _process_history(self, observation):
        """Convert observation history to embedding using transformer"""
        # Extract history from observation and convert to tensor
        if "history" in observation:
            history_data = observation["history"]
        else:
            history_data = self.history[-self.max_history_length :]

        # Convert history to tensor format
        history_tensor = self._convert_history_to_tensor(history_data)

        # Process with transformer
        with torch.no_grad():
            history_embedding = self.history_encoder(history_tensor)

        return history_embedding

    def _convert_history_to_tensor(self, history_data):
        """Convert history data to tensor format for transformer"""
        # Simplified implementation - in practice, you'd need more sophisticated embedding
        history_length = min(len(history_data), self.max_history_length)
        history_tensor = torch.zeros(
            history_length, self.config["embedding_dim"], device=self.device
        )

        for i, item in enumerate(history_data[-history_length:]):
            if item["type"] == "strategy_update":
                # Embed strategy updates
                history_tensor[i, 0] = 1.0  # Type indicator
                # Add more sophisticated embedding based on strategy
            elif item["type"] == "received_news":
                # Embed received news
                history_tensor[i, 1] = 1.0  # Type indicator
                # Add more sophisticated embedding based on news content
            elif item["type"] == "reward":
                # Embed reward information
                history_tensor[i, 2] = 1.0  # Type indicator
                history_tensor[i, 3] = item["value"] / 10.0  # Normalized reward

        return history_tensor

    def _sample_action(self, logits):
        """Sample action from policy logits"""
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Create distribution and sample
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        return action, log_prob

    def _convert_action_to_news(self, action):
        """Convert action index to news content"""
        # Simplified implementation - in practice, this would be more complex
        news_templates = [
            f"Agent {self.agent_id} reports high demand in sector A.",
            f"Agent {self.agent_id} reports low wages in sector B.",
            f"Agent {self.agent_id} announces investment in new infrastructure.",
            f"Agent {self.agent_id} warns about resource shortages.",
            f"Agent {self.agent_id} predicts market growth in coming periods.",
        ]

        # Use action to select a template
        news_index = action % len(news_templates)
        return news_templates[news_index]

    def _convert_action_to_strategy(self, action):
        """Convert action index to distribution strategy"""
        # Simplified implementation - in practice, this would be more complex
        # Create a distribution vector based on the action
        num_sectors = 5  # Assuming 5 sectors/projects

        # Different distribution patterns based on action
        if action % 3 == 0:
            # Balanced distribution
            return [1.0 / num_sectors] * num_sectors
        elif action % 3 == 1:
            # Focus on one sector
            focus_sector = (action // 3) % num_sectors
            strategy = [0.1 / (num_sectors - 1)] * num_sectors
            strategy[focus_sector] = 0.9
            return strategy
        else:
            # Two primary sectors
            sector1 = (action // 3) % num_sectors
            sector2 = (sector1 + 1) % num_sectors
            strategy = [0.1 / (num_sectors - 2)] * num_sectors
            strategy[sector1] = 0.45
            strategy[sector2] = 0.45
            return strategy

    def _add_to_history(self, event):
        """Add an event to the agent's history"""
        self.history.append(event)
        if len(self.history) > self.max_history_length * 2:
            # Keep history from growing too large
            self.history = self.history[-self.max_history_length :]

    def save(self, path):
        """Save agent model"""
        torch.save(
            {
                "history_encoder": self.history_encoder.state_dict(),
                "news_policy": self.news_policy.state_dict(),
                "distribution_policy": self.distribution_policy.state_dict(),
                "value_network": self.value_network.state_dict(),
            },
            path,
        )

    def load(self, path):
        """Load agent model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.history_encoder.load_state_dict(checkpoint["history_encoder"])
        self.news_policy.load_state_dict(checkpoint["news_policy"])
        self.distribution_policy.load_state_dict(checkpoint["distribution_policy"])
        self.value_network.load_state_dict(checkpoint["value_network"])

    def generate_strategic_news(
        self, news_generator, ground_truth=None, deception_level=None
    ):
        """
        Generate strategic news that may be truthful or deceptive

        Args:
            news_generator: NewsGenerator instance
            ground_truth: Actual data about the environment (optional)
            deception_level: How deceptive this agent is (0-1, None=use agent policy)

        Returns:
            Generated news item
        """
        # If we have a learned policy and observation, use it to decide truthfulness
        if self.last_observation is not None and hasattr(self, "news_policy"):
            with torch.no_grad():
                history_embedding = self._process_history(self.last_observation)
                news_logits = self.news_policy(history_embedding)
                news_action, _ = self._sample_action(news_logits)

                # Use action to determine deception level and content focus
                if deception_level is None:
                    deception_level = (
                        news_action % self.config["news_action_dim"]
                    ) / self.config["news_action_dim"]

                # Focus determines what type of news to generate
                focus_idx = (news_action // self.config["news_action_dim"]) % 5
                news_types = [
                    "market_demand",
                    "wage_rates",
                    "resource_availability",
                    "project_profitability",
                    "coalition_performance",
                ]
                news_type = news_types[focus_idx]
        else:
            # Rule-based approach
            if deception_level is None:
                deception_level = np.random.beta(2, 5)  # Usually truth-biased
            news_type = random.choice(
                ["market_demand", "wage_rates", "project_profitability"]
            )

        # Determine if this news will be truthful
        truthful = np.random.random() > deception_level

        # Create content based on ground truth if available, otherwise make it up
        if ground_truth and news_type in ground_truth:
            base_content = ground_truth[news_type].copy()
        else:
            # Generate fictional content
            if news_type == "market_demand":
                base_content = {
                    f"sector_{i}": np.random.uniform(0.5, 1.5) for i in range(5)
                }
            elif news_type == "wage_rates":
                base_content = {
                    f"sector_{i}": np.random.uniform(0.8, 1.2) for i in range(5)
                }
            elif news_type == "project_profitability":
                base_content = {
                    f"project_{i}": np.random.uniform(0.7, 1.3) for i in range(5)
                }
            else:
                base_content = {"value": np.random.uniform(0.7, 1.3)}

        # Generate the news
        news_item = news_generator.generate_news(
            agent_id=self.agent_id,
            content=base_content,
            news_type=news_type,
            truthful=truthful,
        )

        return news_item

    def process_received_news(self, news_items):
        """
        Process news received from other agents

        Args:
            news_items: List of news items

        Returns:
            Processed news summary
        """
        # Store news in history
        for item in news_items:
            self._add_to_history(
                {
                    "type": "received_news",
                    "source": item["agent_id"],
                    "news_type": item["type"],
                    "content": item["content"],
                    "timestamp": item["timestamp"],
                }
            )

        # Summarize news by type
        news_by_type = {}
        for item in news_items:
            news_type = item["type"]
            if news_type not in news_by_type:
                news_by_type[news_type] = []
            news_by_type[news_type].append(item)

        # Process each type of news
        news_summary = {}
        for news_type, items in news_by_type.items():
            if news_type == "market_demand":
                news_summary["market_demand"] = self._process_market_news(items)
            elif news_type == "wage_rates":
                news_summary["wage_rates"] = self._process_wage_news(items)
            elif news_type == "project_profitability":
                news_summary["project_profitability"] = self._process_project_news(
                    items
                )

        return news_summary

    def _process_market_news(self, news_items):
        """Process market demand news"""
        # Simple averaging of reported values
        sector_data = defaultdict(list)
        for item in news_items:
            for sector, value in item["content"].items():
                if isinstance(value, (int, float)):
                    sector_data[sector].append(value)

        # Average the values
        result = {}
        for sector, values in sector_data.items():
            result[sector] = np.mean(values)

        return result

    def _process_wage_news(self, news_items):
        """Process wage news"""
        # Similar to market news processing
        return self._process_market_news(news_items)

    def _process_project_news(self, news_items):
        """Process project profitability news"""
        # Similar to market news processing
        return self._process_market_news(news_items)
