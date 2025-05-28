import numpy as np
import torch

from agents.agent_base import AgentBase
from models.policy_network import PolicyNetwork
from models.transformer import HistoryTransformer


class Level2Agent(AgentBase):
    def __init__(self, agent_id, initial_trust_scores=None, config=None):
        self.agent_id = agent_id
        self.history = []
        self.max_history_length = 50
        self.trust_scores = initial_trust_scores or {}
        self.coalition = None
        self.coalition_strategy = None

        # Default config if none provided
        if config is None:
            config = {
                "embedding_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "policy_hidden_dims": [128, 64],
                "trust_action_dim": 16,
                "coalition_action_dim": 32,
                "learning_rate": 0.0003,
                "news_embedding_dim": 32,
                "trust_threshold": 0.5,
            }

        self.config = config
        self.trust_threshold = config["trust_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize transformer for processing history
        self.history_encoder = HistoryTransformer(
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        # Policy networks for trust evaluation and coalition decisions
        self.trust_policy = PolicyNetwork(
            input_dim=config["embedding_dim"] + config["news_embedding_dim"],
            hidden_dims=config["policy_hidden_dims"],
            output_dim=config["trust_action_dim"],
        ).to(self.device)

        self.coalition_policy = PolicyNetwork(
            input_dim=config["embedding_dim"],
            hidden_dims=config["policy_hidden_dims"],
            output_dim=config["coalition_action_dim"],
        ).to(self.device)

        # Value network for critic
        self.value_network = PolicyNetwork(
            input_dim=config["embedding_dim"],
            hidden_dims=config["policy_hidden_dims"],
            output_dim=1,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.history_encoder.parameters())
            + list(self.trust_policy.parameters())
            + list(self.coalition_policy.parameters())
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
        self.received_news = {}

    def receive_news(self, news):
        """Process news from Level1 agents"""
        self.received_news = news

        # Add to history
        self._add_to_history({"type": "received_news", "content": news})

        # Evaluate trust for each news source
        if isinstance(news, dict):
            for source, info in news.items():
                self.trust_scores[source] = self.evaluate_trust(source, info)

        return self.trust_scores

    def evaluate_trust(self, source, info, observation=None):
        """Evaluate trustworthiness of news from a source using either rule-based or learned policy"""
        if observation is not None:
            # Use learned policy
            with torch.no_grad():
                # Process history
                history_embedding = self._process_history(observation)

                # Create news embedding
                news_embedding = self._embed_news(source, info)

                # Combine embeddings
                combined_embedding = torch.cat(
                    [history_embedding, news_embedding], dim=0
                )

                # Get trust action
                trust_logits = self.trust_policy(combined_embedding)
                trust_action, log_prob = self._sample_action(trust_logits)

                # Store for learning
                self.last_observation = observation
                self.last_action = trust_action
                self.last_action_log_prob = log_prob
                self.last_value = self.value_network(history_embedding).item()

                # Convert action to trust score
                trust_score = self._convert_action_to_trust(trust_action)
                return trust_score
        else:
            # Fallback to rule-based approach
            # Check if we have history with this source
            past_interactions = [
                h
                for h in self.history
                if h["type"] == "trust_outcome" and h.get("source") == source
            ]

            if past_interactions:
                # Base trust on past outcomes
                positive_outcomes = sum(
                    1 for h in past_interactions if h.get("outcome", 0) > 0
                )
                trust_score = positive_outcomes / len(past_interactions)
            else:
                # No history, use neutral trust
                trust_score = 0.5

            # Add some noise for exploration
            trust_score = max(0, min(1, trust_score + np.random.normal(0, 0.1)))

            return trust_score

    def form_coalition(self, potential_members, observation=None):
        """Form coalition with trusted members using either rule-based or learned policy"""
        # DEBUG LOGGING
        print(
            f"Agent {self.agent_id} forming coalition. Potential members: {potential_members}"
        )
        print(f"Current trust scores: {self.trust_scores}")

        # Lower the trust threshold at the beginning to ensure coalitions form
        adaptive_threshold = max(
            0.2, self.trust_threshold - 0.3 / (self.current_iteration + 1)
        )

        # Filter members based on trust scores with adaptive threshold
        trusted_members = [
            member
            for member in potential_members
            if self.trust_scores.get(member, 0) >= adaptive_threshold
        ]

        # ALWAYS form at least a minimal coalition even with no trusted members
        if not trusted_members and potential_members:
            # Add at least one member to avoid empty coalitions
            trusted_members = [potential_members[0]]
            print(
                f"Forming minimal coalition with {trusted_members[0]} despite low trust"
            )

        if observation is not None:
            # Use learned policy
            with torch.no_grad():
                history_embedding = self._process_history(observation)
                coalition_logits = self.coalition_policy(history_embedding)
                coalition_action, log_prob = self._sample_action(coalition_logits)

                # Store for learning
                self.last_observation = observation
                self.last_action = coalition_action
                self.last_action_log_prob = log_prob
                self.last_value = self.value_network(history_embedding).item()

                # Convert action to coalition strategy
                self.coalition = trusted_members
                self.coalition_strategy = self._convert_action_to_strategy(
                    coalition_action, trusted_members
                )
        else:
            # Use rule-based approach
            if trusted_members:
                self.coalition = trusted_members
                self.coalition_strategy = self._rule_based_strategy(trusted_members)
            else:
                self.coalition = []
                self.coalition_strategy = None

        # DEBUG LOGGING
        print(
            f"Coalition formed: {self.coalition} with strategy: {self.coalition_strategy}"
        )

        return {"coalition": self.coalition, "strategy": self.coalition_strategy}

    def execute_strategy(self, infrastructure_objects):
        """Execute the chosen strategy on infrastructure objects"""
        if not self.coalition_strategy:
            # Default strategy: equal distribution
            num_objects = len(infrastructure_objects)
            return {obj: 1.0 / num_objects for obj in infrastructure_objects}

        # Apply the coalition strategy to distribute resources
        distribution = {}
        strategy_values = list(self.coalition_strategy.values())

        for i, obj in enumerate(infrastructure_objects):
            if i < len(strategy_values):
                distribution[obj] = strategy_values[i]
            else:
                # If more objects than strategy values, distribute remaining equally
                remaining = 1.0 - sum(distribution.values())
                remaining_objects = len(infrastructure_objects) - i
                distribution[obj] = remaining / remaining_objects

        return distribution

    def update_trust_scores(self, feedback):
        """Update trust scores based on feedback"""
        for source, outcome in feedback.items():
            if source in self.trust_scores:
                # Adjust trust score based on outcome
                old_score = self.trust_scores[source]
                # Simple trust update rule - can be more sophisticated
                new_score = old_score * 0.7 + (outcome > 0) * 0.3
                self.trust_scores[source] = new_score

                # Add to history
                self._add_to_history(
                    {
                        "type": "trust_outcome",
                        "source": source,
                        "old_score": old_score,
                        "new_score": new_score,
                        "outcome": outcome,
                    }
                )

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

        # Get current policy outputs (assuming we're updating the coalition policy)
        policy_logits = self.coalition_policy(history_embeddings)
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
            + list(self.trust_policy.parameters())
            + list(self.coalition_policy.parameters())
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
            if item["type"] == "received_news":
                # Embed received news
                history_tensor[i, 0] = 1.0  # Type indicator
                # Add more sophisticated embedding based on news content
            elif item["type"] == "trust_outcome":
                # Embed trust outcomes
                history_tensor[i, 1] = 1.0  # Type indicator
                history_tensor[i, 2] = item.get("new_score", 0.5)
                history_tensor[i, 3] = float(item.get("outcome", 0) > 0)
            elif item["type"] == "reward":
                # Embed reward information
                history_tensor[i, 4] = 1.0  # Type indicator
                history_tensor[i, 5] = item["value"] / 10.0  # Normalized reward

        return history_tensor

    def _embed_news(self, source, info):
        """Create embedding for news content"""
        # Simple embedding creation for news
        news_embedding = torch.zeros(
            self.config["news_embedding_dim"], device=self.device
        )

        # Source identifier (one-hot encoding would be better with many sources)
        news_embedding[0] = hash(source) % 10 / 10.0

        # Add more sophisticated content embedding
        if isinstance(info, dict):
            # Extract numeric values
            for i, (key, value) in enumerate(info.items()):
                if (
                    isinstance(value, (int, float))
                    and i < self.config["news_embedding_dim"] - 1
                ):
                    news_embedding[i + 1] = float(value) / 10.0  # Normalize
        elif isinstance(info, str):
            # For string news, use simple hash-based approach
            news_embedding[1] = hash(info) % 1000 / 1000.0

        return news_embedding

    def _sample_action(self, logits):
        """Sample action from policy logits"""
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Create distribution and sample
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        return action, log_prob

    def _convert_action_to_trust(self, action):
        """Convert action index to trust score"""
        # Simplified implementation
        # Map action to trust score in [0, 1] range
        trust_score = action / (self.config["trust_action_dim"] - 1)
        return trust_score

    def _convert_action_to_strategy(self, action, members):
        """Convert action index to coalition strategy"""
        # Simplified implementation
        num_projects = 5  # Assuming 5 infrastructure projects

        if not members:
            return None

        # Different distribution patterns based on action
        if action % 3 == 0:
            # Balanced distribution
            strategy = {f"project_{i}": 1.0 / num_projects for i in range(num_projects)}
        elif action % 3 == 1:
            # Focus on one project
            focus_project = (action // 3) % num_projects
            strategy = {
                f"project_{i}": 0.1 / (num_projects - 1) for i in range(num_projects)
            }
            strategy[f"project_{focus_project}"] = 0.9
        else:
            # Two primary projects
            project1 = (action // 3) % num_projects
            project2 = (project1 + 1) % num_projects
            strategy = {
                f"project_{i}": 0.1 / (num_projects - 2) for i in range(num_projects)
            }
            strategy[f"project_{project1}"] = 0.45
            strategy[f"project_{project2}"] = 0.45

        return strategy

    def _rule_based_strategy(self, members):
        """Generate a rule-based strategy when not using learned policy"""
        num_projects = 5  # Assuming 5 infrastructure projects

        # Simple rule: If we have few members, focus on fewer projects
        if len(members) <= 2:
            # Focus on one project
            focus_project = hash(tuple(members)) % num_projects
            strategy = {
                f"project_{i}": 0.1 / (num_projects - 1) for i in range(num_projects)
            }
            strategy[f"project_{focus_project}"] = 0.9
        else:
            # More members, distribute more evenly
            strategy = {f"project_{i}": 1.0 / num_projects for i in range(num_projects)}

        return strategy

    def _add_to_history(self, event):
        """Add an event to the agent's history"""
        self.history.append(event)
        if len(self.history) > self.max_history_length * 2:
            # Keep history from growing too large
            self.history = self.history[-self.max_history_length :]

    def get_potential_members(self, all_agents=None):
        """Identify potential coalition members"""
        if all_agents is None:
            return []

        # Filter out self and agents with low trust scores
        return [agent_id for agent_id in all_agents if agent_id != self.agent_id]

    def save(self, path):
        """Save agent model"""
        torch.save(
            {
                "history_encoder": self.history_encoder.state_dict(),
                "trust_policy": self.trust_policy.state_dict(),
                "coalition_policy": self.coalition_policy.state_dict(),
                "value_network": self.value_network.state_dict(),
                "trust_scores": self.trust_scores,
            },
            path,
        )

    def load(self, path):
        """Load agent model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.history_encoder.load_state_dict(checkpoint["history_encoder"])
        self.trust_policy.load_state_dict(checkpoint["trust_policy"])
        self.coalition_policy.load_state_dict(checkpoint["coalition_policy"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.trust_scores = checkpoint["trust_scores"]

    def evaluate_news_trustworthiness(self, news_items):
        """
        Evaluate the trustworthiness of received news based on source reputation
        and content plausibility

        Args:
            news_items: List of news items to evaluate

        Returns:
            Dict mapping news item IDs to trustworthiness scores (0-1)
        """
        trust_scores = {}

        for idx, item in enumerate(news_items):
            source_id = item["agent_id"]

            # Base score on source reputation
            source_trust = self.trust_scores.get(source_id, 0.5)

            # Adjust based on content plausibility
            content_plausibility = self._assess_content_plausibility(item)

            # Combine the scores (weighted average)
            trustworthiness = 0.7 * source_trust + 0.3 * content_plausibility

            # Store the result
            trust_scores[idx] = trustworthiness

            # Add to history
            self._add_to_history(
                {
                    "type": "news_evaluation",
                    "source": source_id,
                    "trust_score": trustworthiness,
                }
            )

        return trust_scores

    def _assess_content_plausibility(self, news_item):
        """
        Assess how plausible a news item's content is based on historical data
        and expected ranges

        Returns a score between 0 (implausible) and 1 (plausible)
        """
        news_type = news_item.get("type", "")
        content = news_item.get("content", {})

        if not isinstance(content, dict):
            return 0.5  # Can't assess non-dict content

        # Get historical values for this news type
        history_items = [
            h
            for h in self.history
            if h.get("type") == "received_news" and h.get("news_type") == news_type
        ]

        if not history_items:
            return 0.5  # No history to compare against

        # Check each value against historical range
        plausibility_scores = []

        for key, value in content.items():
            if not isinstance(value, (int, float)):
                continue

            # Get historical values for this key
            historical_values = [
                h["content"].get(key, None)
                for h in history_items
                if isinstance(h.get("content", {}), dict) and key in h["content"]
            ]

            historical_values = [
                v for v in historical_values if isinstance(v, (int, float))
            ]

            if not historical_values:
                plausibility_scores.append(0.5)
                continue

            # Calculate mean and standard deviation
            mean_value = np.mean(historical_values)
            std_value = (
                np.std(historical_values)
                if len(historical_values) > 1
                else mean_value * 0.2
            )

            # Normalize the distance from mean
            if std_value > 0:
                z_score = (abs(value - mean_value) / std_value).item()
                # Convert to plausibility score (high z-score = low plausibility)
                plausibility = max(0, min(1, 1 - min(z_score / 3, 1)))
            else:
                # If std is 0, check if value matches the mean
                plausibility = 1.0 if value == mean_value else 0.5

            plausibility_scores.append(plausibility)

        # Return average plausibility
        return np.mean(plausibility_scores) if plausibility_scores else 0.5

    def update_trust_based_on_news_accuracy(self, news_evaluation_results):
        """
        Update trust scores based on how accurate news turned out to be

        Args:
            news_evaluation_results: Dict mapping news items to accuracy scores
        """
        for news_item, accuracy in news_evaluation_results.items():
            source_id = news_item["agent_id"]

            # Update trust score for this source
            current_trust = self.trust_scores.get(source_id, 0.5)

            # More weight to recent information
            updated_trust = 0.7 * current_trust + 0.3 * accuracy
            self.trust_scores[source_id] = updated_trust

            # Add to history
            self._add_to_history(
                {
                    "type": "trust_update",
                    "source": source_id,
                    "old_trust": current_trust,
                    "new_trust": updated_trust,
                    "accuracy": accuracy,
                }
            )

        return self.trust_scores

    def filter_news_by_trust(self, news_items, min_trust=None):
        """
        Filter news items based on trustworthiness

        Args:
            news_items: List of news items
            min_trust: Minimum trust threshold (default: use agent's threshold)

        Returns:
            List of trusted news items
        """
        if min_trust is None:
            min_trust = self.trust_threshold

        trusted_news = []

        for item in news_items:
            source_id = item["agent_id"]
            if self.trust_scores.get(source_id, 0) >= min_trust:
                trusted_news.append(item)

        return trusted_news
