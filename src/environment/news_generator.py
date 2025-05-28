import random
from collections import defaultdict

import numpy as np


class NewsGenerator:
    def __init__(self, config=None):
        self.news = []
        self.news_history = {}  # Maps agent_id to list of news they've generated
        self.ground_truth = {}  # Actual state of the world
        self.truthfulness_scores = {}  # Track how truthful each agent is

        # Default config if none provided
        if config is None:
            config = {
                "max_news_age": 3,  # How many turns news stays relevant
                "news_types": [
                    "market_demand",
                    "wage_rates",
                    "resource_availability",
                    "project_profitability",
                    "coalition_performance",
                ],
                "deception_range": (0.5, 2.0),  # Range for false information multiplier
                "truth_probability": 0.7,  # Probability of generating truthful news
            }

        self.config = config
        self.current_turn = 0

    def set_ground_truth(self, truth_data):
        """Set the ground truth data about the world state"""
        self.ground_truth = truth_data

    def generate_news(
        self, agent_id, content, news_type=None, truthful=None, timestamp=None
    ):
        """
        Generate news from an agent. Can be truthful or deceptive.

        Args:
            agent_id: ID of the agent generating the news
            content: Content of the news (dict or string)
            news_type: Type of news being generated
            truthful: Whether the news is truthful (if None, determined randomly)
            timestamp: When the news was generated (defaults to current turn)

        Returns:
            The generated news item
        """
        if timestamp is None:
            timestamp = self.current_turn

        if news_type is None and self.config["news_types"]:
            news_type = random.choice(self.config["news_types"])

        if truthful is None:
            # Determine truthfulness based on config probability and agent history
            agent_truth_score = self.truthfulness_scores.get(agent_id, 0.5)
            truthful = (
                random.random()
                < (self.config["truth_probability"] + agent_truth_score) / 2
            )

        news_item = {
            "agent_id": agent_id,
            "content": content,
            "type": news_type,
            "timestamp": timestamp,
            "truthful": truthful,  # Note: This is hidden from receiving agents
        }

        # If news is not truthful, modify the content accordingly
        if (
            not truthful
            and isinstance(content, dict)
            and news_type in self.ground_truth
        ):
            news_item["content"] = self._distort_content(content, news_type)

        self.news.append(news_item)

        # Update agent's news history
        if agent_id not in self.news_history:
            self.news_history[agent_id] = []
        self.news_history[agent_id].append(news_item)

        return news_item

    def _distort_content(self, content, news_type):
        """Generate distorted version of content for false news"""
        distorted = content.copy()

        # Get distortion factor from config
        min_factor, max_factor = self.config["deception_range"]

        # Distort numeric values
        for key, value in content.items():
            if isinstance(value, (int, float)):
                # Randomly inflate or deflate the value
                distortion_factor = np.random.uniform(min_factor, max_factor)
                if random.random() < 0.5:  # 50% chance to invert the distortion
                    distortion_factor = 1.0 / distortion_factor

                distorted[key] = value * distortion_factor

        return distorted

    def distribute_news(self, target_agents=None):
        """
        Distribute news to target agents

        Args:
            target_agents: List of agent IDs to distribute news to (None = all)

        Returns:
            Dict mapping agent IDs to the news they receive
        """
        news_distribution = defaultdict(list)

        # Filter news to only include recent ones
        recent_news = [
            news
            for news in self.news
            if self.current_turn - news["timestamp"] <= self.config["max_news_age"]
        ]

        # Distribute news to specified agents or all agents
        for news_item in recent_news:
            # Create a version without the truthfulness flag
            public_item = {k: v for k, v in news_item.items() if k != "truthful"}

            if target_agents is None:
                # Send to all agents except the source
                for agent_id in self.news_history.keys():
                    if agent_id != news_item["agent_id"]:
                        news_distribution[agent_id].append(public_item)
            else:
                # Send only to specified agents
                for agent_id in target_agents:
                    if agent_id != news_item["agent_id"]:
                        news_distribution[agent_id].append(public_item)

        return dict(news_distribution)

    def get_news_for_agent(self, agent_id):
        """Get all news available to a specific agent"""
        agent_news = []

        for news_item in self.news:
            if news_item["agent_id"] != agent_id:  # Agents don't receive their own news
                # Create a version without the truthfulness flag
                public_item = {k: v for k, v in news_item.items() if k != "truthful"}
                # Only include recent news
                if (
                    self.current_turn - news_item["timestamp"]
                    <= self.config["max_news_age"]
                ):
                    agent_news.append(public_item)

        return agent_news

    def get_news_by_type(self, news_type):
        """Get all news of a specific type"""
        return [news for news in self.news if news["type"] == news_type]

    def evaluate_news_accuracy(self, news_item):
        """
        Evaluate how accurate a news item is compared to ground truth
        Returns a score between 0 (completely false) and 1 (completely true)
        """
        if (
            not isinstance(news_item["content"], dict)
            or news_item["type"] not in self.ground_truth
        ):
            return 0.5  # Can't evaluate non-numeric or news without ground truth

        truth = self.ground_truth[news_item["type"]]
        content = news_item["content"]

        # Calculate accuracy as inverse of normalized difference
        accuracy_scores = []
        for key in content:
            if key in truth and isinstance(content[key], (int, float)):
                # Calculate relative error
                if truth[key] == 0:
                    if content[key] == 0:
                        accuracy = 1.0
                    else:
                        accuracy = 0.0
                else:
                    relative_error = abs(content[key] - truth[key]) / abs(truth[key])
                    accuracy = max(0, 1 - min(relative_error, 1))
                accuracy_scores.append(accuracy)

        # Return average accuracy if we have scores, otherwise 0.5
        return np.mean(accuracy_scores) if accuracy_scores else 0.5

    def update_agent_truthfulness(self, agent_id, accuracy_score):
        """Update an agent's truthfulness score based on news accuracy"""
        current_score = self.truthfulness_scores.get(agent_id, 0.5)
        # Exponential moving average with 0.3 weight for new information
        self.truthfulness_scores[agent_id] = 0.7 * current_score + 0.3 * accuracy_score

    def advance_turn(self):
        """Advance to the next turn"""
        self.current_turn += 1

    def clear_news(self):
        """Clear all current news"""
        self.news = []

    def clear_old_news(self):
        """Remove news older than max_news_age"""
        self.news = [
            news
            for news in self.news
            if self.current_turn - news["timestamp"] <= self.config["max_news_age"]
        ]

    def generate_market_news(self, agent_id, sector_data, truthful=None):
        """Generate news about market conditions in various sectors"""
        return self.generate_news(
            agent_id=agent_id,
            content=sector_data,
            news_type="market_demand",
            truthful=truthful,
        )

    def generate_wage_news(self, agent_id, wage_data, truthful=None):
        """Generate news about wage rates in various sectors"""
        return self.generate_news(
            agent_id=agent_id,
            content=wage_data,
            news_type="wage_rates",
            truthful=truthful,
        )

    def generate_project_news(self, agent_id, project_data, truthful=None):
        """Generate news about project profitability"""
        return self.generate_news(
            agent_id=agent_id,
            content=project_data,
            news_type="project_profitability",
            truthful=truthful,
        )
