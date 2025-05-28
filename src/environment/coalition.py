import itertools
from collections import defaultdict

import numpy as np


class Coalition:
    def __init__(self, agents):
        """
        Initialize a coalition with a set of agents

        Args:
            agents: List of agent IDs that form the coalition
        """
        self.agents = agents
        self.profit_distribution = {}
        self.strategy = None
        self.resources_allocated = 0
        self.total_profit = 0
        self.characteristic_function = {}
        self.formed = False
        self.projects = []

    def form_coalition(self, agent_strategies=None):
        """
        Form a coalition using strategies from member agents

        Args:
            agent_strategies: Dict mapping agent IDs to their preferred strategies
        """
        if not self.agents:
            return False

        if agent_strategies is None:
            agent_strategies = {}

        # Combine strategies from all agents (simple averaging for now)
        combined_strategy = defaultdict(float)
        strategy_count = defaultdict(int)

        for agent_id in self.agents:
            if agent_id in agent_strategies and agent_strategies[agent_id]:
                strategy = agent_strategies[agent_id].get("strategy", {})
                if isinstance(strategy, dict):
                    for project, weight in strategy.items():
                        combined_strategy[project] += weight
                        strategy_count[project] += 1

        # Normalize the combined strategy
        self.strategy = {}
        for project, weight_sum in combined_strategy.items():
            if strategy_count[project] > 0:
                self.strategy[project] = weight_sum / strategy_count[project]

        # If we have a strategy, the coalition is formed
        self.formed = bool(self.strategy)
        return self.formed

    def calculate_profit_distribution(self, total_profit, method="shapley"):
        """
        Calculate profit distribution among agents using either Shapley or Owen value

        Args:
            total_profit: Total profit earned by the coalition
            method: 'shapley' or 'owen' for the distribution method
        """
        self.total_profit = total_profit

        if not self.agents:
            return {}

        if len(self.agents) == 1:
            # Single-agent coalition gets all profit
            self.profit_distribution = {self.agents[0]: total_profit}
            return self.profit_distribution

        # Calculate using the specified method
        if method.lower() == "shapley":
            self._calculate_shapley_values(total_profit)
        elif method.lower() == "owen":
            self._calculate_owen_values(total_profit)
        else:
            # Default to equal distribution
            equal_share = total_profit / len(self.agents)
            self.profit_distribution = {agent: equal_share for agent in self.agents}

        return self.profit_distribution

    def _calculate_shapley_values(self, total_profit):
        """
        Calculate Shapley values for profit distribution
        """
        n = len(self.agents)
        shapley_values: dict[str, float] = {agent: 0 for agent in self.agents}

        # Generate all possible permutations of agents
        all_permutations = list(itertools.permutations(self.agents))

        for perm in all_permutations:
            marginal_contribution = 0
            current_coalition = []

            for agent in perm:
                # Calculate marginal contribution
                previous_value = self._get_coalition_value(current_coalition)
                current_coalition.append(agent)
                new_value = self._get_coalition_value(current_coalition)

                # Add marginal contribution to agent's Shapley value
                shapley_values[agent] += (new_value - previous_value) / len(
                    all_permutations
                )

        # Scale Shapley values to match total profit
        sum_values = sum(shapley_values.values())
        if sum_values > 0:  # Avoid division by zero
            for agent in self.agents:
                shapley_values[agent] = (
                    shapley_values[agent] / sum_values
                ) * total_profit
        else:
            # Equal distribution if all values are 0
            for agent in self.agents:
                shapley_values[agent] = total_profit / n

        self.profit_distribution = shapley_values

    def _calculate_owen_values(self, total_profit):
        """
        Calculate Owen values for profit distribution when agents belong to different groups

        Owen value extends Shapley value for coalitional games with a priori unions
        """
        # For simplicity, we'll assume two levels of coalition structure:
        # Level 1: All agents in this coalition
        # Level 2: Subgroups based on agent types (e.g., Level1 vs Level2 agents)

        # Identify subgroups (could be enhanced with actual agent group info)
        level1_agents = [a for a in self.agents if a.startswith("L1_")]
        level2_agents = [a for a in self.agents if a.startswith("L2_")]
        other_agents = [
            a for a in self.agents if not (a.startswith("L1_") or a.startswith("L2_"))
        ]

        groups = []
        if level1_agents:
            groups.append(level1_agents)
        if level2_agents:
            groups.append(level2_agents)
        if other_agents:
            groups.append(other_agents)

        # If no meaningful groups, fallback to Shapley value
        if len(groups) <= 1:
            return self._calculate_shapley_values(total_profit)

        # Calculate Owen values
        owen_values: dict[str, float] = {agent: 0 for agent in self.agents}

        # Step 1: Calculate Shapley values between groups
        group_shapley = self._calculate_group_shapley(groups)

        # Step 2: Distribute each group's value using Shapley within the group
        for group_idx, group in enumerate(groups):
            group_value = group_shapley[group_idx]

            # Calculate Shapley values within the group
            if len(group) == 1:
                # Single agent gets the full group value
                owen_values[group[0]] = group_value
            else:
                # Multiple agents - distribute using Shapley
                intra_group_shapley = self._calculate_intra_group_shapley(group)

                # Scale values to the group's total value
                for agent in group:
                    owen_values[agent] = intra_group_shapley[agent] * group_value

        # Scale Owen values to match total profit
        sum_values = sum(owen_values.values())
        if sum_values > 0:  # Avoid division by zero
            for agent in self.agents:
                owen_values[agent] = (owen_values[agent] / sum_values) * total_profit
        else:
            # Equal distribution if all values are 0
            for agent in self.agents:
                owen_values[agent] = total_profit / len(self.agents)

        self.profit_distribution = owen_values

    def _calculate_group_shapley(self, groups):
        """Calculate Shapley values between groups"""
        n = len(groups)
        group_shapley: dict[int, float] = {i: 0 for i in range(n)}

        # Generate all possible permutations of group indices
        all_permutations = list(itertools.permutations(range(n)))

        for perm in all_permutations:
            current_coalition = []

            for group_idx in perm:
                # Calculate marginal contribution of this group
                previous_value = self._get_group_coalition_value(
                    current_coalition, groups
                )
                current_coalition.append(group_idx)
                new_value = self._get_group_coalition_value(current_coalition, groups)

                # Add marginal contribution to group's Shapley value
                group_shapley[group_idx] += (new_value - previous_value) / len(
                    all_permutations
                )

        return group_shapley

    def _calculate_intra_group_shapley(self, group: list[str]):
        """Calculate Shapley values within a group"""
        intra_shapley: dict[str, float] = {agent: 0 for agent in group}

        # Generate all possible permutations of agents in this group
        all_permutations = list(itertools.permutations(group))

        for perm in all_permutations:
            current_coalition = []

            for agent in perm:
                # Calculate marginal contribution
                previous_value = self._get_agent_value(current_coalition)
                current_coalition.append(agent)
                new_value = self._get_agent_value(current_coalition)

                # Add marginal contribution to agent's Shapley value
                intra_shapley[agent] += (new_value - previous_value) / len(
                    all_permutations
                )

        # Normalize values to sum to 1
        sum_values = sum(intra_shapley.values())
        if sum_values > 0:  # Avoid division by zero
            for agent in group:
                intra_shapley[agent] = intra_shapley[agent] / sum_values
        else:
            # Equal distribution if all values are 0
            for agent in group:
                intra_shapley[agent] = 1.0 / len(group)

        return intra_shapley

    def _get_coalition_value(self, coalition):
        """
        Get the value of a coalition using the characteristic function

        In practice, this would be based on the game's utility function
        """
        if not coalition:
            return 0

        # Check if we've already calculated this value
        coalition_key = tuple(sorted(coalition))
        if coalition_key in self.characteristic_function:
            return self.characteristic_function[coalition_key]

        # Simple characteristic function: value scales with sqrt of size
        # This captures the idea of synergy but with diminishing returns
        value = np.sqrt(len(coalition))

        # Store for future reference
        self.characteristic_function[coalition_key] = value
        return value

    def _get_group_coalition_value(self, group_indices, groups):
        """Get the value of a coalition of groups"""
        if not group_indices:
            return 0

        # Combine all agents from the specified groups
        coalition = []
        for idx in group_indices:
            coalition.extend(groups[idx])

        return self._get_coalition_value(coalition)

    def _get_agent_value(self, agents):
        """Get the value contribution of a set of agents"""
        # Simple function: each agent contributes 1 unit of value
        return len(agents)

    def get_profit_distribution(self):
        """
        Get the calculated profit distribution

        Returns:
            Dict mapping agent IDs to their profit share
        """
        return self.profit_distribution

    def allocate_resources(self, total_resources):
        """
        Allocate resources according to coalition strategy

        Args:
            total_resources: Total resources available to the coalition

        Returns:
            Dict mapping projects to allocated resources
        """
        if not self.strategy:
            return {}

        self.resources_allocated = total_resources
        allocation = {}

        # Allocate resources according to strategy
        for project, weight in self.strategy.items():
            allocation[project] = weight * total_resources

        return allocation

    def update_agents(self, new_agents):
        """
        Update the agents in the coalition

        Args:
            new_agents: New list of agent IDs
        """
        self.agents = new_agents
        # Reset coalition state since agents changed
        self.formed = False
        self.strategy = None

    def dissolve_coalition(self):
        """
        Dissolve the coalition, resetting all its state
        """
        self.agents = []
        self.profit_distribution = {}
        self.strategy = None
        self.resources_allocated = 0
        self.total_profit = 0
        self.formed = False

    def add_agent(self, agent_id):
        """
        Add a new agent to the coalition

        Args:
            agent_id: ID of the agent to add

        Returns:
            True if agent was added, False if already present
        """
        if agent_id in self.agents:
            return False

        self.agents.append(agent_id)
        # Reset coalition formation since agents changed
        self.formed = False
        return True

    def remove_agent(self, agent_id):
        """
        Remove an agent from the coalition

        Args:
            agent_id: ID of the agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_id not in self.agents:
            return False

        self.agents.remove(agent_id)
        # Reset coalition formation since agents changed
        self.formed = False
        return True

    def size(self):
        """Get the number of agents in the coalition"""
        return len(self.agents)

    def is_formed(self):
        """Check if the coalition is formed"""
        return self.formed
