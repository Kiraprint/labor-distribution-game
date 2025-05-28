from collections import defaultdict

import numpy as np

from agents.level1_agent import Level1Agent
from agents.level2_agent import Level2Agent
from environment.coalition import Coalition
from environment.news_generator import NewsGenerator


class Game:
    def __init__(self, config=None):
        """Initialize the labor distribution game"""
        # Default config if none provided
        if config is None:
            config = {
                "num_level1_agents": 5,
                "num_level2_agents": 3,
                "num_projects": 5,
                "small_project_factor": 0.3,  # Profit factor for small projects vs large
                "coalition_bonus": 1.5,  # Multiplier for coalition profits
                "profit_distribution_method": "shapley",  # "shapley" or "owen"
                "base_project_profitability": 10.0,
                "project_variability": 0.2,  # Variability in project profitability
                "truthfulness_reward": 0.1,  # Reward for truthful news
                "efficient_allocation_bonus": 0.2,  # Bonus for efficient resource allocation
            }

        self.config = config

        # Initialize agents
        self.level1_agents = []
        self.level2_agents = []
        self._create_agents()

        # Initialize projects and infrastructure
        self.projects = self._initialize_projects()

        # Initialize news generator
        self.news_generator = NewsGenerator()

        # Coalition tracking
        self.coalitions = []

        # Game state
        self.current_iteration = 0
        self.ground_truth = {}
        self.agent_profits = defaultdict(float)
        self.coalition_profits = defaultdict(float)
        self.trust_network = defaultdict(dict)

        # Set up initial ground truth
        self._update_ground_truth()

    def _create_agents(self):
        """Create level 1 and level 2 agents"""
        # Create Level 1 agents
        for i in range(self.config["num_level1_agents"]):
            agent_id = f"L1_{i}"
            # Initial distribution strategy - equal allocation to all projects
            initial_strategy = {
                f"project_{j}": 1.0 / self.config["num_projects"]
                for j in range(self.config["num_projects"])
            }
            agent = Level1Agent(agent_id, initial_strategy)
            self.level1_agents.append(agent)

        # Create Level 2 agents
        for i in range(self.config["num_level2_agents"]):
            agent_id = f"L2_{i}"
            # Initial trust scores - neutral towards all agents
            initial_trust = {a.agent_id: 0.5 for a in self.level1_agents}
            agent = Level2Agent(agent_id, initial_trust)
            self.level2_agents.append(agent)

    def _initialize_projects(self):
        """Initialize projects with their base profitability"""
        projects = {}
        for i in range(self.config["num_projects"]):
            project_id = f"project_{i}"
            # Random base profitability around the configured value
            base_profit = self.config["base_project_profitability"] * (
                1
                + np.random.uniform(
                    -self.config["project_variability"],
                    self.config["project_variability"],
                )
            )
            projects[project_id] = {
                "id": project_id,
                "base_profitability": base_profit,
                "current_profitability": base_profit,
                "allocated_resources": 0,
                "optimal_resources": np.random.randint(
                    50, 200
                ),  # Random optimal allocation
            }
        return projects

    def _update_ground_truth(self):
        """Update the ground truth about the environment"""
        # Update project profitability in ground truth
        project_truth = {}
        for project_id, project in self.projects.items():
            project_truth[project_id] = project["current_profitability"]

        self.ground_truth["project_profitability"] = project_truth

        # Update market demand
        market_truth = {}
        for i in range(self.config["num_projects"]):
            sector_id = f"sector_{i}"
            # Random demand factor
            market_truth[sector_id] = max(0.1, np.random.normal(1.0, 0.2))

        self.ground_truth["market_demand"] = market_truth

        # Update wage rates
        wage_truth = {}
        for i in range(self.config["num_projects"]):
            sector_id = f"sector_{i}"
            # Random wage factor
            wage_truth[sector_id] = max(0.5, np.random.normal(1.0, 0.1))

        self.ground_truth["wage_rates"] = wage_truth

        # Update news generator with ground truth
        self.news_generator.set_ground_truth(self.ground_truth)

    def run_game(self, iterations):
        """Run the game for a specified number of iterations"""
        for _ in range(iterations):
            self.play_iteration()
            self.current_iteration += 1

    def play_iteration(self):
        """Play a single iteration of the game"""
        # Update ground truth at the start of each iteration
        self._update_ground_truth()

        # 1.1 Level 1 agents' turn
        self.level1_agent_turn()

        # 1.2 Level 2 agents' turn
        self.level2_agent_turn()

        # 2. Calculate profits based on strategies and allocations
        self.calculate_profits()

        # 3. Distribute profits to agents and coalitions
        self.distribute_profits()

        # 4. Agents analyze results and update strategies for next iteration
        self.update_agents()

        # Advance the news generator turn
        self.news_generator.advance_turn()

    def level1_agent_turn(self):
        """Level 1 agents generate news and announce distribution functions"""
        # Step 1: Generate news
        all_news = []
        for agent in self.level1_agents:
            # Generate news based on ground truth with potential deception
            news = agent.generate_strategic_news(
                self.news_generator, ground_truth=self.ground_truth
            )
            all_news.append(news)

        # Distribute news to level 2 agents and other level 1 agents
        agent_ids = [a.agent_id for a in self.level1_agents + self.level2_agents]
        news_distribution = self.news_generator.distribute_news(agent_ids)

        # Deliver news to each agent
        for agent_id, news_items in news_distribution.items():
            # Find the agent with this ID
            for agent in self.level1_agents + self.level2_agents:
                if agent.agent_id == agent_id:
                    agent.receive_news(news_items)

        # Step 2: Announce distribution functions/strategies
        distribution_strategies = {}
        for agent in self.level1_agents:
            # Get the agent's observation of the environment
            observation = self._get_agent_observation(agent)

            # Announce strategy based on current state and received news
            strategy = agent.announce_distribution(observation)
            distribution_strategies[agent.agent_id] = strategy

        # Store announced strategies for coalition formation
        self.distribution_strategies = distribution_strategies

    def level2_agent_turn(self):
        """Level 2 agents exchange information and form coalitions"""
        # Step 1: Exchange information and evaluate trust
        trust_scores = {}
        for agent in self.level2_agents:
            # Get received news
            agent_news = self.news_generator.get_news_for_agent(agent.agent_id)

            # Evaluate trustworthiness of each news item
            trust_evaluation = agent.evaluate_news_trustworthiness(agent_news)
            trust_scores[agent.agent_id] = trust_evaluation

        # Update trust network
        self.trust_network = trust_scores

        # Step 2: Form coalitions with trusted agents
        self.coalitions = []
        for agent in self.level2_agents:
            # Get observation of current environment
            observation = self._get_agent_observation(agent)

            # Get potential coalition members
            potential_members = agent.get_potential_members(
                [a.agent_id for a in self.level1_agents]
            )

            # Form coalition
            coalition_result = agent.form_coalition(potential_members, observation)

            if coalition_result["coalition"]:
                # Create coalition object
                coalition = Coalition([agent.agent_id] + coalition_result["coalition"])

                # Gather strategies from all coalition members
                member_strategies = {}
                for member_id in coalition.agents:
                    if member_id in self.distribution_strategies:
                        member_strategies[member_id] = self.distribution_strategies[
                            member_id
                        ]

                # Form the coalition with member strategies
                if coalition.form_coalition(member_strategies):
                    self.coalitions.append(coalition)

        # Assign projects to coalitions
        self._assign_projects_to_coalitions()

    def _assign_projects_to_coalitions(self):
        """Assign projects to coalitions based on their strategies"""
        # Reset project allocations
        for project_id in self.projects:
            self.projects[project_id]["allocated_resources"] = 0

        # For each coalition
        for coalition in self.coalitions:
            if not coalition.strategy:
                continue

            # Get Level 2 agents in this coalition (they control resources)
            level2_members = [
                agent_id for agent_id in coalition.agents if agent_id.startswith("L2_")
            ]

            # Calculate total resources from Level 2 agents
            total_resources = (
                len(level2_members) * 100
            )  # Assuming each L2 agent has 100 resources

            # Allocate resources according to coalition strategy
            allocation = coalition.allocate_resources(total_resources)

            # Assign resources to projects
            for project_id, resource_amount in allocation.items():
                if project_id in self.projects:
                    self.projects[project_id]["allocated_resources"] += resource_amount

            # Record projects for this coalition
            coalition.projects = list(allocation.keys())

    def calculate_profits(self):
        """Calculate profits for all agents and coalitions"""
        # Reset profits
        self.agent_profits = defaultdict(float)
        self.coalition_profits = defaultdict(float)

        # Calculate coalition profits based on large projects
        for coalition in self.coalitions:
            total_profit = 0
            for project_id in coalition.projects:
                if project_id in self.projects:
                    project = self.projects[project_id]
                    allocated = project["allocated_resources"]
                    optimal = project["optimal_resources"]

                    # Profit function: highest at optimal allocation, diminishing returns
                    efficiency = 1 - min(1, abs(allocated - optimal) / optimal)
                    profit = project["current_profitability"] * allocated * efficiency

                    # Coalition bonus
                    profit *= self.config["coalition_bonus"] * len(coalition.agents) / 5

                    total_profit += profit

            # Store coalition profit
            self.coalition_profits[tuple(coalition.agents)] = total_profit

        # Calculate Level 1 agent profits from small projects
        for agent in self.level1_agents:
            # Base profit from small projects
            small_project_profit = (
                sum(self.projects[p]["current_profitability"] for p in self.projects)
                * self.config["small_project_factor"]
                / len(self.level1_agents)
            )

            # Store agent profit
            self.agent_profits[agent.agent_id] = small_project_profit

        # Calculate Level 2 agent profits from allocations
        for agent in self.level2_agents:
            # Base profit is proportional to amount of resources allocated effectively
            base_profit = 0

            # Check projects this agent's resources contributed to
            for coalition in self.coalitions:
                if agent.agent_id in coalition.agents:
                    # Add profit from this coalition
                    coalition_size = len(coalition.agents)
                    agent_profit = (
                        self.coalition_profits[tuple(coalition.agents)] / coalition_size
                    )
                    base_profit += agent_profit

            # Store agent profit
            self.agent_profits[agent.agent_id] = base_profit

    def distribute_profits(self):
        """Distribute profits to agents and coalitions using Shapley/Owen value"""
        # Distribute coalition profits
        for coalition in self.coalitions:
            coalition_members = tuple(coalition.agents)
            if coalition_members in self.coalition_profits:
                total_profit = self.coalition_profits[coalition_members]

                # Calculate profit distribution
                distribution = coalition.calculate_profit_distribution(
                    total_profit, method=self.config["profit_distribution_method"]
                )

                # Update individual agent profits
                for agent_id, profit in distribution.items():
                    self.agent_profits[agent_id] += profit

        # Notify each agent of their profits
        for agent in self.level1_agents + self.level2_agents:
            profit = self.agent_profits[agent.agent_id]
            agent.update(
                reward=profit,
                next_observation=self._get_agent_observation(agent),
                done=False,
            )

    def update_agents(self):
        """Agents analyze results and update strategies for next iteration"""
        # Level 1 agents adjust conditions and prepare next news
        for agent in self.level1_agents:
            # Update based on profit received
            profit = self.agent_profits[agent.agent_id]

            # For now, simple update: if profit increased, strengthen current strategy
            if hasattr(agent, "last_profit"):
                if profit > agent.last_profit:
                    # Strategy is working - minor adjustments
                    pass
                else:
                    # Strategy not working well - more exploration
                    pass

            agent.last_profit = profit

        # Level 2 agents update trust vectors
        for agent in self.level2_agents:
            # Evaluate accuracy of received news now that we know true profits
            news_accuracy = {}
            for news_item in agent.received_news:
                if news_item["type"] == "project_profitability":
                    source_id = news_item["agent_id"]
                    reported_profits = news_item["content"]

                    # Calculate accuracy by comparing to actual profits
                    accuracy_scores = []
                    for project_id, reported_profit in reported_profits.items():
                        if project_id in self.projects:
                            actual_profit = self.projects[project_id][
                                "current_profitability"
                            ]
                            relative_error = abs(reported_profit - actual_profit) / max(
                                0.1, actual_profit
                            )
                            accuracy = max(0, 1 - min(1, relative_error))
                            accuracy_scores.append(accuracy)

                    if accuracy_scores:
                        news_accuracy[source_id] = sum(accuracy_scores) / len(
                            accuracy_scores
                        )

            # Update trust scores based on accuracy
            agent.update_trust_scores(news_accuracy)

    def _get_agent_observation(self, agent):
        """Construct observation for an agent based on their knowledge of the environment"""
        observation = {
            "iteration": self.current_iteration,
            "agent_id": agent.agent_id,
            "history": agent.history[-agent.max_history_length :],
        }

        # Add profit history if available
        if agent.agent_id in self.agent_profits:
            observation["profit"] = self.agent_profits[agent.agent_id]

        # Add coalition information if agent is in a coalition
        for coalition in self.coalitions:
            if agent.agent_id in coalition.agents:
                observation["coalition"] = {
                    "members": coalition.agents,
                    "strategy": coalition.strategy,
                    "projects": coalition.projects,
                }
                break

        # Add project information (visible to all agents)
        observation["projects"] = {
            project_id: {
                "id": project["id"],
                # Only basic info visible to all
                "allocated_resources": project["allocated_resources"],
            }
            for project_id, project in self.projects.items()
        }

        return observation

    def get_game_state(self):
        """Get the current state of the game for monitoring/visualization"""
        state = {
            "iteration": self.current_iteration,
            "projects": self.projects,
            "coalitions": [
                {
                    "members": c.agents,
                    "strategy": c.strategy,
                    "profit": self.coalition_profits.get(tuple(c.agents), 0),
                }
                for c in self.coalitions
            ],
            "agent_profits": dict(self.agent_profits),
            "trust_network": self.trust_network,
        }
        return state
