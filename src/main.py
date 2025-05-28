import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from environment.game import Game
from models.memory_buffer import MemoryBuffer
from training.ppo_trainer import PPOTrainer
from utils.embeddings import generate_embeddings


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_simulation(config=None):
    """Run the labor distribution game simulation with learning agents"""
    print("Initializing game environment...")
    game = Game(config)

    # Create memory buffers for each agent type
    level1_buffer = MemoryBuffer(capacity=10000)
    level2_buffer = MemoryBuffer(capacity=10000)

    # Create trainers for each agent type
    level1_trainers = {}
    level2_trainers = {}

    for agent in game.level1_agents:
        level1_trainers[agent.agent_id] = PPOTrainer(
            policy_network=agent.news_policy,
            value_network=agent.value_network,
            memory_buffer=level1_buffer,
        )

    for agent in game.level2_agents:
        level2_trainers[agent.agent_id] = PPOTrainer(
            policy_network=agent.coalition_policy,
            value_network=agent.value_network,
            memory_buffer=level2_buffer,
        )

    # Training metrics
    metrics = {
        "level1_rewards": [],
        "level2_rewards": [],
        "coalition_sizes": [],
        "trust_scores": [],
    }

    # Run simulation for specified number of iterations
    total_iterations = 1000
    train_frequency = 10  # Train every 10 iterations

    print(f"Starting simulation for {total_iterations} iterations...")
    for i in range(total_iterations):
        # Play one iteration of the game
        game.play_iteration()

        # Collect metrics
        level1_rewards = [game.agent_profits[a.agent_id] for a in game.level1_agents]
        level2_rewards = [game.agent_profits[a.agent_id] for a in game.level2_agents]
        coalition_sizes = [len(c.agents) for c in game.coalitions]

        metrics["level1_rewards"].append(np.mean(level1_rewards))
        metrics["level2_rewards"].append(np.mean(level2_rewards))
        metrics["coalition_sizes"].append(
            np.mean(coalition_sizes) if coalition_sizes else 0
        )

        # Average trust scores
        all_trust = []
        for agent in game.level2_agents:
            all_trust.extend(list(agent.trust_scores.values()))
        metrics["trust_scores"].append(np.mean(all_trust))

        # Training step
        if (i + 1) % train_frequency == 0:
            print(f"Iteration {i + 1}/{total_iterations} - Training agents...")

            # Train Level 1 agents
            if level1_buffer.size() > 64:
                for agent_id, trainer in level1_trainers.items():
                    train_metrics = trainer.train()
                    print(
                        f"  Level1 Agent {agent_id}: Loss={train_metrics.get('policy_loss', 'N/A'):.4f}"
                    )

            # Train Level 2 agents
            if level2_buffer.size() > 64:
                for agent_id, trainer in level2_trainers.items():
                    train_metrics = trainer.train()
                    print(
                        f"  Level2 Agent {agent_id}: Loss={train_metrics.get('policy_loss', 'N/A'):.4f}"
                    )

        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Iteration {i + 1}/{total_iterations}")
            print(f"  Avg Level1 Reward: {metrics['level1_rewards'][-1]:.2f}")
            print(f"  Avg Level2 Reward: {metrics['level2_rewards'][-1]:.2f}")
            print(f"  Avg Coalition Size: {metrics['coalition_sizes'][-1]:.2f}")
            print(f"  Avg Trust Score: {metrics['trust_scores'][-1]:.2f}")

    # Plot results
    plot_metrics(metrics)

    return game, metrics


def plot_metrics(metrics):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics["level1_rewards"], label="Level 1 Agents")
    plt.plot(metrics["level2_rewards"], label="Level 2 Agents")
    plt.title("Average Rewards")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()

    # Plot coalition sizes
    plt.subplot(2, 2, 2)
    plt.plot(metrics["coalition_sizes"])
    plt.title("Average Coalition Size")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Agents")

    # Plot trust scores
    plt.subplot(2, 2, 3)
    plt.plot(metrics["trust_scores"])
    plt.title("Average Trust Score")
    plt.xlabel("Iteration")
    plt.ylabel("Trust Score")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()


def main():
    # Load configurations
    environment_config = load_config("configs/environment_config.yaml")
    training_config = load_config("configs/training_config.yaml")

    # Initialize agents
    level1_agents = [
        Level1Agent(i, generate_embeddings())
        for i in range(environment_config["num_level1_agents"])
    ]
    level2_agents = [
        Level2Agent(i) for i in range(environment_config["num_level2_agents"])
    ]

    # Initialize game environment
    game = Game(level1_agents, level2_agents, environment_config)

    # Start the game loop
    game.run(training_config)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Optional: Configure CUDA for memory efficiency
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    # Run the simulation
    game, metrics = run_simulation()

    print("Simulation complete!")
