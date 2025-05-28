from environment.game import Game
from agents.level1_agent import Level1Agent
from agents.level2_agent import Level2Agent
from utils.embeddings import generate_embeddings
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configurations
    environment_config = load_config('configs/environment_config.yaml')
    training_config = load_config('configs/training_config.yaml')

    # Initialize agents
    level1_agents = [Level1Agent(i, generate_embeddings()) for i in range(environment_config['num_level1_agents'])]
    level2_agents = [Level2Agent(i) for i in range(environment_config['num_level2_agents'])]

    # Initialize game environment
    game = Game(level1_agents, level2_agents, environment_config)

    # Start the game loop
    game.run(training_config)

if __name__ == "__main__":
    main()