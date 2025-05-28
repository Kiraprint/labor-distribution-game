# Labor Distribution Game

This project simulates a labor distribution game using a two-level agent network. The game involves agents that generate news, form coalitions, and distribute labor resources across various infrastructure projects. The agents operate in two levels, with Level 1 agents generating news and distributing resources, while Level 2 agents manage interactions between hubs and make strategic decisions based on the received information.

## Project Structure

- **src/**: Contains the main source code for the simulation.
  - **environment/**: Implements the game environment, including the game loop and coalition management.
    - `game.py`: Manages the game state and agent interactions.
    - `coalition.py`: Handles coalition formation and profit distribution.
    - `news_generator.py`: Generates news for agents and hubs.
  - **agents/**: Defines the agent classes and their behaviors.
    - `agent_base.py`: Base class for all agents.
    - `level1_agent.py`: Implements Level 1 agent behavior.
    - `level2_agent.py`: Implements Level 2 agent behavior.
  - **models/**: Contains models for reinforcement learning.
    - `transformer.py`: Implements a transformer architecture for processing historical sequences.
    - `policy_network.py`: Defines the neural network for policy approximation.
    - `memory_buffer.py`: Manages experience storage for training.
  - **utils/**: Utility functions for various calculations.
    - `shapley.py`: Functions for calculating Shapley values.
    - `owen.py`: Functions for calculating Owen values.
    - `embeddings.py`: Functions for managing embeddings.
  - **training/**: Contains training algorithms and utilities.
    - `ppo_trainer.py`: Implements Proximal Policy Optimization.
    - `sac_trainer.py`: Implements Soft Actor-Critic.
    - `optimization.py`: Defines optimization functions.
  - **evaluation/**: Tools for evaluating agent performance.
    - `metrics.py`: Functions for calculating evaluation metrics.
    - `visualizer.py`: Functions for visualizing simulation results.
  - `main.py`: Entry point for running the simulation.

- **configs/**: Configuration files for the project.
  - `default.yaml`: Default settings.
  - `environment_config.yaml`: Environment-specific settings.
  - `training_config.yaml`: Training process settings.

- **notebooks/**: Jupyter notebooks for analysis.
  - `simulation_analysis.ipynb`: Analyzes simulation results.
  - `agent_behavior_analysis.ipynb`: Analyzes agent behaviors.

- **tests/**: Unit tests for the project.
  - `test_environment.py`: Tests for environment components.
  - `test_agents.py`: Tests for agent components.
  - `test_utils.py`: Tests for utility functions.

- **requirements.txt**: Lists project dependencies.

- **setup.py**: Used for packaging and installation.

## Installation

To install the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd labor-distribution-game
pip install -r requirements.txt
```

## Usage

To run the simulation, execute the following command:

```bash
python src/main.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.