# Labor Distribution Game Simulation

## Overview

This simulation models strategic interactions between autonomous agents in a labor distribution economy. The environment features two types of agents (Level1 and Level2) that must navigate information sharing, coalition formation, and resource allocation to maximize profits.

## Model Architecture

### Core Components

- **Game Environment**: Central simulation engine that manages interactions, tracks resources, and calculates profits
- **Agents**: Two hierarchical types with different capabilities and information access
- **Coalitions**: Dynamic groupings of agents that collaborate on projects
- **News System**: Information economy where agents can share truthful or deceptive information
- **Projects**: Infrastructure objects that generate profits based on resource allocation

### Neural Network Models

1. **HistoryTransformer**
   - Architecture: Transformer encoder with positional encoding
   - Purpose: Processes sequential history for context-aware decision making
   - Parameters: 
     - Embedding dimension (default: 64)
     - Number of attention heads (default: 4)
     - Number of transformer layers (default: 2)
     - Dropout (default: 0.1)

2. **PolicyNetwork**
   - Architecture: Multi-layer perceptron with customizable hidden layers
   - Purpose: Maps state representations to action probabilities
   - Components:
     - Shared feature extraction layers
     - Policy head (action logits)
     - Value head (state value estimation)

## Agent Types

### Level1 Agents (Labor Force Controllers)

Level1 agents represent labor force controllers who can distribute workers across infrastructure projects and generate news for other agents.

#### Capabilities:

- Generate news (truthful or deceptive) about market conditions
- Announce labor distribution strategies
- Participate in coalitions formed by Level2 agents
- Earn profits from small projects autonomously

#### Available Actions:

1. **News Generation**:
   - Decide truthfulness level (continuous value between 0-1)
   - Select news type (market demand, wage rates, resource availability, project profitability)
   - Determine content parameters (specific values for different sectors/projects)
   
2. **Labor Distribution**:
   - Specify allocation ratios across different infrastructure projects
   - Adjust wages and work conditions
   - Form provisional alliances with other Level1 agents

#### Strategies:

1. **Honest Broker**: Generate mostly truthful news to build trust, form stable coalitions
2. **Strategic Deception**: Manipulate information to direct resources toward preferred projects
3. **Coalition Specialist**: Focus on maintaining relationships with Level2 agents
4. **Independent Operator**: Maximize profits from small projects while minimizing coalition commitments

### Level2 Agents (Resource Allocators)

Level2 agents represent resource allocators who evaluate news, form coalitions, and distribute resources across infrastructure projects.

#### Capabilities:

- Evaluate trustworthiness of news from Level1 agents
- Form coalitions with trusted Level1 agents
- Allocate resources across infrastructure projects
- Distribute profits using game-theoretic principles (Shapley/Owen value)

#### Available Actions:

1. **Trust Evaluation**:
   - Assign trust scores to Level1 agents based on news reliability
   - Filter information based on trust thresholds
   - Track and update trust over time based on observed outcomes
   
2. **Coalition Formation**:
   - Select coalition members from potential Level1 agents
   - Determine coalition structure and hierarchy
   - Set resource contribution levels
   
3. **Resource Allocation**:
   - Distribute resources across projects based on expected profitability
   - Balance risk and reward across portfolio of projects
   - Adjust allocation based on coalition strategy

#### Strategies:

1. **Trust-First**: Heavily weight trust scores in coalition formation decisions
2. **Diversification**: Spread resources across many projects to reduce risk
3. **Specialization**: Concentrate resources on few high-potential projects
4. **Adaptive Trust**: Rapidly adjust trust scores based on observed outcomes
5. **Conservative**: Maintain stable coalitions with proven trustworthy partners

## Game Mechanics

### Turn Structure

1. **Level1 Agent Phase**:
   - Generate and distribute news
   - Announce labor distribution strategies

2. **Level2 Agent Phase**:
   - Evaluate news trustworthiness
   - Form coalitions with trusted Level1 agents
   - Allocate resources across projects

3. **Profit Generation**:
   - Projects generate profits based on allocated resources
   - Efficiency bonuses for optimal resource allocation
   - Coalition bonuses for effective collaboration

4. **Profit Distribution**:
   - Coalition profits distributed using Shapley/Owen value
   - Level1 agents receive baseline profits from small projects
   - Level2 agents receive profits based on coalition performance

5. **Agent Updates**:
   - Agents analyze results and update strategies
   - Trust scores adjusted based on observed outcomes
   - Neural networks updated via reinforcement learning

### Economic Principles

- **Coalition Efficiency**: Larger coalitions receive bonus multipliers to encourage cooperation
- **Resource Optimization**: Projects have optimal resource levels, with diminishing returns
- **Information Value**: Truthful information leads to better resource allocation and higher profits
- **Trust Dynamics**: Trust is built over time based on news accuracy

## Reinforcement Learning

The simulation uses two primary reinforcement learning algorithms:

1. **Proximal Policy Optimization (PPO)**
   - Clip-constrained policy optimization
   - Generalized Advantage Estimation (GAE)
   - Entropy bonus for exploration
   - Shared networks for policy and value functions

2. **Soft Actor-Critic (SAC)**
   - Maximum entropy reinforcement learning
   - Twin Q-networks for robust value estimation
   - Adaptive temperature parameter
   - Experience replay for sample efficiency

## Running the Simulation

```python
from environment.game import Game
from utils.logger import setup_logger

# Initialize the environment
logger = setup_logger()
game = Game()

# Run for 1000 iterations
for i in range(1000):
    game.play_iteration()
    
    # Optionally train the agents
    if (i + 1) % 10 == 0:
        game.train_agents()
        
    # Print metrics
    if (i + 1) % 50 == 0:
        metrics = game.get_metrics()
        logger.info(f"Iteration {i+1}: Avg Level2 Reward: {metrics['level2_rewards'][-1]:.2f}")
```

## Advanced Configuration

The simulation allows customization of various parameters:

```python
config = {
    # Environment parameters
    "num_level1_agents": 5,
    "num_level2_agents": 3,
    "num_projects": 5,
    "coalition_bonus": 1.5,
    "profit_distribution_method": "shapley",  # or "owen"
    
    # Agent parameters
    "embedding_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "trust_threshold": 0.5,
    
    # Training parameters
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "batch_size": 64
}

game = Game(config)
```

## Implementation Notes

- Memory-optimized for running on 8GB VRAM GPUs
- Supports CPU and CUDA execution with automatic device detection
- Uses efficient transformer implementation for processing sequential data
- Includes comprehensive logging for tracking agent behaviors

---

For more information, see the individual module documentation and example notebooks.