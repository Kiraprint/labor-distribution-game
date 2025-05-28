import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=F.relu):
        """
        Policy network for decision making

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (action space)
            activation: Activation function to use
        """
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

        # Save initialization args for target network creation
        self.init_args = (input_dim, hidden_dims, output_dim)
        self.init_kwargs = {"activation": activation}

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer for policy logits
        self.policy_head = nn.Linear(hidden_dims[-1], output_dim)

        # Output layer for value function
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        """Forward pass for policy logits"""
        # Process through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Get policy logits
        policy_logits = self.policy_head(x)

        return policy_logits

    def value(self, x):
        """Forward pass for value function"""
        # Process through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Get value estimate
        value = self.value_head(x)

        return value

    def get_action(self, state, deterministic=False):
        """Get action from policy given state"""
        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                # Choose most likely action
                action = torch.argmax(probs, dim=-1)
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

        return action.item()

    def get_log_probs(self, states, actions):
        """Get log probabilities of actions given states"""
        logits = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.long())

        return log_probs

    def get_all_parameters(self):
        """Get all parameters for optimization"""
        return self.parameters()
