import random
from collections import deque

import torch


class MemoryBuffer:
    def __init__(self, capacity=10000, device="cpu"):
        """
        Memory buffer for storing agent experiences

        Args:
            capacity: Maximum number of experiences to store
            device: Device to store tensors on
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer"""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        # Randomly sample batch_size experiences
        batch = random.sample(self.buffer, batch_size)

        # Separate the tuple of experiences into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to torch tensors if not already
        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = self._to_tensor(rewards)
        next_states = self._to_tensor(next_states)
        dones = self._to_tensor(dones)

        return [(states, actions, rewards, next_states, dones)]

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

    def size(self):
        """Get the current size of the buffer"""
        return len(self.buffer)

    def _to_tensor(self, data):
        """Convert data to tensor if it's not already"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
