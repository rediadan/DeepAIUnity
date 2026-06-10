from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, int, int], seed: int = 42):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state
        self.dones[index] = 1.0 if done else 0.0
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
        )


def concat_batches(first: ReplayBatch, second: ReplayBatch) -> ReplayBatch:
    return ReplayBatch(
        states=np.concatenate([first.states, second.states], axis=0),
        actions=np.concatenate([first.actions, second.actions], axis=0),
        rewards=np.concatenate([first.rewards, second.rewards], axis=0),
        next_states=np.concatenate([first.next_states, second.next_states], axis=0),
        dones=np.concatenate([first.dones, second.dones], axis=0),
    )
