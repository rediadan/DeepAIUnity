from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.replay_buffer import ReplayBatch


class ExpertReplayBuffer:
    def __init__(self, dataset_path: str | Path, seed: int = 42):
        data = np.load(dataset_path)
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rewards = data["rewards"].astype(np.float32)
        self.next_states = data["next_states"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def sample(self, batch_size: int) -> ReplayBatch:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty expert replay buffer.")
        indices = self.rng.integers(0, len(self), size=int(batch_size))
        return ReplayBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
        )
