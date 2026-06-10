from __future__ import annotations


class LinearEpsilonScheduler:
    def __init__(self, start: float = 1.0, end: float = 0.05, decay_steps: int = 50000):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        progress = min(max(step, 0) / self.decay_steps, 1.0)
        return self.start + (self.end - self.start) * progress
