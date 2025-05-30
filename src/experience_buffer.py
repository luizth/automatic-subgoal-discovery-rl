import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Experience:
    state: any
    action: any
    next_state: any


class ExperienceBuffer:

    def __init__(self, max_size=None):
        self.buffer: List[Experience] = []
        self.max_size = max_size

    @property
    def size(self):
        return len(self.buffer)

    def add(self, experience):
        if self.max_size is not None and len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(Experience(*experience))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("Batch size cannot be larger than the buffer size.")
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self):
        self.buffer.clear()

    def copy(self):
        new_buffer = ExperienceBuffer(max_size=self.max_size)
        new_buffer.buffer = self.buffer.copy()
        return new_buffer

    def transition_matrix(self):
        if not self.buffer:
            return None
        states = [exp.state for exp in self.buffer]
        actions = [exp.action for exp in self.buffer]
        next_states = [exp.next_state for exp in self.buffer]
        return np.array(states), np.array(actions), np.array(next_states)

    def state_action_pairs(self):
        if not self.buffer:
            return None
        return [(exp.state, exp.action) for exp in self.buffer]

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.buffer):
            raise IndexError("Index out of bounds.")
        return self.buffer[index]
