class LearningRate:
    def __init__(
            self,
            initial_lr: float,
            decay_rate: float,
            decay_steps: int,
    ):
        self.lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self._calls = 0

    def __call__(self) -> float:
        if self.decay_rate == 0:
            return self.lr
        if self.decay_steps <= 0:
            return self.lr
        self._calls += 1
        _lr = self.lr
        if self._calls % self.decay_steps == 0:
            self.lr *= self.decay_rate
            self._calls = 0
        return _lr
