import math
from abc import ABC

class IterativePolicy(ABC):
    def __init__(self, max_iterations: int = 1000, policy: str = 'constant', **policy_kwargs) -> None:
        self.max_iterations = max_iterations
        self.policy = policy
        self.train_batch_size = policy_kwargs.get("train_batch_size", 256)
        self._build_policy(**policy_kwargs)

    def _build_policy(self, **policy_kwargs):
        if self.policy == "constant":
            assert "frequency" in policy_kwargs
            frequency = policy_kwargs["frequency"]
            num_iterations = math.ceil(self.max_iterations / frequency)
            self.num_samples = [0]
            if num_iterations == 1:
                self.num_samples.append(self.max_iterations * self.train_batch_size)
            else:
                self.num_samples.extend([frequency * self.train_batch_size] * num_iterations)
        elif self.policy == "exponential":
            assert "initial_size" in policy_kwargs
            assert "growth_rate" in policy_kwargs
            initial_size = policy_kwargs["initial_size"]
            growth_rate = policy_kwargs["growth_rate"]
            count_iterations = initial_size
            num_iteration = initial_size
            self.num_samples = [0, initial_size * self.train_batch_size]
            while count_iterations < self.max_iterations:
                num_iteration *= growth_rate
                self.num_samples.append(int(num_iteration + 0.5) * self.train_batch_size)
                count_iterations += int(num_iteration + 0.5)
        elif self.policy == "linear":
            assert "slope" in policy_kwargs
            slope = policy_kwargs["slope"]
            count_iterations = 0
            num_iteration = 0
            self.num_samples = [0]
            while count_iterations < self.max_iterations:
                num_iteration += slope
                self.num_samples.append(int(num_iteration + 0.5) * self.train_batch_size)
                count_iterations += int(num_iteration + 0.5)
        elif self.policy == "polynomial":
            pass
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        return 