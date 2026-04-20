from typing import Any, Mapping
import torch
from torch import nn
import torch.nn.functional as F
from latentis.transform import Estimator


def sgd_conv_align_state(
    x: torch.Tensor, y: torch.Tensor, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None
) -> nn.Module:
    with torch.random.fork_rng():
        if random_seed is not None:
            torch.manual_seed(random_seed)
        with torch.enable_grad():

            translation = nn.Sequential(
                nn.Linear(x.size(1), 384),
                nn.GELU(),
                nn.Conv1d(128, 128, groups=128, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                nn.Linear(128, x.size(1)),
            )

            optimizer = torch.optim.Adam(translation.parameters(), lr=lr)

            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(x), y)
                loss.backward()
                optimizer.step()

            return dict(translation=translation)


class SGDConvAligner(Estimator):
    def __init__(
        self,
        num_steps: int,
        lr: float,
        random_seed: int,
    ):
        super().__init__(name="sgd_conv_aligner")
        self.num_steps = num_steps
        self.lr = lr
        self.random_seed = random_seed

        self.translation: nn.Linear = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        translation: nn.Module = sgd_conv_align_state(
            x=x, y=y, num_steps=self.num_steps, lr=self.lr, random_seed=self.random_seed
        )["translation"]
        self.translation = translation

        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.translation(x), y
