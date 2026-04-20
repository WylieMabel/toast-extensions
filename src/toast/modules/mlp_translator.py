from typing import Any, Mapping
import torch
from torch import nn
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from latentis.transform import Estimator


def sgd_mlp_align_state(
    x: torch.Tensor, y: torch.Tensor, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None
) -> nn.Module:
    with torch.random.fork_rng():
        seed_everything(random_seed)
        with torch.enable_grad():

            translation = nn.Sequential(
                nn.Linear(x.size(1), y.size(1) // 2, device=x.device, dtype=x.dtype, bias=True),
                nn.GELU(),
                nn.Linear(y.size(1) // 2, y.size(1), device=x.device, dtype=x.dtype, bias=True),
            )

            optimizer = torch.optim.Adam(translation.parameters(), lr=lr)

            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(x), y)
                loss.backward()
                optimizer.step()

            return dict(translation=translation)


class SGDMLPAligner(Estimator):
    def __init__(
        self,
        num_steps: int,
        lr: float,
        random_seed: int,
    ):
        super().__init__(name="sgd_mlp_aligner")
        self.num_steps = num_steps
        self.lr = lr
        self.random_seed = random_seed

        self.translation: nn.Linear = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        translation: nn.Module = sgd_mlp_align_state(
            x=x, y=y, num_steps=self.num_steps, lr=self.lr, random_seed=self.random_seed
        )["translation"]
        self.translation = translation

        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.translation(x), y
