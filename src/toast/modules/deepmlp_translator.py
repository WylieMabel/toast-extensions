from typing import Any, Mapping
import torch
from torch import nn
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from latentis.transform import Estimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearningBlock(nn.Module):
    def __init__(self, num_features: int, dropout_p: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

        self.ff = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = self.norm1(x)
        x_transformed = self.ff(x_normalized)
        return self.norm2(x_transformed + x_normalized)


def sgd_deepmlp_align_state(
    x: torch.Tensor, y: torch.Tensor, num_steps: int = 300, lr: float = 1e-3, random_seed: int = None
) -> nn.Module:
    with torch.random.fork_rng():
        seed_everything(random_seed)
        with torch.enable_grad():

            translation = LearningBlock(num_features=x.size(1), dropout_p=0.2).to(device).double()

            optimizer = torch.optim.Adam(translation.parameters(), lr=lr)

            for _ in range(num_steps):
                optimizer.zero_grad()
                loss = F.mse_loss(translation(x), y)
                loss.backward()
                optimizer.step()

            return dict(translation=translation)


class SGDDeepMLPAligner(Estimator):
    def __init__(
        self,
        num_steps: int,
        lr: float,
        random_seed: int,
    ):
        super().__init__(name="sgd_deepMLP_aligner")
        self.num_steps = num_steps
        self.lr = lr
        self.random_seed = random_seed

        self.translation: nn.Linear = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        translation: nn.Module = sgd_deepmlp_align_state(
            x=x, y=y, num_steps=self.num_steps, lr=self.lr, random_seed=self.random_seed
        )["translation"]
        self.translation = translation

        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.translation(x), y
