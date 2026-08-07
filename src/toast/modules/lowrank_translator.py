"""Low-rank translators, selectable from a config CSV as ``skip_translator=<method>_<rank>``.

Three methods are registered in dictionaries.NAME2TRANSLATORS, over a grid of ranks:

    lowrank_<r>   SVD truncation of the least-squares solution
    rrr_<r>       reduced-rank regression (the closed-form optimum)
    lora_<r>      A and B fit directly by gradient descent

So a sweep is just rows in the config CSV -- ``skip_translator=rrr_32``, ``lora_8``, and so on
-- with no separate driver script. The results CSV records the method and rank in their own
columns so the sweep can be grouped without parsing names.

The default "linear" translator keeps a full ``D x D`` matrix (590k params for ViT-B at D=768,
1M for ViT-L at D=1024). All three methods here constrain it to rank ``r`` and store it
factored as ``A (D x r) @ B (r x D)``, dropping the cost to ``2 D r``.

WHAT IS BEING FIT
    Only the translator. SkipModel.fit_translators hands an aligner two frozen activation
    matrices -- hidden states at layer ``skip_from`` (X) and at ``skip_to`` (Y), collected once
    by extract_representations under no_grad. Fitting touches nothing else: no gradient reaches
    the transformer, no images are re-encoded, and there is one translator per skip span. In
    mode 1 that single map is shared across all token positions.

    Shapes: with ``--samples_to_extract=250`` images and N tokens per image, X and Y are each
    ``[250 * N, D]`` -- ~64k rows (~197MB fp32) for dinov2-base at 224px, and ~343k rows
    (~1.05GB fp32) for rad-dino at 518px where N=1370. The closed-form methods touch that data
    twice (one lstsq, one eigendecomposition); lora_* iterates over it, hence minibatching.

HOW THE THREE DIFFER -- all fit on the same (X, Y); the difference is when rank is imposed:

``lowrank_<r>`` -- fit full W, then truncate (reduction ignores the data)
    Keep W's top-``r`` singular directions. By Eckart-Young this is the best rank-``r``
    approximation *of the matrix W* in Frobenius norm. X plays no part once W exists, so rank
    is spent wherever W is large -- even along directions the data barely occupies, and
    transformer activations are anisotropic enough that such directions exist.

``rrr_<r>`` -- fit full W, then project (reduction uses the data)
    Solve ``min ||X M - Y||`` subject to ``rank(M) <= r``. Closed form (Izenman 1975): with
    ``V_r`` the top-``r`` right singular vectors of the fitted values ``XW``, the optimum is
    ``M = W V_r V_r^T``. Since the projection is built from XW, this is provably identical to
    imposing the rank constraint during fitting. It is the optimum -- nothing beats it here.

``lora_<r>`` -- fit A and B directly, never forming a full W
    The formulation the thesis todo names. On plain squared error it solves the *same* problem
    rrr_<r> solves in closed form, so at best it matches it, more slowly and with
    hyperparameters. Its value is as a check (landing on the RRR error validates both; beating
    it means something is wrong) and as the extension point, since unlike the closed forms it
    can absorb a nonlinearity, weight decay, or streaming fits.

The rank constraint is doing *compression*, not regularisation: ~64k rows leave a 768x768 map
heavily overdetermined to begin with.
"""

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from latentis.transform import Estimator


def _lstsq_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The unconstrained least-squares map W with X @ W ~= Y.

    Matches what the "linear" translator does via latentis' lstsq_align_state, so the
    full-rank case of the closed-form methods reduces exactly to it.
    """
    return torch.linalg.lstsq(x, y).solution


def _split_singular_values(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, rank: int):
    """Factor a truncated SVD into (A, B), splitting the singular values evenly between them.

    Putting sqrt(S) on each side keeps the two factors on a similar scale, which matters once
    they are stored in fp32 buffers and multiplied at inference.
    """
    r = min(rank, s.numel())
    sq = s[:r].sqrt()
    a = u[:, :r] * sq                 # (D, r)  -- scales column j by sq[j]
    b = vh[:r, :] * sq.unsqueeze(1)   # (r, D)  -- scales row j by sq[j]
    return a, b


def _top_right_singular_vectors(m: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-``rank`` right singular vectors of ``m``, as a (D, r) matrix.

    Taking the SVD of ``m`` directly would allocate an (n, D) U factor -- ~1GB for rad-dino --
    and discard it immediately. The Gram matrix yields the same right singular vectors from a
    (D, D) eigendecomposition. Squaring the condition number is fine here because only the
    dominant subspace is needed, not accurate small eigenvalues.
    """
    gram = m.T @ m                            # (D, D), symmetric PSD
    _, eigvecs = torch.linalg.eigh(gram)      # ascending eigenvalues
    r = min(rank, eigvecs.shape[1])
    return eigvecs[:, -r:]                    # (D, r)


class _FactoredAligner(Estimator):
    """Shared machinery: hold a factored map as buffers A, B and apply it as (x @ A) @ B.

    A and B are registered as buffers named exactly "A" and "B" because that is the contract
    save_translator/load_translator rely on -- load_translator restores state by calling
    register_buffer(<state_dict key>, value), so the buffer names *are* the serialisation
    format. Keeping this class flat (no submodules) keeps those keys unqualified.
    """

    #: Set by subclasses; recorded in the results CSV alongside the rank.
    method: str = "unknown"

    def __init__(self, name: str, rank: int):
        super().__init__(name=name)
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.rank = rank
        # Declared up front so state_dict() has a stable set of keys before fit().
        self.register_buffer("A", None)
        self.register_buffer("B", None)

    def _set_factors(self, a: torch.Tensor, b: torch.Tensor):
        self.register_buffer("A", a.detach().contiguous())
        self.register_buffer("B", b.detach().contiguous())
        return self

    def transform(self, x: torch.Tensor, y: torch.Tensor = None):
        if self.A is None or self.B is None:
            raise RuntimeError(f"{self.name} used before fit() (A/B are unset).")
        # Associativity matters: (x @ A) @ B costs 2*n*D*r, whereas x @ (A @ B) would first
        # materialise the D x D product and throw away the point of the factorisation.
        return (x.to(self.A.dtype) @ self.A) @ self.B, y

    @property
    def num_parameters(self) -> int:
        return 0 if self.A is None else self.A.numel() + self.B.numel()


class LowRankAligner(_FactoredAligner):
    """``lowrank_<r>``: rank-r truncation of the least-squares solution."""

    method = "lowrank"

    def __init__(self, rank: int):
        super().__init__(name=f"lowrank_{rank}", rank=rank)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        w = _lstsq_solution(x, y)
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        return self._set_factors(*_split_singular_values(u, s, vh, self.rank))


class ReducedRankAligner(_FactoredAligner):
    """``rrr_<r>``: reduced-rank regression, the best rank-r map for predicting Y from X."""

    method = "rrr"

    def __init__(self, rank: int):
        super().__init__(name=f"rrr_{rank}", rank=rank)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        w = _lstsq_solution(x, y)
        v_r = _top_right_singular_vectors(x @ w, self.rank)
        # M = W V_r V_r^T, factored as A = W V_r, B = V_r^T.
        return self._set_factors(w @ v_r, v_r.T)


class FactoredSGDAligner(_FactoredAligner):
    """``lora_<r>``: fit A (D x r) and B (r x D) directly by gradient descent.

    The full D x D map is never formed -- only the two factors are optimised, against the same
    frozen (X, Y) pair the closed forms use.

    ``init="random"`` tests the factored formulation on its own terms. ``init="rrr"`` warm
    starts from the closed-form optimum, useful for confirming the optimiser holds a good
    solution rather than drifting off it, but not an independent result. Standard LoRA
    initialises B to zero because it *adds* to a frozen base; here the map *replaces* rather
    than adds, so a zero init would start from the zero map -- both factors get scaled random
    values instead.

    ``batch_size`` bounds the autograd graph. Full-batch on rad-dino would hold a ~343k x 768
    intermediate and its graph for all 300 steps; the default minibatches. Pass
    ``batch_size=None`` for full-batch on small models.
    """

    method = "lora"

    def __init__(
        self,
        rank: int,
        num_steps: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        random_seed: int = 0,
        init: str = "random",
        batch_size: Optional[int] = 16384,
    ):
        super().__init__(name=f"lora_{rank}", rank=rank)
        self.num_steps = num_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        if init not in ("random", "rrr"):
            raise ValueError(f"init must be 'random' or 'rrr', got {init!r}")
        self.init = init
        self.batch_size = batch_size

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        n, d_in = x.shape
        d_out = y.shape[-1]
        r = min(self.rank, d_in, d_out)

        with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
            torch.manual_seed(self.random_seed)

            if self.init == "rrr":
                w = _lstsq_solution(x, y)
                v_r = _top_right_singular_vectors(x @ w, r)
                a, b = (w @ v_r).clone(), v_r.T.clone()
            else:
                # Scale each factor by 1/sqrt(its fan-in) so the initial product has a sane
                # gain instead of growing with r.
                a = torch.randn(d_in, r, device=x.device, dtype=x.dtype) / (d_in ** 0.5)
                b = torch.randn(r, d_out, device=x.device, dtype=x.dtype) / (r ** 0.5)

            a.requires_grad_(True)
            b.requires_grad_(True)
            optimizer = torch.optim.Adam([a, b], lr=self.lr, weight_decay=self.weight_decay)

            with torch.enable_grad():
                for _ in range(self.num_steps):
                    if self.batch_size is None or self.batch_size >= n:
                        xb, yb = x, y
                    else:
                        idx = torch.randint(0, n, (self.batch_size,), device=x.device)
                        xb, yb = x[idx], y[idx]
                    optimizer.zero_grad()
                    F.mse_loss((xb @ a) @ b, yb).backward()
                    optimizer.step()

        return self._set_factors(a, b)


# --------------------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------------------- #

#: Ranks exposed as translator names. Powers of two up to 256 -- past that the saving over a
#: full 768x768 map stops being interesting (2*768*256 = 393k vs 590k).
DEFAULT_RANKS = (8, 16, 32, 64, 128, 256)

METHOD2CLASS = {
    "lowrank": LowRankAligner,
    "rrr": ReducedRankAligner,
    "lora": FactoredSGDAligner,
}


def parse_translator_name(name: str):
    """Split ``"rrr_32"`` into ``("rrr", 32)``; return ``(None, None)`` if not a low-rank name.

    Used by the results writer to report method and rank as their own columns, and by the
    parameter accounting to cost a translator as 2*D*r instead of D*D.
    """
    if not isinstance(name, str) or "_" not in name:
        return None, None
    method, _, rank = name.rpartition("_")
    if method not in METHOD2CLASS or not rank.isdigit():
        return None, None
    return method, int(rank)


def build_translator_factories(ranks=DEFAULT_RANKS):
    """``{"lowrank_8": factory, "rrr_8": factory, "lora_8": factory, ...}``.

    Each value is a zero-arg callable returning a fresh aligner, matching what
    NAME2TRANSLATORS entries are expected to be. The default argument binding on ``cls`` and
    ``r`` is deliberate -- a bare closure would capture the loop variables by reference and
    every factory would build the last rank.
    """
    return {
        f"{method}_{r}": (lambda cls=cls, r=r: cls(rank=r))
        for method, cls in METHOD2CLASS.items()
        for r in ranks
    }
