"""Smoke-test the low-rank translators against the real latentis classes.

Run this once on the cluster before queueing the sweep (run_preflight.sh does it for you). It
checks the parts that cannot be tested without latentis installed: that the aligners satisfy
the Estimator interface, that the Translator wrapper fits and transforms them, that they
survive save/load, and that the parameter accounting matches the stored factor sizes.

    python -u src/toast/scripts/verify_lowrank.py

Exits non-zero on the first failure, with the check named.

DEBUGGING A SEGFAULT
    A segfault is a crash inside a native library, so there is no Python traceback and any
    buffered stdout is lost -- the script appears to die having printed nothing. Two things
    guard against that: faulthandler prints the Python stack at the moment of the crash, and
    every line flushes, so the last line printed IS the step that died.

    Section 0 below runs the LAPACK calls the translators depend on (lstsq, svd, eigh) in
    isolation, before any toast or latentis code. A crash there is environmental, not a bug in
    the maths -- almost always an OpenMP/BLAS clash from torch, numpy and scipy linking
    different threading runtimes (libgomp vs libiomp5), which faults on the first threaded
    LAPACK call. The fix is to serialise the threading:

        OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU python -u <this script>

    run_preflight.sh already sets both.
"""

import faulthandler
import os
import sys
import tempfile
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, "src")

import torch  # noqa: E402

D = 64
checks_run = 0


def check(name, condition, detail=""):
    global checks_run
    checks_run += 1
    if condition:
        print(f"  ok   {name}", flush=True)
    else:
        print(f"  FAIL {name}  {detail}", flush=True)
        raise SystemExit(1)


def section0_native():
    """LAPACK calls the translators rely on, in isolation. A crash here is environmental."""
    print("\n0. native/LAPACK sanity (crash here = threading-runtime clash, not a code bug)",
          flush=True)
    for var in ("OMP_NUM_THREADS", "MKL_THREADING_LAYER", "OPENBLAS_NUM_THREADS"):
        print(f"       {var}={os.environ.get(var, '<unset>')}", flush=True)
    print(f"       torch {torch.__version__}, threads={torch.get_num_threads()}", flush=True)

    torch.manual_seed(0)
    x = torch.randn(2000, D) * torch.logspace(0, -2, D)
    y = x @ (torch.randn(D, D) / D**0.5) + 0.05 * torch.randn(2000, D)

    print("       calling matmul ...", flush=True)
    _ = x.T @ x
    check("matmul", True)

    print("       calling torch.linalg.lstsq ...", flush=True)
    w = torch.linalg.lstsq(x, y).solution
    check("lstsq (used by all three methods)", w.shape == (D, D))

    print("       calling torch.linalg.svd ...", flush=True)
    _, s, _ = torch.linalg.svd(w, full_matrices=False)
    check("svd (used by lowrank_*)", s.numel() == D)

    print("       calling torch.linalg.eigh ...", flush=True)
    yh = x @ w
    evals, _ = torch.linalg.eigh(yh.T @ yh)
    check("eigh (used by rrr_*)", evals.numel() == D)

    return x, y


def main():
    # Native ops first: if these fault, nothing below is worth investigating.
    x, y = section0_native()

    # Imports come after section 0 so a crash inside latentis or toast is distinguishable
    # from a crash in bare LAPACK, and each dependency is pulled in on its own line so a
    # segfault names one module rather than a whole import tree.
    #
    # dictionaries.py previously died here, on `import pytorch_lightning` -- which the mlp
    # translators pulled in for seed_everything alone, and which segfaults against this
    # venv's torch 2.11 + numpy 2.x. That dependency is gone (see utils.seed_everything).
    # The per-line granularity stays: a crash while loading a compiled extension defeats
    # faulthandler, so the last line printed is the only evidence of where it died.
    print("\n0b. imports (one dependency per line -- the last line printed is the culprit)",
          flush=True)
    probes = [
        ("latentis.transform",
         "from latentis.transform import Estimator"),
        ("toast.utils.utils  <- local seed_everything, replaces pytorch_lightning",
         "from toast.utils.utils import seed_everything"),
        ("latentis.transform.base",
         "from latentis.transform.base import StandardScaling"),
        ("latentis.transform.translate.functional",
         "from latentis.transform.translate.functional import lstsq_align_state"),
        ("latentis.transform.translate.aligner",
         "from latentis.transform.translate.aligner import Translator, MatrixAligner"),
        ("toast.modules.mlp_translator",
         "from toast.modules.mlp_translator import SGDMLPAligner"),
        ("toast.modules.deepmlp_translator",
         "from toast.modules.deepmlp_translator import SGDDeepMLPAligner"),
        ("toast.modules.lowrank_translator",
         "from toast.modules.lowrank_translator import DEFAULT_RANKS, METHOD2CLASS"),
        ("toast.utils.dictionaries",
         "from toast.utils.dictionaries import NAME2TRANSLATORS, _translator_cost"),
        ("toast.modules.module",
         "from toast.modules.module import load_translator, save_translator, "
         "span_translator_key"),
    ]
    ns = {}
    for label, stmt in probes:
        print(f"       importing {label} ...", flush=True)
        exec(stmt, ns)
        check(label.split()[0], True)

    Estimator = ns["Estimator"]
    DEFAULT_RANKS, METHOD2CLASS = ns["DEFAULT_RANKS"], ns["METHOD2CLASS"]
    NAME2TRANSLATORS, _translator_cost = ns["NAME2TRANSLATORS"], ns["_translator_cost"]
    load_translator = ns["load_translator"]
    save_translator = ns["save_translator"]
    span_translator_key = ns["span_translator_key"]

    check("Estimator is an nn.Module", issubclass(Estimator, torch.nn.Module),
          "register_buffer will not work otherwise")
    check("3 low-rank methods", len(METHOD2CLASS) == 3)
    check("registry populated", len(NAME2TRANSLATORS) > 5, f"got {len(NAME2TRANSLATORS)}")

    print("\n1. registry", flush=True)
    for method in METHOD2CLASS:
        for r in DEFAULT_RANKS:
            check(f"{method}_{r} registered", f"{method}_{r}" in NAME2TRANSLATORS)

    print("\n2. fit/transform through the Translator wrapper", flush=True)
    errors = {}
    for method in METHOD2CLASS:
        print(f"       fitting {method}_8 ...", flush=True)
        t = NAME2TRANSLATORS[f"{method}_8"]()
        t.fit(x=x, y=y)
        out = t.transform(x)[0]
        check(f"{method}_8 transform shape", out.shape == y.shape, f"got {tuple(out.shape)}")
        errors[method] = ((out - y) ** 2).sum().item() / (y**2).sum().item()
        print(f"       rel_error {errors[method]:.6f}", flush=True)

    print("\n3. rrr is the optimum at equal rank", flush=True)
    check("rrr <= lowrank", errors["rrr"] <= errors["lowrank"] + 1e-9,
          f"rrr={errors['rrr']:.6f} lowrank={errors['lowrank']:.6f}")
    check("lora does not beat rrr", errors["lora"] >= errors["rrr"] - 1e-4,
          f"lora={errors['lora']:.6f} rrr={errors['rrr']:.6f}")

    print("\n4. full rank reproduces the plain linear translator", flush=True)
    lin = NAME2TRANSLATORS["linear"]()
    lin.fit(x=x, y=y)
    ref = lin.transform(x)[0]
    for method in ("lowrank", "rrr"):
        print(f"       fitting {method} at full rank D={D} ...", flush=True)
        cls = METHOD2CLASS[method]
        a = cls(rank=D)
        a.fit(x, y)
        got = a.transform(x)[0]
        check(f"{method}_{D} matches linear", torch.allclose(got, ref, atol=1e-3),
              f"max diff {(got - ref).abs().max():.2e}")

    print("\n5. parameter accounting", flush=True)
    for r in (8, 32):
        for method in METHOD2CLASS:
            cost = _translator_cost(f"{method}_{r}", D)
            check(f"{method}_{r} costs 2*D*r", cost == 2 * D * r, f"got {cost}")
    check("linear still costs D*D+D", _translator_cost("linear", D) == D * D + D)
    check("identity still free", _translator_cost("identity", D) == 0)

    print("\n6. stored factors match the accounting", flush=True)
    a = METHOD2CLASS["rrr"](rank=8)
    a.fit(x, y)
    check("A is (D, r)", tuple(a.A.shape) == (D, 8), f"got {tuple(a.A.shape)}")
    check("B is (r, D)", tuple(a.B.shape) == (8, D), f"got {tuple(a.B.shape)}")
    check("num_parameters == 2*D*r", a.num_parameters == 2 * D * 8)

    print("\n7. save/load round-trip (needed for the transfer experiment)", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for method in METHOD2CLASS:
            print(f"       save/load {method}_16 ...", flush=True)
            name = f"{method}_16"
            t = NAME2TRANSLATORS[name]()
            t.fit(x=x, y=y)
            before = t.transform(x)[0]

            key = span_translator_key(f"test__{method}", 2, 4)
            save_translator(t, key, base)
            reloaded = load_translator(key, name, base)
            after = reloaded.transform(x)[0]
            check(f"{name} reloads identically", torch.equal(before, after),
                  f"max diff {(before - after).abs().max():.2e}")

    print(f"\nAll {checks_run} checks passed. Safe to queue the sweep.", flush=True)


if __name__ == "__main__":
    main()
