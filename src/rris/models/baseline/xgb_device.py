"""
XGBoost device selection and safe GPU→CPU fallback.

COMMON ERRORS:
  - cudaErrorMemoryAllocation: VRAM full (notebook already loaded torch, or 6GB GPU).
    Fix: --device cpu when training inside Jupyter.
  - auto picks cuda if torch.cuda.is_available() — may conflict with other GPU users.
  - After GPU OOM, empty_cache() then refit on CPU (see fit_xgb_with_fallback).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class XgbRuntimeParams:
    tree_method: str
    device: str
    n_jobs: int
    requested: str
    resolved: str


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_xgb_params(
    device: DeviceChoice = "auto",
    *,
    n_jobs: int = -1,
) -> XgbRuntimeParams:
    if device == "cpu":
        return XgbRuntimeParams(
            tree_method="hist",
            device="cpu",
            n_jobs=n_jobs,
            requested="cpu",
            resolved="cpu",
        )
    if device == "cuda":
        return XgbRuntimeParams(
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            requested="cuda",
            resolved="cuda",
        )
    if _cuda_available():
        logger.info("XGBoost auto: CUDA available, using device=cuda")
        return XgbRuntimeParams(
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            requested="auto",
            resolved="cuda",
        )
    logger.info("XGBoost auto: no CUDA, using device=cpu (n_jobs=%s)", n_jobs)
    return XgbRuntimeParams(
        tree_method="hist",
        device="cpu",
        n_jobs=n_jobs,
        requested="auto",
        resolved="cpu",
    )


def apply_runtime_params(model: XGBClassifier, params: XgbRuntimeParams) -> XGBClassifier:
    model.set_params(
        tree_method=params.tree_method,
        device=params.device,
        n_jobs=params.n_jobs,
    )
    return model


def fit_xgb_with_fallback(
    model: XGBClassifier,
    X: Any,
    y: np.ndarray,
    *,
    runtime: XgbRuntimeParams,
    build_model: Callable[[XgbRuntimeParams], XGBClassifier],
    eval_set: Optional[list] = None,
    verbose: bool = True,
) -> tuple[XGBClassifier, XgbRuntimeParams]:
    fit_kwargs: dict = {"verbose": verbose}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    apply_runtime_params(model, runtime)
    try:
        model.fit(X, y, **fit_kwargs)
        return model, runtime
    except Exception as e:
        err = str(e).lower()
        is_malloc = "bad_malloc" in err or "failed to allocate" in err
        if is_malloc and runtime.resolved == "cpu" and runtime.n_jobs not in (1, 0):
            logger.warning(
                "XGBoost bad_malloc with n_jobs=%s; retrying fit with n_jobs=1",
                runtime.n_jobs,
            )
            cpu_rt = resolve_xgb_params("cpu", n_jobs=1)
            cpu_model = build_model(cpu_rt)
            try:
                cpu_model.fit(X, y, **fit_kwargs)
                return cpu_model, cpu_rt
            except Exception as e2:
                e = e2
                err = str(e).lower()
                is_malloc = "bad_malloc" in err or "failed to allocate" in err
        if is_malloc:
            raise RuntimeError(
                "XGBoost could not allocate memory (bad_malloc). On Windows: free space on C: "
                "(pagefile), use --scratch_dir on a drive with free GB (e.g. D:\\rris-scratch), "
                "or set env RRIS_SCRATCH_DIR. Also try --max_features 25000 --n_jobs 1."
            ) from e
        if runtime.resolved != "cuda":
            raise
        msg = str(e).lower()
        cuda_like = any(
            k in msg
            for k in ("cuda", "gpu", "device", "cublas", "out of memory", "driver")
        )
        if not cuda_like:
            raise
        logger.warning("XGBoost GPU fit failed (%s); falling back to CPU", e)
        try:
            import torch

            # Free VRAM before CPU refit — helps after cudaErrorMemoryAllocation in Jupyter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        cpu_n_jobs = runtime.n_jobs if runtime.n_jobs > 0 else -1
        cpu_runtime = resolve_xgb_params("cpu", n_jobs=cpu_n_jobs)
        cpu_model = build_model(cpu_runtime)
        cpu_model.fit(X, y, **fit_kwargs)
        return cpu_model, cpu_runtime


def build_xgb_classifier(
    *,
    base_kwargs: dict,
    device: DeviceChoice = "auto",
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> tuple[XGBClassifier, str, str]:
    runtime = resolve_xgb_params(device, n_jobs=n_jobs)
    model = XGBClassifier(
        **base_kwargs,
        tree_method=runtime.tree_method,
        device=runtime.device,
        n_jobs=runtime.n_jobs,
    )
    return model, runtime.resolved, runtime.requested


def fit_xgb_with_device_fallback(
    model: XGBClassifier,
    X: Any,
    y: np.ndarray,
    *,
    device_requested: DeviceChoice,
    n_jobs: int = -1,
    tree_method: str = "hist",
    base_kwargs: dict,
    eval_set: Optional[list] = None,
    verbose: bool = True,
) -> tuple[XGBClassifier, str]:
    runtime = resolve_xgb_params(device_requested, n_jobs=n_jobs)

    def _build(rt: XgbRuntimeParams) -> XGBClassifier:
        return XGBClassifier(
            **base_kwargs,
            tree_method=rt.tree_method,
            device=rt.device,
            n_jobs=rt.n_jobs,
        )

    fitted, used_rt = fit_xgb_with_fallback(
        model,
        X,
        y,
        runtime=runtime,
        build_model=_build,
        eval_set=eval_set,
        verbose=verbose,
    )
    return fitted, used_rt.resolved
