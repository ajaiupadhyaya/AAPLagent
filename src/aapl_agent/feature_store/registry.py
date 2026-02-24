from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

FeatureFn = Callable[..., pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    func: FeatureFn
    description: str = ""
    allow_lookahead: bool = False


@dataclass
class FeatureRegistry:
    specs: dict[str, FeatureSpec] = field(default_factory=dict)

    def register(
        self,
        name: str,
        *,
        description: str = "",
        allow_lookahead: bool = False,
    ) -> Callable[[FeatureFn], FeatureFn]:
        def _decorator(func: FeatureFn) -> FeatureFn:
            if name in self.specs:
                raise ValueError(f"Feature '{name}' is already registered.")

            if not allow_lookahead:
                source = inspect.getsource(func)
                if ".shift(-" in source:
                    raise ValueError(
                        f"Feature '{name}' appears to use lookahead (negative shift)."
                    )

            self.specs[name] = FeatureSpec(
                name=name,
                func=func,
                description=description,
                allow_lookahead=allow_lookahead,
            )
            return func

        return _decorator

    def list_features(self) -> list[str]:
        return sorted(self.specs.keys())

    def compute(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        *,
        cache_dir: str | Path,
        namespace: str,
        market_data: dict[str, pd.DataFrame] | None = None,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        cache_path = Path(cache_dir) / namespace
        cache_path.mkdir(parents=True, exist_ok=True)

        outputs: list[pd.DataFrame] = []
        data_hash = _data_hash(df)

        for feature_name in feature_names:
            if feature_name not in self.specs:
                raise KeyError(f"Unknown feature: {feature_name}")

            spec = self.specs[feature_name]
            version_hash = _feature_version_hash(spec)
            artifact = cache_path / f"{feature_name}__{version_hash}__{data_hash}.parquet"

            if artifact.exists() and not force_recompute:
                feature_df = _read_parquet(artifact)
            else:
                computed = spec.func(df=df.copy(), market_data=market_data or {})
                feature_df = _coerce_to_frame(feature_name, computed, df.index)
                _write_parquet(feature_df, artifact)

            outputs.append(feature_df)

        if not outputs:
            return pd.DataFrame(index=df.index)

        return pd.concat(outputs, axis=1)


def _coerce_to_frame(name: str, computed: pd.Series | pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if isinstance(computed, pd.Series):
        out = computed.to_frame(name=name)
    elif isinstance(computed, pd.DataFrame):
        out = computed.copy()
    else:
        raise TypeError(f"Feature '{name}' returned unsupported type: {type(computed)}")

    out = out.reindex(index)
    if out.empty:
        raise ValueError(f"Feature '{name}' produced empty output.")
    return out


def _feature_version_hash(spec: FeatureSpec) -> str:
    payload = {
        "name": spec.name,
        "description": spec.description,
        "allow_lookahead": spec.allow_lookahead,
        "source": inspect.getsource(spec.func),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _data_hash(df: pd.DataFrame) -> str:
    payload = {
        "rows": len(df),
        "cols": list(df.columns),
        "index_start": str(df.index[0]) if len(df.index) else None,
        "index_end": str(df.index[-1]) if len(df.index) else None,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet feature cache. Ensure a parquet engine is installed (pyarrow)."
        ) from exc


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read feature cache: {path}") from exc


def _load_default_modules() -> None:
    from . import macro_features  # noqa: F401
    from . import microstructure_features  # noqa: F401
    from . import price_features  # noqa: F401
    from . import volatility_features  # noqa: F401


DEFAULT_REGISTRY = FeatureRegistry()
register_feature = DEFAULT_REGISTRY.register


def build_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    cache_dir: str | Path = "data/feature_cache",
    namespace: str = "aapl",
    market_data: dict[str, pd.DataFrame] | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    _load_default_modules()
    return DEFAULT_REGISTRY.compute(
        df=df,
        feature_names=feature_names,
        cache_dir=cache_dir,
        namespace=namespace,
        market_data=market_data,
        force_recompute=force_recompute,
    )
