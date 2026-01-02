from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetRef:
    """Reference to a saved DataHandler dataset folder."""

    folder: Path

    @property
    def data_json(self) -> Path:
        return self.folder / "data.json"

    @property
    def arrays_npz(self) -> Path:
        return self.folder / "arrays.npz"


class DatasetReader:
    """Locate and load `qualang_tools.results.data_handler.DataHandler` datasets.

    This reads the on-disk format written by DataHandler:
      - `data.json` containing metadata and references like `./arrays.npz#counts_data`
      - `arrays.npz` containing numpy arrays

    Note: In the qualang_tools version in this workspace, DataHandler doesn't expose a public load API.
    """

    def __init__(self, nv_root: Path, data_root: Optional[Path] = None):
        self.nv_root = Path(nv_root).resolve()
        self.data_root = Path(data_root).resolve() if data_root is not None else self._pick_default_data_root()

    def _pick_default_data_root(self) -> Path:
        preferred = self.nv_root / "Data"
        legacy = self.nv_root / "experiments" / "Data"
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
        return preferred

    def list_datasets(self) -> Sequence[DatasetRef]:
        if not self.data_root.exists():
            return []
        folders = [p.parent for p in self.data_root.rglob("data.json") if p.is_file()]
        # newest first
        folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [DatasetRef(folder=f) for f in folders]

    def latest(self) -> DatasetRef:
        datasets = self.list_datasets()
        if not datasets:
            raise FileNotFoundError(f"No datasets found under {self.data_root}")
        return datasets[0]

    def resolve_dataset(self, dataset: str | Path) -> DatasetRef:
        p = Path(dataset)
        if not p.is_absolute():
            # try as-is first
            if not p.exists():
                p = self.data_root / p
        if p.name.lower() in {"data.json", "arrays.npz", "node.json"}:
            p = p.parent
        p = p.resolve()
        return DatasetRef(folder=p)

    @staticmethod
    def _resolve_npz_ref(base_folder: Path, ref: str) -> np.ndarray:
        if "#" not in ref:
            raise ValueError(f"Expected npz reference '<file>.npz#<key>', got: {ref!r}")
        file_part, key = ref.split("#", 1)
        npz_path = (base_folder / file_part).resolve()
        with np.load(npz_path, allow_pickle=True) as data:
            if key not in data:
                raise KeyError(f"Key {key!r} not found in {npz_path}")
            return np.array(data[key])

    def load(self, dataset: DatasetRef) -> Dict[str, Any]:
        folder = dataset.folder
        data_json = folder / "data.json"
        if not data_json.exists():
            raise FileNotFoundError(f"Missing data.json in {folder}")

        payload: Dict[str, Any] = json.loads(data_json.read_text(encoding="utf-8"))

        def maybe_resolve(value: Any) -> Any:
            if isinstance(value, str) and "#" in value and ".npz" in value:
                return self._resolve_npz_ref(folder, value)
            return value

        for k, v in list(payload.items()):
            payload[k] = maybe_resolve(v)

        return payload

    def load_arrays(self, dataset: DatasetRef, keys: Iterable[str]) -> Dict[str, np.ndarray]:
        loaded = self.load(dataset)
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            if k not in loaded:
                raise KeyError(f"Missing key {k!r} in dataset {dataset.folder}")
            out[k] = np.asarray(loaded[k])
        return out


def infer_nv_root_from_this_file(file_path: Path) -> Path:
    """Given a file inside NV2_array/analysis, infer NV2_array root."""
    p = Path(file_path).resolve()
    # .../NV2_array/analysis/nv2_analysis/dataset.py -> NV2_array
    for parent in p.parents:
        if parent.name == "NV2_array":
            return parent
    raise RuntimeError(f"Could not infer NV2_array root from {file_path}")
