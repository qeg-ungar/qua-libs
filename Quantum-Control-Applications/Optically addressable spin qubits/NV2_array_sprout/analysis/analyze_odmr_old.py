"""Analyze CW ODMR data saved by `05a_cw_odmr.py`.

This script targets the file format produced by `qualang_tools.results.data_handler.DataHandler`
(which is part of the `py-qua-tools` project): a folder containing `data.json` and `arrays.npz`.

Outputs:
  - Prints the fitted resonance frequency and a suggested update for `NV_IF_freq`.
  - Saves plots and a small JSON summary next to the dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# `qualang_tools` is distributed via the `py-qua-tools` project.
from qualang_tools.plot.fitting import Fit
from qualang_tools.units import unit


u = unit(coerce_to_integer=True)


#################
# User settings #
#################
# Optional: set this to a specific dataset folder (the folder containing `data.json` and `arrays.npz`).
# - Can be an absolute path, or a path *relative to the data root*.
# - Use `Path(r"...")` or forward slashes to avoid accidental escape sequences in strings.
# If left as None, the script analyzes the latest dataset under the default data root.
# CLI `--dataset` (if provided) always overrides this value.
DATASET_FOLDER: str | Path | None = None
#DATASET_FOLDER = "2025-12-31\\#13_cw_odmr_152618"  # Example relative path

# Optional: override where to search for datasets when DATASET_FOLDER is None.
# If left as None, the script tries (in order):
#   1) NV2_array/Data             (one level above experiments/)
#   2) NV2_array/experiments/Data (legacy layout)
# CLI `--data-root` (if provided) always overrides this value.
DATA_ROOT: str | Path | None = None

# Use the `py-qua-tools` fitting helper by default.
# This uses `qualang_tools.plot.fitting.Fit.transmission_resonator_spectroscopy` by fitting (1 - normalized_signal)
# so the ODMR dip becomes a Lorentzian peak.
USE_QUALANG_TOOLS_FIT: bool = True


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _pick_default_data_root(nv_root: Path) -> Path:
    if DATA_ROOT is not None:
        return _as_path(DATA_ROOT).resolve()

    preferred = nv_root / "Data"
    legacy = nv_root / "experiments" / "Data"
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    # Default to the preferred layout even if it doesn't exist yet
    return preferred


def _resolve_dataset_folder(dataset: str | Path, data_root: Path) -> Path:
    """Resolve a dataset input into a dataset *folder*.

    Accepts:
    - a dataset folder path
    - a path to data.json / arrays.npz inside a dataset folder
    - a relative path (resolved against data_root)
    """
    p = _as_path(dataset)
    data_root = data_root.resolve()

    # If user passed a relative path, first try it as-is (relative to CWD).
    # If that doesn't exist, fall back to interpreting it relative to the data root.
    if not p.is_absolute():
        if p.exists():
            pass
        else:
            p = data_root / p

    # If they passed the file inside the dataset, use its parent
    if p.name.lower() in {"data.json", "arrays.npz", "node.json"}:
        p = p.parent

    return p.resolve()


@dataclass(frozen=True)
class OdmrFitResult:
    f0_hz: float
    gamma_hz: float
    depth: float
    baseline: float
    nv_lo_hz: float

    @property
    def f0_mw_hz(self) -> float:
        return float(self.f0_hz)

    @property
    def f0_if_hz(self) -> float:
        return float(self.f0_hz - self.nv_lo_hz)


def _resolve_npz_ref(base_folder: Path, ref: str) -> np.ndarray:
    """Resolve DataHandler json references like './arrays.npz#counts_data'."""
    if "#" not in ref:
        raise ValueError(f"Expected npz reference of form '<file>.npz#<key>', got: {ref!r}")
    file_part, key = ref.split("#", 1)
    npz_path = (base_folder / file_part).resolve()
    with np.load(npz_path, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"Key {key!r} not found in {npz_path}")
        return np.array(data[key])


def load_datahandler_dataset(folder: Path) -> Dict[str, Any]:
    folder = folder.resolve()
    data_json = folder / "data.json"
    if not data_json.exists():
        raise FileNotFoundError(
            f"Missing data.json in {folder}. "
            "Expected a DataHandler dataset folder containing data.json and arrays.npz. "
            "Tip: you can pass the dataset folder or the path to data.json via --dataset."
        )

    payload: Dict[str, Any] = json.loads(data_json.read_text(encoding="utf-8"))

    def _maybe_resolve(value: Any) -> Any:
        if isinstance(value, str) and value.endswith(".npz") is False and "#" in value and ".npz" in value:
            return _resolve_npz_ref(folder, value)
        return value

    for k in ["IF_frequencies", "counts_data", "counts_ref_data"]:
        if k not in payload:
            continue
        payload[k] = _maybe_resolve(payload[k])

    return payload


def _get_lo_frequency_hz(dataset: Dict[str, Any]) -> float:
    config = dataset.get("config") or {}
    try:
        return float(config["elements"]["NV"]["mixInputs"]["lo_frequency"])
    except Exception:
        pass
    try:
        mixers = config.get("mixers", {}).get("mixer_NV", [])
        if mixers:
            return float(mixers[0]["lo_frequency"])
    except Exception:
        pass
    raise KeyError("Could not find NV LO frequency in saved config")


def _get_readout_len_ns(dataset: Dict[str, Any]) -> float:
    config = dataset.get("config") or {}
    pulses = config.get("pulses", {})
    for k in ("long_readout_pulse_1", "long_readout_pulse_2"):
        if k in pulses and "length" in pulses[k]:
            return float(pulses[k]["length"])  # ns
    raise KeyError("Could not infer readout length from saved config")


def lorentzian_peak(f_hz: np.ndarray, baseline: float, amplitude: float, f0_hz: float, gamma_hz: float) -> np.ndarray:
    """Lorentzian peak model.

    This script defines `signal_norm` as (ref - signal) / ref, which produces a *peak* at resonance.
    """
    return baseline + amplitude * (gamma_hz**2 / ((f_hz - f0_hz) ** 2 + gamma_hz**2))


def fit_odmr(f_hz: np.ndarray, signal_norm: np.ndarray, nv_lo_hz: float) -> OdmrFitResult:
    f_hz = np.asarray(f_hz, dtype=float)
    y = np.asarray(signal_norm, dtype=float)

    if USE_QUALANG_TOOLS_FIT:
        # `signal_norm` is already a peak, so we can fit it directly.
        out = Fit().transmission_resonator_spectroscopy(f_hz, y, plot=False, verbose=False, save=False)
        f0_hz = float(out["f"][0])
        k_fwhm_hz = float(out["k"][0])
        kc = float(out["kc"][0])
        offset = float(out["offset"][0])

        # Map the Fit() model:
        # y = (kc/k) * 1/(1 + 4((f-f0)^2/k^2)) + offset
        # which is equivalent to our Lorentzian peak with gamma=k/2 and amplitude=kc/k.
        gamma_hz = k_fwhm_hz / 2.0
        baseline = offset
        depth = kc / k_fwhm_hz if k_fwhm_hz != 0 else 0.0

        return OdmrFitResult(
            f0_hz=f0_hz,
            gamma_hz=gamma_hz,
            depth=depth,
            baseline=baseline,
            nv_lo_hz=float(nv_lo_hz),
        )

    # Initial guesses: peak at max of normalized signal
    idx0 = int(np.nanargmax(y))
    f0_guess = float(f_hz[idx0])
    baseline_guess = float(np.nanmedian(y))
    depth_guess = float(max(1e-6, float(np.nanmax(y)) - baseline_guess))
    gamma_guess = float(1.0 * u.MHz)

    # Bounds: keep gamma reasonable, depth positive
    p0 = [baseline_guess, depth_guess, f0_guess, gamma_guess]
    bounds = (
        [0.0, 0.0, float(np.min(f_hz)), float(1e3)],
        [2.0, 2.0, float(np.max(f_hz)), float(200 * u.MHz)],
    )

    popt, _pcov = curve_fit(
        lorentzian_peak,
        f_hz,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50_000,
    )
    baseline, depth, f0_hz, gamma_hz = map(float, popt)
    return OdmrFitResult(
        f0_hz=f0_hz,
        gamma_hz=gamma_hz,
        depth=depth,
        baseline=baseline,
        nv_lo_hz=float(nv_lo_hz),
    )


def find_latest_dataset(data_root: Path) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    candidates = [p for p in data_root.rglob("data.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No datasets found under {data_root}")

    # Pick most recently modified dataset folder
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def analyze_dataset(dataset_folder: Path, show: bool = True) -> Tuple[OdmrFitResult, Path]:
    dataset = load_datahandler_dataset(dataset_folder)
    f_if_hz = np.asarray(dataset["IF_frequencies"], dtype=float)
    counts = np.asarray(dataset["counts_data"], dtype=float)
    counts_ref = np.asarray(dataset["counts_ref_data"], dtype=float)

    nv_lo_hz = _get_lo_frequency_hz(dataset)
    readout_len_ns = _get_readout_len_ns(dataset)
    readout_len_s = readout_len_ns * 1e-9
    if readout_len_s <= 0:
        raise ValueError(f"Invalid readout_len_s={readout_len_s}")

    f_mw_hz = nv_lo_hz + f_if_hz

    # Convert to kcps (kcounts/s)
    signal_kcps = counts / 1000.0 / readout_len_s
    ref_kcps = counts_ref / 1000.0 / readout_len_s

    # Normalized signal for fitting (contrast peak)
    # Requested definition: (ref - signal) / ref
    with np.errstate(divide="ignore", invalid="ignore"):
        signal_norm = (ref_kcps - signal_kcps) / ref_kcps

    fit = fit_odmr(f_mw_hz, signal_norm, nv_lo_hz)

    # Plot
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax0.plot(f_mw_hz / 1e6, signal_kcps, label="signal")
    ax0.plot(f_mw_hz / 1e6, ref_kcps, label="reference")
    ax0.set_ylabel("Intensity [kcps]")
    ax0.legend(loc="best")

    f_dense = np.linspace(float(np.min(f_mw_hz)), float(np.max(f_mw_hz)), 2000)

    # Keep the fit on the contrast peak (ref - signal)/ref, but display the familiar ODMR dip: signal/ref = 1 - contrast
    signal_over_ref = 1.0 - signal_norm
    ax1.plot(f_mw_hz / 1e6, signal_over_ref, marker="o", linestyle=":", markersize=3, label="normalized signal")
    ax1.plot(
        f_dense / 1e6,
        1.0 - lorentzian_peak(f_dense, fit.baseline, fit.depth, fit.f0_hz, fit.gamma_hz),
        label="fit",
    )
    ax1.set_xlabel("MW frequency [MHz]")
    ax1.set_ylabel("Normalized")
    ax1.legend(loc="best")

    fig.suptitle("CW ODMR analysis")
    fig.tight_layout()

    out_png = dataset_folder / "odmr_analysis.png"
    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Save summary
    summary = {
        **asdict(fit),
        "f0_mw_mhz": fit.f0_mw_hz / 1e6,
        "f0_if_mhz": fit.f0_if_hz / 1e6,
        "readout_len_ns": readout_len_ns,
        "suggested_NV_IF_freq_hz": fit.f0_if_hz,
    }
    (dataset_folder / "odmr_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return fit, out_png


def main(argv: Optional[list[str]] = None) -> int:
    here = Path(__file__).resolve()
    nv_root = here.parents[1]
    default_data_root = _pick_default_data_root(nv_root)

    parser = argparse.ArgumentParser(description="Analyze CW ODMR dataset saved by DataHandler")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to a specific dataset folder containing data.json (default: latest under the data root)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Root folder containing dated DataHandler datasets (default: auto-detect)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib window")
    args = parser.parse_args(argv)

    if args.dataset is not None:
        dataset_folder = _resolve_dataset_folder(args.dataset, args.data_root)
    elif DATASET_FOLDER is not None:
        dataset_folder = _resolve_dataset_folder(DATASET_FOLDER, args.data_root)
    else:
        dataset_folder = find_latest_dataset(args.data_root)
    fit, out_png = analyze_dataset(dataset_folder, show=(not args.no_show))

    print(f"Dataset: {dataset_folder}")
    print(f"Resonance (MW): {fit.f0_mw_hz/1e6:.6f} MHz")
    print(f"Suggested NV_IF_freq: {fit.f0_if_hz/1e6:.6f} MHz  ({fit.f0_if_hz:.0f} Hz)")
    print(f"Fit linewidth gamma: {fit.gamma_hz/1e6:.3f} MHz")
    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
