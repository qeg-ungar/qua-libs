from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from qualang_tools.plot.fitting import Fit


@dataclass(frozen=True)
class FitResult:
    kind: str
    params: Dict[str, Any]
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    y_fit: Optional[np.ndarray] = None


class ExperimentFitter:
    """Fitting utilities for NV2_array experiments using py-qua-tools Fit."""

    def __init__(self):
        self._fit = Fit()

    @staticmethod
    def _config_lo_hz(dataset: Dict[str, Any]) -> float:
        cfg = dataset.get("config") or {}
        return float(cfg["elements"]["NV"]["mixInputs"]["lo_frequency"])

    @staticmethod
    def _config_readout_len_ns(dataset: Dict[str, Any]) -> float:
        cfg = dataset.get("config") or {}
        pulses = cfg.get("pulses", {})
        for k in ("long_readout_pulse_1", "long_readout_pulse_2", "readout_pulse_1", "readout_pulse_2"):
            if k in pulses and "length" in pulses[k]:
                return float(pulses[k]["length"])
        raise KeyError("Could not infer readout pulse length from config")

    def fit_cw_odmr(self, dataset: Dict[str, Any]) -> FitResult:
        """ODMR fit (CW or pulsed) using qualang_tools Fit.transmission_resonator_spectroscopy.

        Supported inputs:
          - Absolute MW frequency axis (preferred): dataset['MW_frequencies'] (Hz)
          - IF axis + LO: dataset['IF_frequencies'] (Hz) + dataset['config'] containing NV LO frequency

        Counts inputs:
          - dataset['counts_data']
          - dataset['counts_ref_data'] (or dataset['counts_dark_data'] fallback)

        Normalization convention:
          - contrast = (ref - signal) / ref

        Returns resonance frequency and (when LO is known) a suggested NV_IF update.
        """
        counts = np.asarray(dataset["counts_data"], dtype=float)
        if "counts_ref_data" in dataset:
            counts_ref = np.asarray(dataset["counts_ref_data"], dtype=float)
        elif "counts_dark_data" in dataset:
            counts_ref = np.asarray(dataset["counts_dark_data"], dtype=float)
        else:
            raise KeyError("Missing reference counts array: expected 'counts_ref_data' or 'counts_dark_data'")

        lo_hz: Optional[float] = None
        if "MW_frequencies" in dataset:
            f_mw = np.asarray(dataset["MW_frequencies"], dtype=float)
        elif "mw_frequencies" in dataset:
            f_mw = np.asarray(dataset["mw_frequencies"], dtype=float)
        else:
            f_if = np.asarray(dataset["IF_frequencies"], dtype=float)
            try:
                lo_hz = self._config_lo_hz(dataset)
            except Exception:
                lo_hz = None
            f_mw = (lo_hz + f_if) if lo_hz is not None else f_if

        # Convert counts to rates when a readout length is available.
        # This does not change the contrast value, but keeps raw plots consistent with other notebooks.
        try:
            readout_len_ns = self._config_readout_len_ns(dataset)
            readout_s = readout_len_ns * 1e-9
            signal = counts / 1000.0 / readout_s
            ref = counts_ref / 1000.0 / readout_s
        except Exception:
            signal = counts
            ref = counts_ref

        with np.errstate(divide="ignore", invalid="ignore"):
            contrast = (ref - signal) / ref

        out = self._fit.transmission_resonator_spectroscopy(f_mw, contrast, plot=False, verbose=False, save=False)

        fit_func = out.get("fit_func", None)
        y_fit = None
        if callable(fit_func):
            try:
                y_fit = np.asarray(fit_func(f_mw), dtype=float)
            except Exception:
                y_fit = None

        f0_hz = float(out["f"][0])
        gamma_hz = float(out["k"][0]) / 2.0
        depth = float(out["kc"][0]) / float(out["k"][0]) if float(out["k"][0]) != 0 else 0.0
        baseline = float(out["offset"][0])

        params: Dict[str, Any] = {
            "f0_hz": f0_hz,
            "f0_mw_mhz": f0_hz / 1e6,
            "gamma_hz": gamma_hz,
            "gamma_mhz": gamma_hz / 1e6,
            "baseline": baseline,
            "depth": depth,
        }
        if lo_hz is not None:
            params["nv_lo_hz"] = lo_hz
            params["suggested_NV_IF_freq_hz"] = f0_hz - lo_hz
            params["suggested_NV_IF_freq_mhz"] = (f0_hz - lo_hz) / 1e6
        else:
            params["nv_lo_hz"] = None
            params["suggested_NV_IF_freq_hz"] = None
            params["suggested_NV_IF_freq_mhz"] = None
        params["fit_func_available"] = callable(fit_func)
        return FitResult(kind="odmr", params=params, x=f_mw, y=contrast, y_fit=y_fit)

    def fit_time_rabi(self, dataset: Dict[str, Any]) -> FitResult:
        """Time Rabi fit using qualang_tools Fit.rabi.

        Expected keys:
          - t_vec: clock cycles (4ns)
                    - counts_data
                    - counts_ref_data or counts_dark_data
                    - normalized_data (optional): if provided, is fit as-is

                Normalization convention (when `normalized_data` is not provided):
                    - contrast: (ref - signal) / ref

        Returns pi time estimate (in ns) if available from the fit output.
        """
        t_vec = np.asarray(dataset["t_vec"], dtype=float)
        # Convert clock cycles to ns (scripts define 1 cc = 4ns)
        t_ns = t_vec * 4.0

        normalized_data = dataset.get("normalized_data", None)
        if normalized_data is not None:
            y = np.asarray(normalized_data, dtype=float)
        else:
            counts = np.asarray(dataset["counts_data"], dtype=float)
            if "counts_ref_data" in dataset:
                ref = np.asarray(dataset["counts_ref_data"], dtype=float)
            elif "counts_dark_data" in dataset:
                ref = np.asarray(dataset["counts_dark_data"], dtype=float)
            else:
                raise KeyError("Missing reference counts array: expected 'counts_ref_data' or 'counts_dark_data'")

            with np.errstate(divide="ignore", invalid="ignore"):
                y = (ref - counts) / ref

        out = self._fit.rabi(t_ns, y, plot=False, verbose=False, save=False)

        def _as_scalar(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                arr = np.asarray(v, dtype=float).ravel()
                if arr.size == 0:
                    return None
                return float(arr[0])
            except Exception:
                return None

        fit_func = out.get("fit_func", None)
        y_fit: Optional[np.ndarray] = None
        if callable(fit_func):
            try:
                y_fit = np.asarray(fit_func(t_ns), dtype=float)
            except Exception:
                y_fit = None

        # Some qualang_tools versions don't return a callable fit function.
        # Reconstruct a fitted curve from the returned parameters when possible.
        if y_fit is None:
            amp = _as_scalar(out.get("amp"))
            freq = _as_scalar(out.get("f"))
            phase = _as_scalar(out.get("phase"))
            offset = _as_scalar(out.get("offset"))
            decay = _as_scalar(out.get("T"))

            if amp is not None and freq is not None and phase is not None and offset is not None:
                y_fit = amp * np.cos(2.0 * np.pi * freq * t_ns + phase)
                if decay is not None and decay > 0:
                    y_fit = y_fit * np.exp(-t_ns / decay)
                y_fit = y_fit + offset

        # out is a dict with many keys depending on Fit implementation.
        params = {k: v for k, v in out.items() if k != "fit_func"}
        params["x_unit"] = "ns"
        params["y_norm"] = "contrast_(ref-signal)/ref" if normalized_data is None else "provided_normalized_data"
        params["fit_func_available"] = callable(fit_func)
        return FitResult(kind="time_rabi", params=params, x=t_ns, y=y, y_fit=y_fit)

    def fit_power_rabi(self, dataset: Dict[str, Any], x180_amp_nv: Optional[float] = None) -> FitResult:
        """Power Rabi fit.

        Expected keys:
          - a_vec: amplitude pre-factor
          - normalized_data (counts/counts_ref)

        If x180_amp_nv (Volts) is provided, x-axis is converted to volts (a_vec * x180_amp_nv).
        Otherwise returns fit in terms of the unitless pre-factor.
        """
        a_vec = np.asarray(dataset["a_vec"], dtype=float)
        y = np.asarray(dataset.get("normalized_data"), dtype=float)

        if x180_amp_nv is not None:
            x = a_vec * float(x180_amp_nv)
            x_unit = "V"
        else:
            x = a_vec
            x_unit = "a.u."

        out = self._fit.rabi(x, y, plot=False, verbose=False, save=False)
        params = {k: v for k, v in out.items() if k != "fit_func"}
        params["x_unit"] = x_unit
        return FitResult(kind="power_rabi", params=params)
