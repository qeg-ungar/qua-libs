from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

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

    def fit_cw_odmr(self, f_mw_hz: np.ndarray, contrast: np.ndarray, lo_hz: Optional[float] = None) -> FitResult:
        """ODMR fit (CW or pulsed) using qualang_tools Fit.transmission_resonator_spectroscopy.

        Args:
            f_mw_hz: MW frequency array in Hz (absolute frequency)
            contrast: normalized contrast array (ref - signal) / ref
            lo_hz: Optional LO frequency in Hz (for computing suggested IF update)

        Returns:
            FitResult with resonance parameters and optional suggested NV_IF update.
        """
        f_mw = np.asarray(f_mw_hz, dtype=float)
        y = np.asarray(contrast, dtype=float)

        out = self._fit.transmission_resonator_spectroscopy(f_mw, y, plot=False, verbose=False, save=False)

        fit_func = out.get("fit_func", None)
        y_fit = None
        if callable(fit_func):
            try:
                y_fit = np.asarray(fit_func(f_mw), dtype=float)
            except Exception:
                y_fit = None

        f0_hz = float(out["f"][0])
        gamma_hz = float(out["k"][0]) / 2.0
        fwhm_hz = 2.0 * gamma_hz
        amplitude = float(out["kc"][0]) / float(out["k"][0]) if float(out["k"][0]) != 0 else 0.0
        baseline = float(out["offset"][0])

        params: Dict[str, Any] = {
            "f0_hz": f0_hz,
            "gamma_hz": gamma_hz,
            "fwhm_hz": fwhm_hz,
            "baseline": baseline,
            "amplitude": amplitude,
        }
        if lo_hz is not None:
            params["nv_lo_hz"] = lo_hz
            params["suggested_NV_IF_freq_hz"] = f0_hz - lo_hz
        else:
            params["nv_lo_hz"] = None
            params["suggested_NV_IF_freq_hz"] = None
        params["fit_func_available"] = callable(fit_func)
        return FitResult(kind="odmr", params=params, x=f_mw, y=y, y_fit=y_fit)

    def fit_time_rabi(self, t_ns: np.ndarray, contrast: np.ndarray) -> FitResult:
        """Time Rabi fit using qualang_tools Fit.rabi.

        Args:
            t_ns: time array in nanoseconds
            contrast: normalized contrast array (ref - signal) / ref

        Returns:
            FitResult with Rabi parameters including estimated pi-time.
        """
        t_ns = np.asarray(t_ns, dtype=float)
        y = np.asarray(contrast, dtype=float)

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
        params["fit_func_available"] = callable(fit_func)
        return FitResult(kind="time_rabi", params=params, x=t_ns, y=y, y_fit=y_fit)

    def fit_power_rabi(self, dataset: Dict[str, Any], x180_amp_nv: Optional[float] = None) -> FitResult:
        """Power Rabi fit using custom scipy fitting.

        Expected keys:
          - a_vec: amplitude pre-factor
          - normalized_data (counts/counts_ref)

        If x180_amp_nv (Volts) is provided, x-axis is converted to volts (a_vec * x180_amp_nv).
        Otherwise returns fit in terms of the unitless pre-factor.

        Fits: y = amp * cos(2π f x) + offset
        (phase is fixed at 0 for power Rabi starting from zero amplitude)
        """
        a_vec = np.asarray(dataset["a_vec"], dtype=float)
        y = np.asarray(dataset.get("normalized_data"), dtype=float)

        if x180_amp_nv is not None:
            x = a_vec * float(x180_amp_nv)
            x_unit = "V"
        else:
            x = a_vec
            x_unit = "a.u."

        # Define fit function: cosine with zero phase
        def power_rabi_func(x_val, amplitude, frequency, offset):
            return amplitude * np.cos(2.0 * np.pi * frequency * x_val) + offset

        # Estimate initial parameters
        y_mean = np.mean(y)
        y_amplitude = (np.max(y) - np.min(y)) / 2.0

        # Find first minimum to estimate frequency
        min_idx = np.argmin(y)
        if min_idx > 0 and x[min_idx] > x[0]:
            # Pi pulse is at first minimum
            estimated_A_pi = x[min_idx]
            initial_freq = 0.5 / estimated_A_pi if estimated_A_pi > 0 else 0.5 / x.max()
        else:
            # Rough estimate: one period over data range
            x_range = x.max() - x.min()
            initial_freq = 1.0 / (2.0 * x_range) if x_range > 0 else 0.5

        p0 = [y_amplitude, initial_freq, y_mean]

        try:
            # Perform curve fitting
            popt, pcov = curve_fit(power_rabi_func, x, y, p0=p0, maxfev=10000)

            fitted_amp = float(popt[0])
            fitted_freq = float(popt[1])
            fitted_offset = float(popt[2])

            # Calculate fitted curve
            y_fit = power_rabi_func(x, *popt)

            # Calculate uncertainties
            perr = np.sqrt(np.diag(pcov))

            params = {
                "amp": np.array([fitted_amp]),
                "f": np.array([fitted_freq]),
                "phase": np.array([0.0]),  # Fixed at zero for power Rabi
                "offset": np.array([fitted_offset]),
                "T": None,  # No decay for simple power Rabi
                "amp_err": float(perr[0]),
                "f_err": float(perr[1]),
                "offset_err": float(perr[2]),
                "x_unit": x_unit,
                "fit_method": "scipy_curve_fit",
            }

            return FitResult(kind="power_rabi", params=params, x=x, y=y, y_fit=y_fit)

        except Exception as e:
            # Fitting failed, return data without fit
            params = {
                "amp": None,
                "f": None,
                "phase": None,
                "offset": None,
                "T": None,
                "x_unit": x_unit,
                "fit_method": "failed",
                "error": str(e),
            }
            return FitResult(kind="power_rabi", params=params, x=x, y=y, y_fit=None)
