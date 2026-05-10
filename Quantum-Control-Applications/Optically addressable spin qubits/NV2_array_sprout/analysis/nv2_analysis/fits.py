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

    def fit_cw_odmr_double_lorentzian(
        self,
        f_mw_hz: np.ndarray,
        dip_depth: np.ndarray,
        lo_hz: Optional[float] = None,
    ) -> FitResult:
        """Fit ODMR to a sum of two Lorentzians (two resonances).

        This is intended for CW/pulsed ODMR traces that contain two dips.
        The input should be the *dip depth* (e.g. ``1 - contrast``) so that
        resonances appear as positive peaks.

        Model:
            y(f) = baseline + L(f; f01, gamma1, amp1) + L(f; f02, gamma2, amp2)

            L(f; f0, gamma, amp) = amp * gamma^2 / ((f - f0)^2 + gamma^2)

        Args:
            f_mw_hz: MW frequency array in Hz.
            dip_depth: dip depth array (positive-going resonances), e.g. ``1 - contrast``.
            lo_hz: Optional LO frequency in Hz (to compute suggested IF updates for both resonances).

        Returns:
            FitResult with parameters for both resonances.
        """
        f = np.asarray(f_mw_hz, dtype=float).ravel()
        y = np.asarray(dip_depth, dtype=float).ravel()
        if f.size != y.size:
            raise ValueError(f"f_mw_hz and dip_depth must have same length, got {f.size} and {y.size}")
        if f.size < 6:
            raise ValueError("Need at least 6 points for a stable double-Lorentzian fit")

        # Sort for more stable initial-guess heuristics.
        order = np.argsort(f)
        f_sorted = f[order]
        y_sorted = y[order]

        # Robust baseline guess from the edges (avoid resonance region).
        n = f_sorted.size
        edge = max(1, n // 10)
        edge_vals = np.concatenate([y_sorted[:edge], y_sorted[-edge:]])
        baseline0 = float(np.nanmedian(edge_vals))

        # Pick two separated peak indices.
        y_work = np.asarray(y_sorted, dtype=float)
        idx1 = int(np.nanargmax(y_work))
        exclusion = max(3, n // 20)
        y_work2 = y_work.copy()
        y_work2[max(0, idx1 - exclusion) : min(n, idx1 + exclusion + 1)] = -np.inf
        if np.all(~np.isfinite(y_work2)):
            idx2 = int(np.nanargmax(y_work))
        else:
            idx2 = int(np.nanargmax(y_work2))

        f01_0 = float(f_sorted[idx1])
        f02_0 = float(f_sorted[idx2])
        if f02_0 < f01_0:
            f01_0, f02_0 = f02_0, f01_0

        amp1_0 = float(max(0.0, y_sorted[idx1] - baseline0))
        amp2_0 = float(max(0.0, y_sorted[idx2] - baseline0))
        amp0_max = float(np.nanmax(y_sorted) - np.nanmin(y_sorted))
        if not np.isfinite(amp0_max) or amp0_max <= 0:
            amp0_max = 1.0
        if amp1_0 <= 0:
            amp1_0 = 0.5 * amp0_max
        if amp2_0 <= 0:
            amp2_0 = 0.5 * amp0_max

        span = float(np.nanmax(f_sorted) - np.nanmin(f_sorted))
        df = np.diff(f_sorted)
        df_med = float(np.nanmedian(np.abs(df))) if df.size else 0.0
        if not np.isfinite(df_med) or df_med <= 0:
            df_med = span / max(10.0, float(n)) if span > 0 else 1.0

        gamma0 = max(5.0 * df_med, span / 200.0)
        if not np.isfinite(gamma0) or gamma0 <= 0:
            gamma0 = 1e6

        def _lorentz(f_hz: np.ndarray, f0: float, gamma: float, amp: float) -> np.ndarray:
            return amp * (gamma**2) / ((f_hz - f0) ** 2 + gamma**2)

        def model(
            f_hz: np.ndarray,
            baseline: float,
            f01: float,
            gamma1: float,
            amp1: float,
            f02: float,
            gamma2: float,
            amp2: float,
        ) -> np.ndarray:
            return baseline + _lorentz(f_hz, f01, gamma1, amp1) + _lorentz(f_hz, f02, gamma2, amp2)

        # Parameter bounds.
        y_min = float(np.nanmin(y_sorted))
        y_max = float(np.nanmax(y_sorted))
        y_rng = float(y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else amp0_max

        gamma_min = max(df_med / 10.0, 1.0)
        gamma_max = max(span, gamma_min * 10.0)
        amp_max = max(10.0 * y_rng, 1e-12)

        lower = [
            y_min - 2.0 * y_rng,
            float(np.nanmin(f_sorted)),
            gamma_min,
            0.0,
            float(np.nanmin(f_sorted)),
            gamma_min,
            0.0,
        ]
        upper = [
            y_max + 2.0 * y_rng,
            float(np.nanmax(f_sorted)),
            gamma_max,
            amp_max,
            float(np.nanmax(f_sorted)),
            gamma_max,
            amp_max,
        ]

        p0 = [baseline0, f01_0, gamma0, amp1_0, f02_0, gamma0, amp2_0]

        popt, pcov = curve_fit(
            model,
            f_sorted,
            y_sorted,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )

        baseline, f01, gamma1, amp1, f02, gamma2, amp2 = [float(v) for v in popt]
        if f02 < f01:
            f01, f02 = f02, f01
            gamma1, gamma2 = gamma2, gamma1
            amp1, amp2 = amp2, amp1

        y_fit_sorted = model(f_sorted, baseline, f01, gamma1, amp1, f02, gamma2, amp2)
        y_fit = np.empty_like(y_fit_sorted)
        y_fit[order] = y_fit_sorted

        # Parameter uncertainties.
        perr = None
        try:
            perr = np.sqrt(np.diag(pcov)).astype(float)
        except Exception:
            perr = None

        params: Dict[str, Any] = {
            "baseline": baseline,
            "f01_hz": f01,
            "gamma1_hz": gamma1,
            "fwhm1_hz": 2.0 * gamma1,
            "amplitude1": amp1,
            "f02_hz": f02,
            "gamma2_hz": gamma2,
            "fwhm2_hz": 2.0 * gamma2,
            "amplitude2": amp2,
            "split_hz": abs(f02 - f01),
            "fit_method": "scipy_curve_fit",
        }
        if perr is not None and perr.size == 7:
            params.update(
                {
                    "baseline_err": float(perr[0]),
                    "f01_err_hz": float(perr[1]),
                    "gamma1_err_hz": float(perr[2]),
                    "amplitude1_err": float(perr[3]),
                    "f02_err_hz": float(perr[4]),
                    "gamma2_err_hz": float(perr[5]),
                    "amplitude2_err": float(perr[6]),
                }
            )

        if lo_hz is not None:
            params["nv_lo_hz"] = float(lo_hz)
            params["suggested_NV_IF_freqs_hz"] = [f01 - float(lo_hz), f02 - float(lo_hz)]
        else:
            params["nv_lo_hz"] = None
            params["suggested_NV_IF_freqs_hz"] = None

        return FitResult(kind="odmr_double_lorentzian", params=params, x=f, y=y, y_fit=y_fit)

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

        Fits: y = amp * cos(2π f x + phase) * exp(-x/T) + offset
        where x is in volts if x180_amp_nv is provided, else in a.u.
        """
        a_vec = np.asarray(dataset["a_vec"], dtype=float)
        y = np.asarray(dataset.get("normalized_data"), dtype=float)

        if x180_amp_nv is not None:
            x = a_vec * float(x180_amp_nv)
            x_unit = "V"
        else:
            x = a_vec
            x_unit = "a.u."

        # Fit function: damped cosine.
        # Use x0 = x - x.min() so T is well-defined even if x doesn't start at 0.
        x0 = x - float(np.nanmin(x))

        def power_rabi_func(x0_val, amplitude, frequency, phase, offset, decay):
            return amplitude * np.cos(2.0 * np.pi * frequency * x0_val + phase) * np.exp(-x0_val / decay) + offset

        # Estimate initial parameters
        y_mean = float(np.nanmean(y))
        y_amplitude = float((np.nanmax(y) - np.nanmin(y)) / 2.0)

        # Find first minimum to estimate frequency
        min_idx = int(np.nanargmin(y))
        if min_idx > 0 and x0[min_idx] > x0[0]:
            # Pi pulse is at first minimum
            estimated_A_pi = float(x0[min_idx])
            initial_freq = 0.5 / estimated_A_pi if estimated_A_pi > 0 else 0.5 / float(np.nanmax(x0))
        else:
            # Rough estimate: one period over data range
            x_range = float(np.nanmax(x0) - np.nanmin(x0))
            initial_freq = 1.0 / (2.0 * x_range) if x_range > 0 else 0.5

        x_range = float(np.nanmax(x0) - np.nanmin(x0))
        T0 = max(x_range, 1e-6) * 5.0
        p0 = [y_amplitude, initial_freq, 0.0, y_mean, T0]

        # Bounds: enforce f >= 0 and T > 0.
        y_span = float(np.nanmax(y) - np.nanmin(y))
        if not np.isfinite(y_span) or y_span <= 0:
            y_span = 1.0
        amp_max = 10.0 * y_span
        off_min = float(np.nanmin(y) - 2.0 * y_span)
        off_max = float(np.nanmax(y) + 2.0 * y_span)
        T_min = max(x_range / 100.0, 1e-9)
        T_max = max(x_range * 1e6, T_min * 10.0)
        bounds = (
            [-amp_max, 0.0, -4.0 * np.pi, off_min, T_min],
            [amp_max, np.inf, 4.0 * np.pi, off_max, T_max],
        )

        try:
            # Perform curve fitting
            popt, pcov = curve_fit(power_rabi_func, x0, y, p0=p0, bounds=bounds, maxfev=20000)

            fitted_amp = float(popt[0])
            fitted_freq = float(popt[1])
            fitted_phase = float(popt[2])
            fitted_offset = float(popt[3])
            fitted_T = float(popt[4])

            # Calculate fitted curve
            y_fit = power_rabi_func(x0, *popt)

            # Calculate uncertainties
            perr = np.sqrt(np.diag(pcov))

            params = {
                "amp": np.array([fitted_amp]),
                "f": np.array([fitted_freq]),
                "phase": np.array([fitted_phase]),
                "offset": np.array([fitted_offset]),
                "T": np.array([fitted_T]),
                "amp_err": float(perr[0]) if perr.size > 0 else None,
                "f_err": float(perr[1]) if perr.size > 1 else None,
                "phase_err": float(perr[2]) if perr.size > 2 else None,
                "offset_err": float(perr[3]) if perr.size > 3 else None,
                "T_err": float(perr[4]) if perr.size > 4 else None,
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
