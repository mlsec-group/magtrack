from typing import Tuple

import numpy as np
import pandas as pd
import scipy.signal
from fastgoertzel import goertzel_sliding_batch
from scipy.interpolate import UnivariateSpline


def _sampling_rate(data: pd.DataFrame) -> float:
    timestamps = data["timestamp"].astype("int64").to_numpy() / 1e9
    dt = np.median(np.diff(timestamps))
    return 1.0 / dt


def compute_fft_spectrum(
        data: pd.DataFrame,
        detrend: bool = True,
        window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the (single-shot) FFT amplitude spectrum of *data*.

    Returns ``(freqs, amplitude, fs)``.  Cacheable: depends only on the time
    series and the *detrend* / *window* flags.
    """
    signal = data["magnitude"].to_numpy()
    fs = _sampling_rate(data)

    if detrend:
        signal = scipy.signal.detrend(signal, type="constant")
    if window:
        signal = signal * scipy.signal.windows.hann(len(signal))

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    spectrum = np.fft.rfft(signal)
    amplitude = np.abs(spectrum) / n
    return freqs.astype(np.float64), amplitude.astype(np.float64), float(fs)


def compute_psd_spectrum(
        data: pd.DataFrame,
        nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the Welch PSD spectrum of *data* for a given *nperseg*.

    Returns ``(freqs, psd, fs)``.  Cacheable per ``nperseg``.
    """
    signal = data["magnitude"].to_numpy()
    fs = _sampling_rate(data)
    nperseg_eff = min(int(nperseg), len(signal))
    noverlap_eff = nperseg_eff // 2
    freqs, psd = scipy.signal.welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        detrend="constant",
        scaling="density",
    )
    return freqs.astype(np.float64), psd.astype(np.float64), float(fs)


def compute_snr_from_spectrum(
        freqs: np.ndarray,
        spectrum: np.ndarray,
        *,
        target_freq: float,
        signal_delta: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        harmonics_mask: bool,
        noise_model: str,
        guard_band: float,
        noise_band: float,
        min_noise_bins: int = 4,
) -> float:
    """Derive the peak-to-noise-floor SNR from a precomputed spectrum.

    Same logic as :func:`compute_snr_fft` / :func:`compute_snr_psd` but skips
    the (cacheable) FFT / Welch step.  Returns ``np.nan`` on any error.
    """
    try:
        valid = (freqs >= spectrum_min_freq) & (freqs <= spectrum_max_freq)
        f = freqs[valid]
        s = spectrum[valid]
        if f.size == 0:
            return np.nan
        target_mask = np.abs(f - target_freq) <= signal_delta
        if not np.any(target_mask):
            return np.nan
        target_peak = float(np.max(s[target_mask]))
        noise_mask = ~target_mask
        if harmonics_mask:
            for h in range(2, 4):
                noise_mask &= np.abs(f - target_freq * h) > signal_delta
        if np.sum(noise_mask) < min_noise_bins:
            return np.nan
        local_noise = _estimate_noise_floor(
            f, s, noise_mask, target_freq, noise_model, guard_band, noise_band,
        )
        if local_noise <= 0:
            return np.nan
        return float(target_peak / local_noise)
    except Exception:
        return np.nan


def _estimate_noise_floor(
        freqs: np.ndarray,
        signal: np.ndarray,
        noise_mask: np.ndarray,
        target_freq: float,
        noise_model: str,
        guard_band: float,
        noise_band: float,
) -> float:
    """Estimate the local noise floor from an array.

    Parameters
    ----------
    freqs : array
        Frequency axis, already clipped to the analysis band.
    signal : array
        Spectrum or PSD values, same length as *freqs*.
    noise_mask : bool array
        True for bins considered noise (signal band + harmonics already excluded).
    target_freq : float
        Centre of the signal of interest (Hz).
    noise_model : {"median", "interpolate", "spline"}
        Which estimator to use.
    guard_band : float
        Distance from *target_freq* to the inner edge of the reference cells (Hz).
        Used only by ``"interpolate"``.
    noise_band : float
        Width of each reference cell (Hz).  Used only by ``"interpolate"``.

    Returns
    -------
    float
        Estimated noise floor at *target_freq*.
    """
    if noise_model == "median":
        return float(np.median(signal[noise_mask]))

    if noise_model == "interpolate":
        lo = (noise_mask
              & (freqs >= target_freq - guard_band - noise_band)
              & (freqs < target_freq - guard_band))
        hi = (noise_mask
              & (freqs > target_freq + guard_band)
              & (freqs <= target_freq + guard_band + noise_band))
        lo_ok, hi_ok = np.any(lo), np.any(hi)

        if lo_ok and hi_ok:
            f_lo = float(np.mean(freqs[lo]))
            f_hi = float(np.mean(freqs[hi]))
            s_lo = float(np.median(signal[lo]))
            s_hi = float(np.median(signal[hi]))
            if s_lo > 0 and s_hi > 0 and f_lo > 0 and f_hi > 0:
                slope = (np.log(s_hi) - np.log(s_lo)) / (np.log(f_hi) - np.log(f_lo))
                return float(np.exp(np.log(s_lo) + slope * (np.log(target_freq) - np.log(f_lo))))
        raise ValueError(
            "Not enough valid noise bins for interpolation. "
            "Check guard_band and noise_band parameters."
        )

    if noise_model == "spline":
        nm = noise_mask & (signal > 0) & (freqs > 0)
        if np.sum(nm) >= 4:
            log_f = np.log(freqs[nm])
            log_s = np.log(signal[nm])
            try:
                spline = UnivariateSpline(log_f, log_s, k=3, s=int(np.sum(nm)))
                val = float(np.exp(float(spline(np.log(target_freq)))))
                if val > 0:
                    return val
            except Exception:
                pass
        raise ValueError(
            "Not enough valid noise bins for spline fitting. "
            "Check noise_model and parameters."
        )

    raise ValueError(
        f"Unknown noise_model '{noise_model}'. Choose from 'median', 'interpolate', 'spline'."
    )


def compute_snr_fft(
        data: pd.DataFrame,
        target_freq: float = 16.7,
        signal_delta: float = 0.15,
        spectrum_min_freq: float = 1.0,
        spectrum_max_freq: float = 45.0,
        detrend: bool = True,
        window: bool = True,
        harmonics_mask: bool = True,
        noise_model: str = "median",
        guard_band: float = 0.5,
        noise_band: float = 3.0,
        **kwargs,
) -> float:
    """Compute the peak-to-noise-floor SNR using a single-shot windowed FFT.

    Returns ``np.nan`` on any error (e.g. too few frequency bins).
    """
    try:
        signal = data["magnitude"].to_numpy()
        timestamps = data["timestamp"].astype("int64").to_numpy() / 1e9
        dt = np.median(np.diff(timestamps))
        fs = 1.0 / dt
        if detrend:
            signal = scipy.signal.detrend(signal, type="constant")
        if window:
            signal = signal * scipy.signal.windows.hann(len(signal))
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1 / fs)
        spectrum = np.fft.rfft(signal)
        amplitude = np.abs(spectrum) / n
        valid = (freqs >= spectrum_min_freq) & (freqs <= spectrum_max_freq)
        freqs = freqs[valid]
        amplitude = amplitude[valid]
        if len(freqs) == 0:
            return np.nan
        target_mask = np.abs(freqs - target_freq) <= signal_delta
        if not np.any(target_mask):
            return np.nan
        target_peak = float(np.max(amplitude[target_mask]))
        noise_mask = ~target_mask
        if harmonics_mask:
            for h in range(2, 4):
                noise_mask &= np.abs(freqs - target_freq * h) > signal_delta
        if np.sum(noise_mask) < 4:
            return np.nan
        local_noise = _estimate_noise_floor(
            freqs, amplitude, noise_mask, target_freq, noise_model, guard_band, noise_band,
        )
        if local_noise <= 0:
            return np.nan
        return float(target_peak / local_noise)
    except Exception:
        return np.nan


def _build_goertzel_probes(
        target_freq: float,
        spectrum_min_freq: float,
        spectrum_max_freq: float,
        guard_band: float,
        noise_n_probes: int,
        probe_split: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (below_probes, above_probes) frequency arrays in Hz.

    ``probe_split`` is the fraction of probes placed *below* the target.
    Each side is laid out with ``np.linspace`` between the spectrum edge and
    the inner guard band.
    """
    n_below = int(round(probe_split * noise_n_probes))
    n_above = int(noise_n_probes) - n_below

    below_lo = float(spectrum_min_freq)
    below_hi = float(target_freq) - float(guard_band)
    above_lo = float(target_freq) + float(guard_band)
    above_hi = float(spectrum_max_freq)

    if n_below > 0 and below_hi > below_lo:
        below = np.linspace(below_lo, below_hi, n_below, dtype=np.float64)
    else:
        below = np.empty(0, dtype=np.float64)

    if n_above > 0 and above_hi > above_lo:
        above = np.linspace(above_lo, above_hi, n_above, dtype=np.float64)
    else:
        above = np.empty(0, dtype=np.float64)

    return below, above


def compute_snr_goertzel(
        data: pd.DataFrame,
        target_freq: float = 16.7,
        spectrum_min_freq: float = 1.0,
        spectrum_max_freq: float = 45.0,
        guard_band: float = 0.5,
        noise_n_probes: int = 12,
        probe_split: float = 0.5,
        noise_model: str = "median",
        **kwargs,
) -> float:
    """Compute peak-to-noise-floor SNR using the Goertzel algorithm.

    Evaluates the amplitude at *target_freq* plus a set of off-target probe
    frequencies via a single ``goertzel_sliding_batch`` call with
    ``window_size = len(signal)`` (i.e. one window over the whole chunk).
    Probes are placed on a linspace below and above the target, split via
    *probe_split* and excluded from the ``[target ± guard_band]`` interval.

    Returns ``np.nan`` on any error.
    """
    try:
        signal = data["magnitude"].to_numpy(dtype=np.float64)
        n = len(signal)
        if n < 8:
            return np.nan
        fs = _sampling_rate(data)
        nyquist = fs / 2.0

        below, above = _build_goertzel_probes(
            target_freq=target_freq,
            spectrum_min_freq=spectrum_min_freq,
            spectrum_max_freq=spectrum_max_freq,
            guard_band=guard_band,
            noise_n_probes=int(noise_n_probes),
            probe_split=float(probe_split),
        )
        probes = np.concatenate([below, above])
        if probes.size < 2:
            return np.nan
        if noise_model == "interpolate" and (below.size == 0 or above.size == 0):
            return np.nan

        all_freqs = np.concatenate([[float(target_freq)], probes]).astype(np.float64)
        if np.any(all_freqs <= 0.0) or np.any(all_freqs >= nyquist):
            return np.nan

        norm_freqs = all_freqs / float(fs)
        result = goertzel_sliding_batch(signal, n, norm_freqs)
        amps = np.asarray(result)[0, :, 0].astype(np.float64)
        target_amp = float(amps[0])
        probe_amps = amps[1:]
        below_amps = probe_amps[: below.size]
        above_amps = probe_amps[below.size:]

        if noise_model == "median":
            local_noise = float(np.median(probe_amps))
        elif noise_model == "interpolate":
            f_lo = float(np.mean(below))
            f_hi = float(np.mean(above))
            s_lo = float(np.median(below_amps))
            s_hi = float(np.median(above_amps))
            if s_lo <= 0 or s_hi <= 0 or f_lo <= 0 or f_hi <= 0:
                return np.nan
            slope = (np.log(s_hi) - np.log(s_lo)) / (np.log(f_hi) - np.log(f_lo))
            local_noise = float(np.exp(np.log(s_lo) + slope * (np.log(target_freq) - np.log(f_lo))))
        else:
            raise ValueError(
                f"Unknown noise_model '{noise_model}'. Choose from 'median', 'interpolate'."
            )

        if local_noise <= 0:
            return np.nan
        return float(target_amp / local_noise)
    except Exception:
        return np.nan


def compute_snr_psd(
        data: pd.DataFrame,
        target_freq: float = 16.7,
        signal_delta: float = 0.15,
        spectrum_min_freq: float = 1.0,
        spectrum_max_freq: float = 45.0,
        harmonics_mask: bool = True,
        noise_model: str = "median",
        guard_band: float = 0.5,
        noise_band: float = 3.0,
        nperseg: int = 256,
        **kwargs,
) -> float:
    """Compute the peak-to-noise-floor SNR using Welch's PSD.

    Returns ``np.nan`` on any error.
    """
    try:
        signal = data["magnitude"].to_numpy()
        timestamps = data["timestamp"].astype("int64").to_numpy() / 1e9
        dt = np.median(np.diff(timestamps))
        fs = 1.0 / dt
        nperseg_eff = min(nperseg, len(signal))
        noverlap_eff = nperseg_eff // 2
        freqs, psd = scipy.signal.welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=nperseg_eff,
            noverlap=noverlap_eff,
            detrend="constant",
            scaling="density",
        )
        valid = (freqs >= spectrum_min_freq) & (freqs <= spectrum_max_freq)
        freqs = freqs[valid]
        psd = psd[valid]
        if len(freqs) == 0:
            return np.nan
        target_mask = np.abs(freqs - target_freq) <= signal_delta
        if not np.any(target_mask):
            return np.nan
        target_peak = float(np.max(psd[target_mask]))
        noise_mask = ~target_mask
        if harmonics_mask:
            for h in range(2, 4):
                noise_mask &= np.abs(freqs - target_freq * h) > signal_delta
        if np.sum(noise_mask) < 5:
            return np.nan
        local_noise = _estimate_noise_floor(
            freqs, psd, noise_mask, target_freq, noise_model, guard_band, noise_band,
        )
        if local_noise <= 0:
            return np.nan
        return float(target_peak / local_noise)
    except Exception:
        return np.nan
