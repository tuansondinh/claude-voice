"""aec.py — Acoustic Echo Cancellation module.

Implements a partitioned-block frequency-domain LMS (PBFDLMS) adaptive filter
for real-time echo cancellation. The known TTS reference signal is subtracted
from the microphone input, leaving only the user's voice.

Public API
----------
EchoCanceller
    Adaptive filter using PBFDLMS. Call cancel(mic_chunk, ref_chunk) per frame.

ReferenceBuffer
    Lock-free SPSC ring buffer for the TTS reference signal. Handles 24kHz→16kHz
    resampling on write.

estimate_delay(mic, ref, max_delay) -> int
    Cross-correlation based acoustic delay estimator.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz — processing rate
CHUNK_SAMPLES = 512           # samples per processing frame at 16kHz
FILTER_LENGTH = 4800          # ~300ms at 16kHz
SILENCE_POWER_THRESHOLD = 1e-7  # below this → reference is silent


# ---------------------------------------------------------------------------
# stderr logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[lazy-claude aec] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Delay estimator
# ---------------------------------------------------------------------------

def estimate_delay(
    mic: np.ndarray,
    ref: np.ndarray,
    max_delay: int = 1024,
) -> int:
    """Estimate acoustic delay between mic and reference via cross-correlation.

    Parameters
    ----------
    mic:
        Microphone signal (16kHz float32).
    ref:
        Reference (TTS output) signal (16kHz float32).
    max_delay:
        Maximum delay in samples to search.

    Returns
    -------
    int
        Estimated delay in samples (non-negative). The mic signal is delayed
        by this many samples relative to the reference.
    """
    n = len(mic)
    if n == 0 or len(ref) == 0:
        return 0

    # Use FFT-based cross-correlation for efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size <<= 1

    mic_f = np.fft.rfft(mic.astype(np.float64), n=fft_size)
    ref_f = np.fft.rfft(ref.astype(np.float64), n=fft_size)

    # Cross-correlation: xcorr[k] = sum_n mic[n] * ref[n - k]
    xcorr_f = mic_f * np.conj(ref_f)
    xcorr = np.fft.irfft(xcorr_f, n=fft_size)

    # Search positive lags only (mic is delayed relative to ref)
    search_len = min(max_delay + 1, fft_size // 2)
    positive_lags = xcorr[:search_len]
    delay = int(np.argmax(np.abs(positive_lags)))
    return delay


# ---------------------------------------------------------------------------
# Low-pass FIR filter for anti-alias resampling
# ---------------------------------------------------------------------------

def _design_lowpass_fir(cutoff_normalized: float, num_taps: int = 31) -> np.ndarray:
    """Design a simple windowed-sinc low-pass FIR filter.

    Parameters
    ----------
    cutoff_normalized:
        Cutoff frequency as a fraction of Nyquist (0 < cutoff < 1).
    num_taps:
        Filter length (odd preferred).
    """
    num_taps = num_taps if num_taps % 2 == 1 else num_taps + 1
    M = num_taps - 1
    n = np.arange(num_taps)
    # Sinc function
    h = np.sinc(2 * cutoff_normalized * (n - M / 2))
    # Hamming window
    window = np.hamming(num_taps)
    h = h * window
    h = h / np.sum(h)
    return h.astype(np.float32)


# ---------------------------------------------------------------------------
# ReferenceBuffer — lock-free SPSC ring buffer
# ---------------------------------------------------------------------------

class ReferenceBuffer:
    """Lock-free single-producer single-consumer ring buffer for TTS reference.

    The producer (TTS thread) writes at write_sr (e.g. 24kHz).
    The consumer (audio callback) reads at read_sr (16kHz).

    Resampling is done on write using linear interpolation after anti-alias
    low-pass filtering when downsampling.

    Parameters
    ----------
    capacity:
        Buffer capacity in samples (at read_sr). Must be a power of two.
    write_sr:
        Sample rate of incoming data (e.g. 24000 for TTS output).
    read_sr:
        Sample rate for output (16000 for mic pipeline).
    """

    def __init__(
        self,
        capacity: int = 16384,
        write_sr: int = 24_000,
        read_sr: int = 16_000,
    ) -> None:
        # Round up capacity to power of two for fast modulo
        cap = 1
        while cap < capacity:
            cap <<= 1
        self._capacity = cap
        self._mask = cap - 1
        self._buffer = np.zeros(cap, dtype=np.float32)
        # Indices stored as Python ints — GIL ensures atomic read/write on CPython
        self._write_idx: int = 0
        self._read_idx: int = 0

        self._write_sr = write_sr
        self._read_sr = read_sr
        self._ratio = read_sr / write_sr  # conversion ratio

        # Anti-alias FIR filter state (only needed when downsampling)
        self._do_filter = write_sr > read_sr
        if self._do_filter:
            cutoff = (read_sr / 2.0) / (write_sr / 2.0) * 0.9  # 90% of Nyquist
            self._lpf = _design_lowpass_fir(cutoff, num_taps=31)
            self._lpf_zi = np.zeros(len(self._lpf) - 1, dtype=np.float32)

        # Fractional sample accumulator for non-integer ratio resampling
        self._frac_acc: float = 0.0

    # ------------------------------------------------------------------
    # Producer side (TTS thread)
    # ------------------------------------------------------------------

    def write(self, samples: np.ndarray) -> None:
        """Resample and write samples (at write_sr) into the buffer.

        Parameters
        ----------
        samples:
            Audio at write_sr (float32 1-D).
        """
        samples = np.asarray(samples, dtype=np.float32)

        # Anti-alias filter before downsampling
        if self._do_filter:
            from numpy import convolve  # noqa: PLC0415
            filtered = np.convolve(samples, self._lpf, mode="full")
            # Keep causal output (trim extra samples from filter delay)
            delay = len(self._lpf) // 2
            filtered = filtered[delay: delay + len(samples)]
        else:
            filtered = samples

        # Resample using linear interpolation
        resampled = self._resample(filtered)

        # Write resampled data into the ring buffer
        n = len(resampled)
        write_idx = self._write_idx
        for i in range(n):
            self._buffer[(write_idx + i) & self._mask] = resampled[i]
        # Atomic-style: update write index after all data is written
        self._write_idx = (write_idx + n) & self._mask

    def _resample(self, data: np.ndarray) -> np.ndarray:
        """Resample data from write_sr to read_sr via linear interpolation."""
        if self._write_sr == self._read_sr:
            return data

        n_in = len(data)
        if n_in == 0:
            return data

        # Build input positions array at write_sr, output at read_sr
        # We track a fractional accumulator so we don't lose sub-sample alignment
        x_in = np.arange(n_in, dtype=np.float64)
        # Output positions in input sample coordinates
        # Each output sample steps by (write_sr / read_sr) input samples
        step = self._write_sr / self._read_sr
        # Number of output samples
        n_out = int((n_in - self._frac_acc) / step)
        if n_out <= 0:
            # Update fractional accumulator and return empty
            self._frac_acc += n_in / (self._write_sr / self._read_sr)
            # Clamp to avoid unbounded growth
            if self._frac_acc >= n_in:
                self._frac_acc = 0.0
            return np.array([], dtype=np.float32)

        x_out = self._frac_acc + np.arange(n_out, dtype=np.float64) * step
        resampled = np.interp(x_out, x_in, data.astype(np.float64)).astype(np.float32)

        # Update fractional accumulator: where did the next output sample start?
        next_pos = self._frac_acc + n_out * step
        self._frac_acc = next_pos - n_in  # fractional remainder
        if self._frac_acc < 0:
            self._frac_acc = 0.0

        return resampled

    # ------------------------------------------------------------------
    # Consumer side (audio callback)
    # ------------------------------------------------------------------

    def read(self, n: int) -> np.ndarray:
        """Read n samples at read_sr from the buffer.

        Returns zeros (silence) for any samples not yet available.
        Never raises — safe to call from a PortAudio callback.
        """
        out = np.zeros(n, dtype=np.float32)
        read_idx = self._read_idx
        write_idx = self._write_idx

        # Available samples (SPSC: read_idx only written by consumer,
        # write_idx only written by producer)
        available = (write_idx - read_idx) & self._mask
        to_read = min(n, available)

        for i in range(to_read):
            out[i] = self._buffer[(read_idx + i) & self._mask]

        # Atomic-style: update read index after consuming
        self._read_idx = (read_idx + to_read) & self._mask
        return out

    def available(self) -> int:
        """Return the number of samples available to read."""
        return (self._write_idx - self._read_idx) & self._mask


# ---------------------------------------------------------------------------
# PBFDLMS — Partitioned-Block Frequency-Domain LMS
# ---------------------------------------------------------------------------

class _PBFDLMS:
    """Partitioned-Block Frequency-Domain LMS (NLMS) adaptive filter.

    Uses overlap-save with FFT blocks. The filter is partitioned into
    P = ceil(filter_length / block_size) partitions.

    Implementation follows Shynk (1992) with the constrained FDLMS
    (gradient causal constraint) and per-block power normalization.

    Parameters
    ----------
    filter_length:
        Total filter length in samples.
    block_size:
        Processing block size in samples (must match chunk size).
    mu:
        LMS step size (learning rate). Smaller = more stable, slower convergence.
    """

    def __init__(
        self,
        filter_length: int = FILTER_LENGTH,
        block_size: int = CHUNK_SAMPLES,
        mu: float = 0.01,
    ) -> None:
        self.filter_length = filter_length
        self.block_size = block_size
        self.mu = mu

        # Number of partitions
        self.P = max(1, (filter_length + block_size - 1) // block_size)
        # FFT size = 2 * block_size (overlap-save)
        B = block_size
        self.fft_size = 2 * B
        self.n_freq = self.fft_size // 2 + 1

        # Filter coefficients in frequency domain: shape (P, n_freq) complex128
        self.W = np.zeros((self.P, self.n_freq), dtype=np.complex128)

        # Reference FFT history: circular buffer of P most-recent frames
        # Each frame is the FFT of [prev_B | cur_B] (length 2B)
        self._X_hist = np.zeros((self.P, self.n_freq), dtype=np.complex128)
        # Pointer to next write slot
        self._ptr: int = 0

        # Overlap-save: keep the last B reference samples
        self._ref_buf = np.zeros(B, dtype=np.float64)

        # Per-bin smoothed input power for frequency-domain NLMS normalization.
        # Initialised to a small positive value; warms up after a few frames.
        self._bin_power = np.ones(self.n_freq, dtype=np.float64) * 0.01
        self._power_alpha: float = 0.9

        # Scalar reference power tracker (used for silence detection)
        self._input_power: float = 0.01

    def _push_ref(self, ref_block: np.ndarray) -> np.ndarray:
        """Push a reference block, compute its FFT frame, return the FFT.

        The FFT frame is [prev_B | cur_B], length fft_size.
        Stores result in X_hist at current pointer.
        """
        B = self.block_size
        ref_d = ref_block.astype(np.float64)
        frame = np.concatenate([self._ref_buf, ref_d])  # length 2B
        X = np.fft.rfft(frame)  # length n_freq
        self._X_hist[self._ptr] = X
        self._ptr = (self._ptr + 1) % self.P
        self._ref_buf = ref_d.copy()
        return X

    def _get_hist(self, lag: int) -> np.ndarray:
        """Get X_hist at lag samples back (lag=0 = most recent)."""
        idx = (self._ptr - 1 - lag) % self.P
        return self._X_hist[idx]

    def filter(
        self, mic_block: np.ndarray, ref_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute echo estimate and return (error, X_k).

        Parameters
        ----------
        mic_block:
            Microphone input (float32, length block_size).
        ref_block:
            Reference input (float32, length block_size).

        Returns
        -------
        (error, X_k)
            error: residual = mic - echo_estimate (float32, length block_size)
            X_k: current reference FFT (complex128, length n_freq) — pass to update()
        """
        B = self.block_size
        N = self.fft_size

        # Push reference and get current FFT
        X_k = self._push_ref(ref_block)

        # Echo estimate: sum over partitions
        # Most recent partition uses X_k (lag 0), oldest uses X_{lag=P-1}
        Y = np.zeros(self.n_freq, dtype=np.complex128)
        for p in range(self.P):
            Y += self.W[p] * self._get_hist(p)

        # Overlap-save: take last B samples of IFFT
        y_time = np.fft.irfft(Y, n=N)
        echo_est = y_time[B:].astype(np.float32)  # valid output samples

        error = mic_block.astype(np.float32) - echo_est

        # Update per-bin power estimate and scalar power
        x_power_bins = np.abs(X_k) ** 2  # shape (n_freq,)
        self._bin_power = (
            self._power_alpha * self._bin_power
            + (1.0 - self._power_alpha) * x_power_bins
        )
        ref_power = float(np.mean(ref_block.astype(np.float64) ** 2))
        self._input_power = (
            self._power_alpha * self._input_power
            + (1.0 - self._power_alpha) * ref_power
        )

        return error, X_k

    def update(self, error: np.ndarray, X_k: np.ndarray) -> None:
        """Update filter weights with the constrained PBFDLMS gradient.

        Parameters
        ----------
        error:
            Residual signal (float32, length block_size).
        X_k:
            Current reference FFT (not used; retained for API compatibility).
        """
        B = self.block_size
        N = self.fft_size

        # Per-bin NLMS normalisation: each frequency bin has its own step size.
        # This guarantees convergence for mu in (0, 2) regardless of the input
        # spectrum. The per-bin denominator is:
        #   denom_k = P * sigma_k^2 + eps
        # where sigma_k^2 is the smoothed power in bin k (from filter()).
        denom = self.P * self._bin_power + 1e-8  # shape (n_freq,)
        mu_k = self.mu / denom  # per-bin step, shape (n_freq,)

        # Error FFT: zero-pad first half (overlap-save causal constraint)
        error_d = error.astype(np.float64)
        e_pad = np.concatenate([np.zeros(B, dtype=np.float64), error_d])
        E = np.fft.rfft(e_pad, n=N)

        for p in range(self.P):
            X_p = self._get_hist(p)

            # Unconstrained gradient in frequency domain (per-bin normalised)
            G_unc = mu_k * E * np.conj(X_p)

            # Constrain to causal filter (first B taps only):
            # zero out the last B samples (anticausal / aliased portion)
            g_time = np.fft.irfft(G_unc, n=N)
            g_time[B:] = 0.0
            G = np.fft.rfft(g_time, n=N)

            self.W[p] += G

    def filter_norm(self) -> float:
        """Return L2 norm of all filter coefficients (diagnostic)."""
        return float(np.sqrt(np.sum(np.abs(self.W) ** 2)))


# ---------------------------------------------------------------------------
# Geigel Double-Talk Detector
# ---------------------------------------------------------------------------

class _GeigelDTD:
    """Geigel double-talk detector.

    Fires when the microphone power exceeds a fraction of the maximum
    reference power in a recent window — indicating user speech is present.

    Parameters
    ----------
    threshold:
        DTD threshold (Geigel parameter, typically 0.5). When
        mic_power > threshold * max_ref_power → double-talk detected.
    window_frames:
        Number of frames to track the maximum reference power.
    """

    def __init__(self, threshold: float = 0.5, window_frames: int = 8) -> None:
        self.threshold = threshold
        self.window_frames = window_frames
        self._ref_power_history: list[float] = [0.0] * window_frames
        self._ptr = 0

    def detect(self, mic_block: np.ndarray, ref_block: np.ndarray) -> bool:
        """Return True if double-talk is detected.

        Parameters
        ----------
        mic_block, ref_block:
            Current mic and reference frames (float32).
        """
        mic_power = float(np.mean(mic_block.astype(np.float64) ** 2))
        ref_power = float(np.mean(ref_block.astype(np.float64) ** 2))

        # Update reference power history
        self._ref_power_history[self._ptr] = ref_power
        self._ptr = (self._ptr + 1) % self.window_frames

        max_ref_power = max(self._ref_power_history)

        # Geigel criterion: double-talk if mic louder than threshold * max_ref
        return mic_power > self.threshold * max_ref_power


# ---------------------------------------------------------------------------
# EchoCanceller — public API
# ---------------------------------------------------------------------------

class EchoCanceller:
    """Real-time acoustic echo canceller using partitioned-block FDLMS.

    The cancel() method processes one chunk (block_size samples) at a time.
    It:
    1. Runs the Geigel DTD to detect double-talk.
    2. On the first call, estimates the acoustic delay and pre-fills the
       reference buffer to align it with the microphone.
    3. Applies the PBFDLMS filter to estimate and subtract the echo.
    4. Optionally updates the filter weights (frozen during double-talk or
       when reference is silent).

    Parameters
    ----------
    filter_length:
        Total adaptive filter length in samples. Default ~300ms at 16kHz.
    mu:
        LMS step size (learning rate). Larger = faster convergence, less stable.
    dtd_threshold:
        Geigel DTD threshold. Typical range 0.3–0.7.
    chunk_size:
        Processing block size (must match the mic capture chunk size).
    """

    def __init__(
        self,
        filter_length: int = FILTER_LENGTH,
        mu: float = 0.01,
        dtd_threshold: float = 0.5,
        chunk_size: int = CHUNK_SAMPLES,
    ) -> None:
        self.filter_length = filter_length
        self.mu = mu
        self.dtd_threshold = dtd_threshold
        self.chunk_size = chunk_size

        self._filter = _PBFDLMS(
            filter_length=filter_length,
            block_size=chunk_size,
            mu=mu,
        )
        self._dtd = _GeigelDTD(threshold=dtd_threshold)

        # Delay estimation state
        self._delay_estimated: bool = False
        self._estimated_delay: int = 0
        # Delay compensation buffer (stores delayed reference samples)
        self._delay_buf = np.zeros(
            filter_length + chunk_size, dtype=np.float32
        )

        # First-call reference snapshot for delay estimation
        self._first_mic: Optional[np.ndarray] = None
        self._first_ref: Optional[np.ndarray] = None

    def cancel(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """Cancel echo from mic_chunk given the reference ref_chunk.

        Parameters
        ----------
        mic_chunk:
            Raw microphone input (16kHz float32, length chunk_size).
        ref_chunk:
            TTS reference signal (16kHz float32, length chunk_size).

        Returns
        -------
        np.ndarray
            Cleaned signal with echo subtracted (float32, length chunk_size).
        """
        mic_chunk = np.asarray(mic_chunk, dtype=np.float32)
        ref_chunk = np.asarray(ref_chunk, dtype=np.float32)

        # Validate lengths
        if len(mic_chunk) != self.chunk_size or len(ref_chunk) != self.chunk_size:
            # Passthrough if unexpected length
            return mic_chunk.copy()

        # --- Delay estimation on first call ---
        if not self._delay_estimated:
            if self._first_mic is None:
                self._first_mic = mic_chunk.copy()
                self._first_ref = ref_chunk.copy()
            else:
                # Estimate delay from accumulated first frames
                combined_mic = np.concatenate([self._first_mic, mic_chunk])
                combined_ref = np.concatenate([self._first_ref, ref_chunk])
                self._estimated_delay = estimate_delay(
                    combined_mic, combined_ref,
                    max_delay=min(self.filter_length, 2048)
                )
                self._delay_estimated = True
                _log(f"Estimated acoustic delay: {self._estimated_delay} samples "
                     f"({self._estimated_delay / SAMPLE_RATE * 1000:.1f} ms)")

        # --- Shift reference through delay compensation buffer ---
        delay = self._estimated_delay
        buf = self._delay_buf
        buf_len = len(buf)

        # Shift buffer left by chunk_size
        buf[:-self.chunk_size] = buf[self.chunk_size:]
        buf[-self.chunk_size:] = ref_chunk

        # Extract the delayed reference
        if delay >= buf_len:
            delayed_ref = np.zeros(self.chunk_size, dtype=np.float32)
        elif delay == 0:
            delayed_ref = buf[-self.chunk_size:].copy()
        else:
            start = buf_len - self.chunk_size - delay
            if start < 0:
                # Not enough history yet
                delayed_ref = np.zeros(self.chunk_size, dtype=np.float32)
            else:
                delayed_ref = buf[start: start + self.chunk_size].copy()

        # --- Reference silence check — freeze filter if ref is silent ---
        ref_power = float(np.mean(delayed_ref.astype(np.float64) ** 2))
        ref_is_silent = ref_power < SILENCE_POWER_THRESHOLD

        # --- Geigel DTD ---
        double_talk = self._dtd.detect(mic_chunk, delayed_ref)

        # --- PBFDLMS processing ---
        error, X_k = self._filter.filter(mic_chunk, delayed_ref)

        # Update filter only if:
        # - Not in double-talk
        # - Reference is not silent
        if not double_talk and not ref_is_silent:
            self._filter.update(error, X_k)

        return error

    def filter_norm(self) -> float:
        """Return the L2 norm of the current filter coefficients.

        Useful for diagnostics and divergence detection.
        """
        return self._filter.filter_norm()

    def reset(self) -> None:
        """Reset the filter state (but keep the coefficients)."""
        self._delay_estimated = False
        self._first_mic = None
        self._first_ref = None
        self._delay_buf[:] = 0.0
