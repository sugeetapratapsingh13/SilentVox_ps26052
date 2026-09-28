from dataclasses import dataclass
import time
import numpy as np
from .io import block_iter

@dataclass
class AudioConfig:
    input_sample_rate: int = 16000
    output_sample_rate: int = 16000
    block_size: int = 512
    channels: tuple = ("REF", "ERR", "SPCH")
    input_bit_depth: int = 16
    output_bit_depth: int = 16

    @property
    def sample_rate(self):
        return self.input_sample_rate

class VirtualAudioInterface:
    def __init__(self, config=None, gains=None, delays_ms=None, noise_rms=0.0, quantize_bits=None):
        self.config = config or AudioConfig()
        self.gains = gains or {c: 1.0 for c in self.config.channels}
        self.delays_ms = delays_ms or {c: 0.0 for c in self.config.channels}
        self.noise_rms = noise_rms
        self.quantize_bits = quantize_bits

    def process(self, streams):
        missing = [c for c in self.config.channels if c not in streams]
        if missing:
            raise KeyError(f"Missing logical channels: {missing}")
        n = min(len(streams[c]) for c in self.config.channels)
        out = {}
        for c in self.config.channels:
            x = np.asarray(streams[c][:n], dtype=np.float32) * self.gains.get(c, 1.0)
            delay = int(round(self.delays_ms.get(c, 0.0) * self.config.input_sample_rate / 1000))
            if delay > 0:
                x = np.concatenate([np.zeros(delay, dtype=np.float32), x[:-delay]])
            if self.noise_rms:
                seed = sum(ord(ch) for ch in c) + 42
                rng = np.random.default_rng(seed)
                x = x + rng.normal(0, self.noise_rms, len(x)).astype(np.float32)
            if self.quantize_bits:
                levels = 2**(self.quantize_bits-1)-1
                x = np.round(np.clip(x, -1, 1)*levels)/levels
            out[c] = np.clip(x, -1, 1).astype(np.float32)
        return out

    def simulate_output(self, speech_stream, gain=1.0, clip_limit=1.0):
        x = np.asarray(speech_stream, dtype=np.float32) * gain
        return np.clip(x, -clip_limit, clip_limit).astype(np.float32)

    def benchmark(self, streams, block_size=None, trials=10):
        bs = block_size or self.config.block_size
        n = min(len(streams[c]) for c in self.config.channels)
        trial_values = []
        for trial in range(trials):
            t0 = time.perf_counter(); blocks = 0
            for start in range(0, n-bs+1, bs):
                block = {c: streams[c][start:start+bs] for c in self.config.channels}
                self.process(block)
                blocks += 1
            elapsed = time.perf_counter() - t0
            trial_values.append(elapsed / blocks * 1000 if blocks else float("nan"))
        arr = np.asarray(trial_values, dtype=float)
        frame_ms = bs / self.config.input_sample_rate * 1000
        mean_ms = float(np.mean(arr))
        return {
            "block_size": bs,
            "sample_rate": self.config.input_sample_rate,
            "channels": "REF|ERR|SPCH",
            "frame_duration_ms": frame_ms,
            "processing_time_ms_mean": mean_ms,
            "processing_time_ms_min": float(np.min(arr)),
            "processing_time_ms_max": float(np.max(arr)),
            "processing_time_ms_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "buffering_frame_duration_ms": frame_ms,
            "real_time_factor": float(frame_ms / mean_ms) if mean_ms else float("inf"),
            "benchmark_type": "SOFTWARE/VIRTUAL AUDIO BENCHMARK",
            "trials": trials,
        }
