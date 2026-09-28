from pathlib import Path
import csv
from audio_pipeline.io import load_audio
from audio_pipeline.pipeline import AudioConfig, VirtualAudioInterface
from config import DIRS, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, BLOCK_SIZES, DEFAULT_BLOCK_SIZE, BENCHMARK_TRIALS

streams = {}
for c, name in (("REF", "reference.wav"), ("ERR", "error.wav"), ("SPCH", "speech.wav")):
    a, sr = load_audio(DIRS["virtual_mics"] / name, target_sr=INPUT_SAMPLE_RATE, mono=True)
    streams[c] = a[:, 0]

rows = []
for bs in BLOCK_SIZES:
    cfg = AudioConfig(input_sample_rate=INPUT_SAMPLE_RATE, output_sample_rate=OUTPUT_SAMPLE_RATE, block_size=bs)
    rows.append(VirtualAudioInterface(cfg).benchmark(streams, bs, trials=BENCHMARK_TRIALS))

out = DIRS["results"] / "block_latency_benchmark.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

print("M4 SOFTWARE/VIRTUAL AUDIO BENCHMARK")
for r in rows:
    print(f"[PASS] block={r['block_size']} frame={r['frame_duration_ms']:.3f} ms; processing mean={r['processing_time_ms_mean']:.4f} ms; RTF={r['real_time_factor']:.1f}x")
print(f"Wrote {out}")
