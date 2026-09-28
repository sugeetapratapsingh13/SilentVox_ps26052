import csv
from pathlib import Path
import numpy as np
from audio_pipeline.signals import write_demo_signals
from audio_pipeline.io import load_audio, save_audio
from audio_pipeline.pipeline import AudioConfig, VirtualAudioInterface
from .analyze_audio import analyze_file
from .metrics import rms, snr_db, peak, crest_factor, dbfs
from config import DIRS, INPUT_SAMPLE_RATE, DEFAULT_BLOCK_SIZE, SYNCHRONIZATION_TOLERANCE_MS

if __name__ == '__main__':
    write_demo_signals(DIRS['virtual_mics'], INPUT_SAMPLE_RATE)
    rows = []
    for p in sorted(DIRS['virtual_mics'].glob('*.wav')):
        rows.append(analyze_file(p, DIRS['plots'], INPUT_SAMPLE_RATE))

    # Condition-level digital SNR: speech_clean is treated as the signal and
    # the difference between speech_plus_noise and speech_clean as the noise.
    clean, _ = load_audio(DIRS['virtual_mics'] / 'speech_clean.wav', target_sr=INPUT_SAMPLE_RATE, mono=True)
    mixed, _ = load_audio(DIRS['virtual_mics'] / 'speech_plus_noise.wav', target_sr=INPUT_SAMPLE_RATE, mono=True)
    noise = mixed[:, 0] - clean[:, 0]
    digital_snr = snr_db(clean[:, 0], noise)
    rows.append({
        'file': 'speech_plus_noise_condition', 'sample_rate': INPUT_SAMPLE_RATE,
        'samples': len(clean), 'duration_s': len(clean)/INPUT_SAMPLE_RATE,
        'rms': rms(mixed[:,0]), 'peak': peak(mixed[:,0]),
        'crest_factor': crest_factor(mixed[:,0]), 'peak_dbfs': dbfs(mixed[:,0]),
        'condition': 'signal=speech_clean; noise=speech_plus_noise-speech_clean',
        'snr_db': digital_snr, 'noise_rms': rms(noise),
    })

    # Virtual interface + output-path simulation. This is not ANC performance.
    streams = {}
    for c, name in (("REF", 'reference.wav'), ('ERR', 'error.wav'), ('SPCH', 'speech.wav')):
        a, _ = load_audio(DIRS['virtual_mics']/name, target_sr=INPUT_SAMPLE_RATE, mono=True)
        streams[c] = a[:,0]
    vif = VirtualAudioInterface(AudioConfig(block_size=DEFAULT_BLOCK_SIZE))
    processed = vif.process(streams)
    out = vif.simulate_output(processed['SPCH'], gain=0.9)
    save_audio(DIRS['results'] / 'output_simulation.wav', out, INPUT_SAMPLE_RATE)

    with open(DIRS['results']/ 'audio_metrics.csv','w',newline='',encoding='utf-8') as f:
        fieldnames = sorted(set().union(*(r.keys() for r in rows)))
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

    (DIRS['results'] / 'analysis_summary.md').write_text(
        '# M4 Digital Audio Analysis Summary\n\n'
        f'- Sample rate: {INPUT_SAMPLE_RATE} Hz\n'
        f'- Default block size: {DEFAULT_BLOCK_SIZE} samples ({DEFAULT_BLOCK_SIZE/INPUT_SAMPLE_RATE*1000:.1f} ms frame duration)\n'
        f'- Synchronization tolerance used for documentation: {SYNCHRONIZATION_TOLERANCE_MS} ms (software acceptance parameter, not a physical measurement)\n'
        f'- Digital speech-plus-noise test SNR: {digital_snr:.3f} dB\n'
        f'- Digital noise RMS: {rms(noise):.6f} normalized units\n'
        '- Provenance: reproducible synthetic/generated WAV test streams.\n'
        '- The SNR above is a digital test-stream result, not microphone SNR.\n'
        '- output_simulation.wav is a software playback-path demonstration, not a physical DAC measurement.\n', encoding='utf-8')
    print('Generated virtual microphone WAVs, analysis plots, audio_metrics.csv, and output_simulation.wav')
