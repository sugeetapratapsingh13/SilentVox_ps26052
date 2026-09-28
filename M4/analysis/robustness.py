import csv
import numpy as np
from audio_pipeline.io import load_audio
from audio_pipeline.hardware_sim import adc_simulate, delay_samples, lowpass, dac_simulate, clip
from analysis.metrics import rms, peak, crest_factor, snr_db
from config import DIRS, INPUT_SAMPLE_RATE

x, _ = load_audio(DIRS['virtual_mics'] / 'reference.wav', target_sr=INPUT_SAMPLE_RATE, mono=True)
x = x[:, 0]
rows = []
for delay_ms in (1, 5, 10):
    y = delay_samples(x, int(delay_ms * INPUT_SAMPLE_RATE / 1000))
    rows.append({'test':'timing_delay','parameter':delay_ms,'value':rms(x-y),'unit':'RMS difference','interpretation':'software-simulated reference delay'})
for gain in (0.85, 1.0, 1.1):
    y = adc_simulate(x, gain=gain); rows.append({'test':'gain_mismatch','parameter':gain,'value':peak(y),'unit':'peak','interpretation':'software-simulated channel gain'})
for bits in (8, 12, 16, 24):
    y = adc_simulate(x, bits=bits); rows.append({'test':'quantization','parameter':bits,'value':rms(x-y),'unit':'RMS error','interpretation':'software quantization model'})
for cutoff in (3000, 6000, 7900):
    y = lowpass(x, INPUT_SAMPLE_RATE, cutoff); rows.append({'test':'frequency_limitation','parameter':cutoff,'value':rms(x-y),'unit':'RMS difference','interpretation':'software low-pass model'})
for noise in (0.001, 0.01, 0.03):
    y = adc_simulate(x, noise_rms=noise); rows.append({'test':'additive_noise','parameter':noise,'value':snr_db(x,y-x),'unit':'digital test-stream SNR dB','interpretation':'software noise injection'})
clipped = clip(x * 1.8, limit=0.8)
rows.append({'test':'clipping','parameter':0.8,'value':peak(clipped),'unit':'peak after clip','interpretation':'software saturation model'})
dac = dac_simulate(x, bits=16)
rows.append({'test':'dac_simulation','parameter':16,'value':rms(x-dac),'unit':'RMS reconstruction error','interpretation':'digital reconstruction-side model'})
with open(DIRS['results']/ 'robustness_tests.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('Wrote robustness_tests.csv')
