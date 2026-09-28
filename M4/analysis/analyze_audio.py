from pathlib import Path
import csv, numpy as np
import matplotlib.pyplot as plt
from audio_pipeline.io import load_audio
from .metrics import rms, peak, crest_factor, dbfs, fft_spectrum


def analyze_file(path, out_dir, sr=16000):
    audio, actual_sr = load_audio(path, target_sr=sr, mono=True)
    x=audio[:,0]; out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t=np.arange(len(x))/actual_sr
    plt.figure(figsize=(10,3.5)); plt.plot(t,x); plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.title(Path(path).stem+' waveform'); plt.tight_layout(); plt.savefig(out/(Path(path).stem+'_waveform.png'),dpi=140); plt.close()
    f,m=fft_spectrum(x,actual_sr)
    plt.figure(figsize=(10,3.5)); plt.plot(f,m); plt.xlim(0,actual_sr/2); plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude (dB, relative)'); plt.title(Path(path).stem+' FFT'); plt.tight_layout(); plt.savefig(out/(Path(path).stem+'_fft.png'),dpi=140); plt.close()
    plt.figure(figsize=(10,4)); plt.specgram(x,Fs=actual_sr,NFFT=512,noverlap=384,cmap='magma'); plt.ylim(0,actual_sr/2); plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)'); plt.title(Path(path).stem+' spectrogram'); plt.tight_layout(); plt.savefig(out/(Path(path).stem+'_spectrogram.png'),dpi=140); plt.close()
    return {'file':Path(path).name,'sample_rate':actual_sr,'samples':len(x),'duration_s':len(x)/actual_sr,'rms':rms(x),'peak':peak(x),'crest_factor':crest_factor(x),'peak_dbfs':dbfs(x)}
