import numpy as np
from scipy.signal import welch

def rms(x): return float(np.sqrt(np.mean(np.square(np.asarray(x), dtype=np.float64))))
def peak(x): return float(np.max(np.abs(x)))
def crest_factor(x):
    r = rms(x); return float(peak(x)/r) if r else float("inf")
def dbfs(x):
    p = peak(x); return float(20*np.log10(max(p, 1e-12)))
def snr_db(signal, noise):
    rs, rn = rms(signal), rms(noise)
    return float(20*np.log10(max(rs,1e-12)/max(rn,1e-12)))
def fft_spectrum(x, sr):
    x=np.asarray(x); w=np.hanning(len(x)); spec=np.fft.rfft(x*w); freqs=np.fft.rfftfreq(len(x),1/sr)
    mag=20*np.log10(np.maximum(np.abs(spec)/(np.sum(w)+1e-12),1e-12)); return freqs, mag
def psd(x,sr): return welch(np.asarray(x), fs=sr, nperseg=min(2048,len(x)))
