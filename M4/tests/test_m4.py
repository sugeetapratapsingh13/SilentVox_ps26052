import unittest
from pathlib import Path
import numpy as np
from audio_pipeline.hardware_sim import quantize, delay_samples, adc_simulate, dac_simulate
from audio_pipeline.pipeline import AudioConfig, VirtualAudioInterface
from config import SAMPLE_RATE, BLOCK_SIZES, LOGICAL_CHANNELS

class TestM4(unittest.TestCase):
    def test_quantization_range(self):
        x=np.linspace(-1,1,1000); y=quantize(x,16); self.assertTrue(np.max(y)<=1 and np.min(y)>=-1)
    def test_delay(self):
        x=np.ones(10); y=delay_samples(x,3); self.assertTrue(np.all(y[:3]==0)); self.assertEqual(len(y),10)
    def test_adc(self):
        x=np.array([-1.2,-.5,0,.5,1.2],dtype=float); y=adc_simulate(x); self.assertTrue(np.all(np.abs(y)<=1))
    def test_dac(self):
        x=np.array([-1.2,-.5,0,.5,1.2],dtype=float); y=dac_simulate(x); self.assertTrue(np.all(np.abs(y)<=1))
    def test_pipeline(self):
        n=1024; streams={c:np.zeros(n,dtype=np.float32) for c in LOGICAL_CHANNELS}; out=VirtualAudioInterface(AudioConfig()).process(streams); self.assertEqual(set(out),set(streams)); self.assertEqual(len(out['REF']),n)
    def test_config(self):
        self.assertEqual(SAMPLE_RATE,16000); self.assertEqual(BLOCK_SIZES,(256,512,1024)); self.assertEqual(LOGICAL_CHANNELS,('REF','ERR','SPCH'))

if __name__=='__main__': unittest.main()
