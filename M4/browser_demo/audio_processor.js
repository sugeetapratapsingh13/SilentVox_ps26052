class SilentVoxProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.blockCounter = 0; }
  process(inputs, outputs) {
    const input = inputs[0]; const output = outputs[0];
    if (!input || input.length === 0) return true;
    for (let ch = 0; ch < output.length; ch++) {
      const src = input[ch] || input[0]; const dst = output[ch];
      if (!src) continue;
      for (let i = 0; i < dst.length; i++) dst[i] = src[i] || 0;
    }
    this.blockCounter++;
    if (this.blockCounter % 20 === 0) this.port.postMessage({type:'block', count:this.blockCounter});
    return true;
  }
}
registerProcessor('silentvox-m4-processor', SilentVoxProcessor);
