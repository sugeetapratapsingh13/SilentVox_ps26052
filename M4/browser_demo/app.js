let ctx, node, stream;
const $=id=>document.getElementById(id);
async function start(){
  $('status').textContent='Requesting microphone permission…';
  stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
  ctx=new AudioContext({sampleRate:16000});
  await ctx.audioWorklet.addModule('audio_processor.js');
  const source=ctx.createMediaStreamSource(stream);
  node=new AudioWorkletNode(ctx,'silentvox-m4-processor');
  node.port.onmessage=e=>{ if(e.data.type==='block') $('status').textContent=`AudioWorklet running — processed ${e.data.count} blocks.`; };
  source.connect(node); node.connect(ctx.destination);
  $('start').disabled=true; $('stop').disabled=false;
  $('status').textContent=`Running at ${ctx.sampleRate} Hz. Browser/device may choose its own actual channel topology.`;
}
function stop(){ if(stream) stream.getTracks().forEach(t=>t.stop()); if(ctx) ctx.close(); $('start').disabled=false; $('stop').disabled=true; $('status').textContent='Stopped.'; }
$('start').onclick=()=>start().catch(e=>{$('status').textContent='Error: '+e.message;}); $('stop').onclick=stop;
