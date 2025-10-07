import type { separateTracks } from '../src/apply.ts';
import { samplesToWav, type RawAudio } from '../src/wav-utils.ts';

type TracksObject = Awaited<ReturnType<typeof separateTracks>>;

type TrackState = {
  audioContexts: Map<string, AudioContext>;
  audioBuffers: Map<string, AudioBuffer>;
  sources: Map<string, AudioBufferSourceNode>;
  gainNodes: Map<string, GainNode>;
  tracks: Map<string, TrackUIElements>;
  isPlaying: boolean;
  startTime: number;
  pauseTime: number;
  duration: number;
  animationFrameId: number | null;
  playButton: HTMLButtonElement;
  container: HTMLDivElement;
};

type TrackUIElements = {
  canvas: HTMLCanvasElement;
  volumeSlider: HTMLInputElement;
  muteButton: HTMLButtonElement;
};

let worker = new Worker('worker.js');

let fileInput = document.getElementById('fileInput') as HTMLInputElement;
let fileButton = document.getElementById('fileButton') as HTMLButtonElement;
let progress = document.getElementById('progress') as HTMLDivElement;
let activeState: TrackState | null = null;

// @ts-expect-error - gpu is not in standard Navigator type yet, but exists in browsers with WebGPU
if (!navigator.gpu) {
  document.getElementById('noWebgpu')!.style.display = 'initial';
}

function computeAcceptableAudioTypes() {
  const audio = document.createElement('audio');
  const commonTypes = [
    { ext: 'mp3', mime: 'audio/mpeg' },
    { ext: 'm4a', mime: 'audio/mp4' },
    { ext: 'aac', mime: 'audio/aac' },
    { ext: 'ogg', mime: 'audio/ogg; codecs="vorbis"' },
    { ext: 'opus', mime: 'audio/ogg; codecs="opus"' },
    { ext: 'oga', mime: 'audio/ogg; codecs="vorbis"' },
    { ext: 'webm', mime: 'audio/webm; codecs="opus"' },
    { ext: 'flac', mime: 'audio/flac' },
  ];

  const accepted = ['.wav']; // assume WAV always works

  for (const { ext, mime } of commonTypes) {
    const support = audio.canPlayType(mime);
    if (support === 'probably' || support === 'maybe') {
      accepted.push('.' + ext);
    }
  }

  return accepted.join(',');
}
fileInput.accept = computeAcceptableAudioTypes();

// this has to be done on the main thread because AudioContext isn't in workers
let sharedAudioContext: AudioContext | undefined;
async function decodeAudio(audioFileBytes: ArrayBuffer): Promise<RawAudio> {
  if (!sharedAudioContext) {
    sharedAudioContext = new AudioContext();
  }

  const audioBuffer = await sharedAudioContext.decodeAudioData(audioFileBytes);

  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;

  let channelData = Array.from({ length: numChannels }, (_i, idx) => audioBuffer.getChannelData(idx));

  return {
    channelData,
    sampleRate,
  };
}

fileButton.addEventListener('click', () => {
  fileInput.click();
});

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && activeState && (e.target as HTMLElement).tagName !== 'INPUT' && (e.target as HTMLElement).tagName !== 'BUTTON') {
    e.preventDefault();
    togglePlayPause(activeState);
  }
});

let pendingMp3Encodes = new Map<string, { button: HTMLButtonElement; spinner: HTMLElement }>();

worker.addEventListener('message', e => {
  switch (e.data.type) {
    case 'split progress': {
      let { id, fileName, step, total } = e.data;
      progress.classList.remove('error');
      progress.innerHTML = `Working... ${step}/${total}`;
      break;
    }
    case 'split done': {
      let { id, tracks, error, fileName } = e.data;
      fileButton.style.display = '';
      if (error) {
        console.log(error);
        logError(error);
        return;
      }
      progress.innerHTML = '';
      renderTracks(tracks, fileName);
      break;
    }
    case 'encode done': {
      let { id, fileName, buffer, error } = e.data;
      let pending = pendingMp3Encodes.get(id);
      if (!pending) return;

      let { button, spinner } = pending;
      button.disabled = false;
      spinner.style.visibility = 'hidden';
      pendingMp3Encodes.delete(id);

      if (error) {
        console.log(error);
        logError(error);
        return;
      }
      downloadTrack(fileName, 'mp3', new Uint8Array(buffer));
      break;
    }
  }
});

fileInput.addEventListener('change', async () => {
  const file = fileInput.files![0];
  if (file) {
    fileButton.blur();
    fileButton.style.display = 'none';
    let id = Math.floor(Math.random() * 1e9).toString();
    progress.classList.remove('error');
    progress.innerHTML = 'Loading...';
    try {
      let rawAudio = await decodeAudio(await file.arrayBuffer());
      // why doesn't postMessage handle passing the same ArrayBuffer multiple times? idk man.
      worker.postMessage({ type: 'split', id, rawAudio, fileName: file.name }, [...new Set(rawAudio.channelData.map(c => c.buffer))]);
    } catch (e) {
      console.log(e);
      logError(e);
      fileButton.style.display = '';
    }
  }
});

function logError(e: any) {
  progress.classList.add('error');
  progress.innerText = 'Error: ' + e.message;
}

async function renderTracks(tracksObject: TracksObject, fileName: string) {
  // sort vocals/drums first
  tracksObject = {
    vocals: tracksObject.vocals,
    drums: tracksObject.drums,
    ...tracksObject,
  };

  const container = document.createElement('div');
  const playButton = document.createElement('button');
  const state: TrackState = {
    audioContexts: new Map(),
    audioBuffers: new Map(),
    sources: new Map(),
    gainNodes: new Map(),
    tracks: new Map(),
    isPlaying: false,
    startTime: 0,
    pauseTime: 0,
    duration: 0,
    animationFrameId: null,
    playButton,
    container,
  };

  container.className = 'multi-track-container';
  document.body.appendChild(container);

  if (fileName) {
    const fileNameDiv = document.createElement('div');
    fileNameDiv.className = 'file-name';
    fileNameDiv.textContent = fileName;
    container.appendChild(fileNameDiv);
  }

  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'global-controls';

  playButton.className = 'play-button';
  playButton.textContent = '▶';
  playButton.title = 'Play';
  playButton.onclick = () => togglePlayPause(state);
  controlsDiv.appendChild(playButton);
  container.appendChild(controlsDiv);

  state.playButton = playButton;
  state.container = container;
  activeState = state;

  for (let [trackName, rawAudio] of Object.entries(tracksObject)) {
    await addTrack(state, trackName, fileName, rawAudio);
  }
}

async function addTrack(state: TrackState, trackName: string, fileName: string, rawAudio: RawAudio) {
  let wavBuffer = samplesToWav(rawAudio.channelData, rawAudio.sampleRate);

  // @ts-expect-error - webkitAudioContext is non-standard but needed for old Safari
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(wavBuffer.buffer);

  if (state.duration === 0) {
    state.duration = audioBuffer.duration;
  }

  state.audioContexts.set(trackName, audioContext);
  state.audioBuffers.set(trackName, audioBuffer);

  const gainNode = audioContext.createGain();
  gainNode.connect(audioContext.destination);
  state.gainNodes.set(trackName, gainNode);

  const trackDiv = document.createElement('div');
  trackDiv.className = 'track';

  const label = document.createElement('div');
  label.className = 'track-label';
  label.textContent = trackName;
  trackDiv.appendChild(label);

  const muteButton = document.createElement('button');
  muteButton.className = 'mute-button';
  muteButton.textContent = '🔊';
  muteButton.title = 'Mute';
  muteButton.onclick = () => toggleMute(state, trackName, muteButton);
  trackDiv.appendChild(muteButton);

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.value = '100';
  volumeSlider.className = 'volume-slider';
  volumeSlider.oninput = e => setVolume(state, trackName, Number((e.target as HTMLInputElement).value) / 100);
  trackDiv.appendChild(volumeSlider);

  const canvas = document.createElement('canvas');
  canvas.className = 'waveform';
  canvas.width = 1800;
  canvas.height = 200;
  drawWaveform(canvas, audioBuffer);

  canvas.onmousedown = e => handleWaveformInteraction(state, e, canvas);
  canvas.onmousemove = e => {
    if (e.buttons === 1) {
      handleWaveformInteraction(state, e, canvas);
    }
  };

  trackDiv.appendChild(canvas);

  const downloadButton = document.createElement('button');
  downloadButton.className = 'download-button';
  downloadButton.textContent = 'Download WAV';
  let name = fileName ? fileName.split('.').slice(0, -1).join('.') + '_' + trackName : trackName;
  // can't re-use wavBuffer because decodeAudioData has detached it
  downloadButton.onclick = () => downloadTrack(name, 'wav', samplesToWav(rawAudio.channelData, rawAudio.sampleRate));
  trackDiv.appendChild(downloadButton);

  const mp3Parent = document.createElement('div');
  mp3Parent.className = 'mp3-parent';

  const downloadMp3Button = document.createElement('button');
  downloadMp3Button.className = 'download-button';
  downloadMp3Button.textContent = 'Download MP3';
  mp3Parent.appendChild(downloadMp3Button);

  const spinner = document.createElement('span');
  // spinner from https://github.com/n3r4zzurr0/svg-spinners/blob/main/svg-css/bars-rotate-fade.svg
  spinner.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><style>.spinner_OSmW{transform-origin:center;animation:spinner_T6mA .75s step-end infinite}@keyframes spinner_T6mA{8.3%{transform:rotate(30deg)}16.6%{transform:rotate(60deg)}25%{transform:rotate(90deg)}33.3%{transform:rotate(120deg)}41.6%{transform:rotate(150deg)}50%{transform:rotate(180deg)}58.3%{transform:rotate(210deg)}66.6%{transform:rotate(240deg)}75%{transform:rotate(270deg)}83.3%{transform:rotate(300deg)}91.6%{transform:rotate(330deg)}100%{transform:rotate(360deg)}}</style><g class="spinner_OSmW"><rect x="11" y="1" width="2" height="5" opacity=".14"/><rect x="11" y="1" width="2" height="5" transform="rotate(30 12 12)" opacity=".29"/><rect x="11" y="1" width="2" height="5" transform="rotate(60 12 12)" opacity=".43"/><rect x="11" y="1" width="2" height="5" transform="rotate(90 12 12)" opacity=".57"/><rect x="11" y="1" width="2" height="5" transform="rotate(120 12 12)" opacity=".71"/><rect x="11" y="1" width="2" height="5" transform="rotate(150 12 12)" opacity=".86"/><rect x="11" y="1" width="2" height="5" transform="rotate(180 12 12)"/></g></svg>`;
  spinner.style.marginLeft = '5px';
  spinner.style.visibility = 'hidden';
  mp3Parent.appendChild(spinner);

  downloadMp3Button.onclick = () => downloadMp3(name, rawAudio, downloadMp3Button, spinner);

  trackDiv.appendChild(mp3Parent);

  state.container.appendChild(trackDiv);
  state.tracks.set(trackName, { canvas, volumeSlider, muteButton });
}

function drawWaveform(canvas: HTMLCanvasElement, audioBuffer: AudioBuffer) {
  const ctx = canvas.getContext('2d')!;
  const width = canvas.width;
  const height = canvas.height;
  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / width);
  const amp = height / 2;

  canvas.width = width; // reset

  // ctx.fillStyle = '#ffffff';
  // ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#b5b5b5';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  ctx.strokeStyle = '#4a9eff';
  ctx.beginPath();

  for (let i = 0; i < width; i++) {
    let min = 1.0;
    let max = -1.0;

    for (let j = 0; j < step; j++) {
      const datum = data[i * step + j];
      if (datum < min) min = datum;
      if (datum > max) max = datum;
    }

    ctx.moveTo(i, (1 + min) * amp);
    ctx.lineTo(i, (1 + max) * amp);
  }

  ctx.stroke();
}

function updateWaveformProgress(state: TrackState) {
  if (!state.isPlaying) return;

  const currentContext = state.audioContexts.values().next().value!;
  const currentTime = Math.max(0, currentContext.currentTime - state.startTime - (currentContext.outputLatency ?? 0) / 2);
  const progress = currentTime / state.duration;

  for (let [trackName, { canvas }] of state.tracks) {
    const ctx = canvas.getContext('2d')!;
    const audioBuffer = state.audioBuffers.get(trackName)!;

    drawWaveform(canvas, audioBuffer);

    const progressWidth = canvas.width * progress;
    ctx.fillStyle = 'rgba(74, 158, 255, 0.3)';
    ctx.fillRect(0, 0, progressWidth, canvas.height);
  }

  if (currentTime < state.duration) {
    state.animationFrameId = requestAnimationFrame(() => updateWaveformProgress(state));
  } else {
    stop(state);
  }
}

function handleWaveformInteraction(state: TrackState, e: MouseEvent, canvas: HTMLCanvasElement) {
  if (e.button !== 0 || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
  activeState = state;
  const rect = canvas.getBoundingClientRect();
  const x = Math.max(0, e.clientX - rect.left);
  const progress = x / rect.width;
  const newTime = progress * state.duration;

  const wasPlaying = state.isPlaying;
  stop(state);
  state.pauseTime = newTime;

  if (wasPlaying) {
    play(state);
  } else {
    for (let [trackName, { canvas }] of state.tracks) {
      const ctx = canvas.getContext('2d')!;
      const audioBuffer = state.audioBuffers.get(trackName)!;
      drawWaveform(canvas, audioBuffer);
      const progressWidth = canvas.width * progress;
      ctx.fillStyle = 'rgba(74, 158, 255, 0.3)';
      ctx.fillRect(0, 0, progressWidth, canvas.height);
    }
  }
}

function togglePlayPause(state: TrackState) {
  if (state.isPlaying) {
    pause(state);
  } else {
    play(state);
  }
}

function play(state: TrackState) {
  state.isPlaying = true;
  state.playButton.textContent = '⏸';
  state.playButton.title = 'Pause';
  activeState = state;

  for (let [trackName, audioBuffer] of state.audioBuffers) {
    const audioContext = state.audioContexts.get(trackName)!;
    const gainNode = state.gainNodes.get(trackName)!;

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(gainNode);

    state.sources.set(trackName, source);
    source.start(0, state.pauseTime);
  }

  const ctx = state.audioContexts.values().next().value!;
  state.startTime = ctx.currentTime - state.pauseTime;
  updateWaveformProgress(state);
}

function pause(state: TrackState) {
  state.isPlaying = false;
  state.playButton.textContent = '▶';
  state.playButton.title = 'Play';
  activeState = state;

  const currentContext = state.audioContexts.values().next().value!;
  state.pauseTime = currentContext.currentTime - state.startTime;

  stop(state);
}

function stop(state: TrackState) {
  state.isPlaying = false;

  for (let source of state.sources.values()) {
    try {
      source.stop();
    } catch (e) {
      // Already stopped
    }
  }

  state.sources.clear();

  if (state.animationFrameId) {
    cancelAnimationFrame(state.animationFrameId);
  }

  const currentContext = state.audioContexts.values().next().value!;
  const played = currentContext.currentTime - state.startTime;
  if (played >= state.duration) {
    state.pauseTime = 0;
    state.playButton.textContent = '▶';
    state.playButton.title = 'Play';
  }
}

function setVolume(state: TrackState, trackName: string, volume: number) {
  activeState = state;
  const gainNode = state.gainNodes.get(trackName)!;
  gainNode.gain.value = volume;
}

function toggleMute(state: TrackState, trackName: string, button: HTMLButtonElement) {
  activeState = state;
  const gainNode = state.gainNodes.get(trackName)!;

  if (gainNode.gain.value > 0) {
    gainNode.gain.value = 0;
    button.textContent = '🔇';
    button.title = 'Unmute';
  } else {
    const volumeSlider = state.tracks.get(trackName)!.volumeSlider;
    gainNode.gain.value = Number(volumeSlider.value) / 100;
    button.textContent = '🔊';
    button.title = 'Mute';
  }
}

function downloadTrack(name: string, ext: string, uint8Array: Uint8Array<ArrayBuffer>) {
  const mimeType = ext === 'mp3' ? 'audio/mpeg' : 'audio/wav';
  const blob = new Blob([uint8Array], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadMp3(fileName: string, rawAudio: RawAudio, button: HTMLButtonElement, spinner: HTMLElement) {
  let id = Math.floor(Math.random() * 1e9).toString();

  button.disabled = true;
  spinner.style.visibility = '';

  pendingMp3Encodes.set(id, { button, spinner });

  // deliberately not transfering the buffers so we can use them elsewhere
  worker.postMessage({ type: 'mp3 encode', id, fileName, rawAudio });
}
