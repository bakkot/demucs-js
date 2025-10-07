/// <reference lib="webworker" />

import { separateTracks, type ProgressCallback } from '../src/apply.ts';
import { ONNXHTDemucs } from '../src/onnx-htdemucs.ts';
import { encode as encodeMP3 } from '../src/mp3-encoder.ts';
import type { RawAudio } from '../src/wav-utils.ts';

let model: ONNXHTDemucs;

let active = false;

addEventListener('message', async e => {
  switch (e.data.type) {
    case 'split': {
      let { id, fileName, rawAudio } = e.data;
      let progress = (step: number, total: number) => {
        postMessage({ type: 'split progress', id, fileName, step, total });
      };
      try {
        let tracks = await split(rawAudio, progress);
        // we're going to rely on the implementation detail that each track from separateTracks has all of its Float32Arrays backed by the same buffer
        postMessage({ type: 'split done', id, fileName, tracks }, [...new Set(Object.values(tracks).map(t => t.channelData[0].buffer))]);
      } catch (error) {
        console.log(error);
        postMessage({ type: 'split done', id: id, error });
      }
      return;
    }
    case 'mp3 encode': {
      let { id, fileName, rawAudio } = e.data;
      try {
        let buffer = await encodeMP3(rawAudio);
        postMessage({ type: 'encode done', id, fileName, buffer }, [buffer]);
      } catch (error) {
        console.log(error);
        postMessage({ type: 'encode done', id, error });
      }
      return;
    }
  }
});

async function split(rawAudio: RawAudio, progress: ProgressCallback) {
  if (active) {
    // TODO maybe just cancel?
    throw new Error('already splitting a track');
  }
  active = true;
  // TODO controller isn't actually wired up to anything
  // separateTracks could support aborting between passes of the model, but doesn't
  let controller = new AbortController;
  try {
    if (!model) {
      let cache = await caches.open('demucs-weights-cache');
      let request = new Request('./htdemucs.onnx');
      let cachedResponse = await cache.match(request);

      if (cachedResponse) {
        let weights = await cachedResponse.arrayBuffer();
        model = await ONNXHTDemucs.init(weights);
      } else {
        let response = await fetch(request, { signal: controller.signal });
        let weights = await response.arrayBuffer();
        model = await ONNXHTDemucs.init(weights);
        await cache.put(request, new Response(weights));
      }
    }
    return await separateTracks(model, rawAudio, progress);
  } finally {
    active = false;
  }
}
