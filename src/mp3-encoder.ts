import type { RawAudio } from './wav-utils.ts';
import { planarize } from './wav-utils.ts';

import {
  AudioSampleSource,
  Output,
  BufferTarget,
  Mp3OutputFormat,
  AudioSample,
} from 'mediabunny';
import { registerMp3Encoder } from '@mediabunny/mp3-encoder';

registerMp3Encoder();

export async function encode(audio: RawAudio): Promise<ArrayBuffer> {
  let sample: AudioSample;
  try {
    const sampleSource = new AudioSampleSource({
      codec: 'mp3',
      bitrate: 256e3,
    });
    const output = new Output({
      format: new Mp3OutputFormat(),
      target: new BufferTarget(),
    });
    output.addAudioTrack(sampleSource);
    await output.start();

    sample = new AudioSample({
      data: planarize(audio.channelData),
      format: 'f32-planar',
      numberOfChannels: 2,
      sampleRate: audio.sampleRate,
      timestamp: 0,
    });
    await sampleSource.add(sample);
    await output.finalize();

    return output.target.buffer!;
  } finally {
    // @ts-expect-error it's fine
    sample?.close();
  }
}
