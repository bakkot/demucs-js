export type RawAudio = { channelData: Float32Array[]; sampleRate: number };

export function wavToSamples(buffer: Uint8Array): RawAudio {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let offset = 0;

  // Helper to read ASCII string
  const readString = (offset: number, length: number): string => {
    let str = '';
    for (let i = 0; i < length; i++) {
      str += String.fromCharCode(buffer[offset + i]);
    }
    return str;
  };

  // Read RIFF header
  const riff = readString(offset, 4);
  offset += 4;
  if (riff !== 'RIFF') {
    throw new Error('Not a valid WAV file (missing RIFF header)');
  }

  const fileSize = view.getUint32(offset, true);
  offset += 4;

  const wave = readString(offset, 4);
  offset += 4;
  if (wave !== 'WAVE') {
    throw new Error('Not a valid WAV file (missing WAVE header)');
  }

  // Read fmt chunk
  let fmtChunk = readString(offset, 4);
  offset += 4;

  // Skip any non-fmt chunks
  while (fmtChunk !== 'fmt ') {
    const chunkSize = view.getUint32(offset, true);
    offset += 4;
    offset += chunkSize;
    fmtChunk = readString(offset, 4);
    offset += 4;
  }

  const fmtSize = view.getUint32(offset, true);
  offset += 4;

  const audioFormat = view.getUint16(offset, true);
  offset += 2;

  const numChannels = view.getUint16(offset, true);
  offset += 2;

  const sampleRate = view.getUint32(offset, true);
  offset += 4;

  const byteRate = view.getUint32(offset, true);
  offset += 4;

  const blockAlign = view.getUint16(offset, true);
  offset += 2;

  const bitsPerSample = view.getUint16(offset, true);
  offset += 2;

  // Skip any extra fmt bytes
  offset += fmtSize - 16;

  // Find data chunk
  let dataChunk = readString(offset, 4);
  offset += 4;

  while (dataChunk !== 'data') {
    const chunkSize = view.getUint32(offset, true);
    offset += 4;
    offset += chunkSize;
    if (offset >= buffer.length) {
      throw new Error('No data chunk found in WAV file');
    }
    dataChunk = readString(offset, 4);
    offset += 4;
  }

  const dataSize = view.getUint32(offset, true);
  offset += 4;

  // Read audio data
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataSize / (numChannels * bytesPerSample));

  // Create single buffer to hold all channel data
  const buffer_data = new Float32Array(numChannels * numSamples);

  // Convert samples to float32 [-1, 1]
  for (let s = 0; s < numSamples; s++) {
    for (let c = 0; c < numChannels; c++) {
      let sample;

      if (audioFormat === 1) {
        // PCM integer format
        if (bitsPerSample === 16) {
          sample = view.getInt16(offset, true);
          sample = sample / 32768.0;
        } else if (bitsPerSample === 24) {
          const byte1 = buffer[offset];
          const byte2 = buffer[offset + 1];
          const byte3 = buffer[offset + 2];
          // Reconstruct 24-bit signed integer
          sample = byte1 | (byte2 << 8) | (byte3 << 16);
          // Sign extend if negative
          if (sample & 0x800000) {
            sample |= ~0xffffff;
          }
          sample = sample / 8388608.0;
        } else if (bitsPerSample === 32) {
          sample = view.getInt32(offset, true);
          sample = sample / 2147483648.0;
        } else if (bitsPerSample === 8) {
          sample = buffer[offset];
          sample = (sample - 128) / 128.0;
        } else {
          throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
        }
      } else if (audioFormat === 3) {
        // IEEE float format
        if (bitsPerSample === 32) {
          sample = view.getFloat32(offset, true);
        } else if (bitsPerSample === 64) {
          sample = view.getFloat64(offset, true);
        } else {
          throw new Error(`Unsupported float bits per sample: ${bitsPerSample}`);
        }
      } else {
        throw new Error(`Unsupported audio format: ${audioFormat}`);
      }

      buffer_data[c * numSamples + s] = sample;
      offset += bytesPerSample;
    }
  }

  // Create separate Float32Array views for each channel
  const channelData: Float32Array[] = [];
  for (let c = 0; c < numChannels; c++) {
    channelData.push(new Float32Array(buffer_data.buffer, c * numSamples * 4, numSamples));
  }

  return {
    channelData,
    sampleRate,
  };
}

export function samplesToWav(channelData: Float32Array[], sampleRate = 44100, asFloat = false): Uint8Array<ArrayBuffer> {
  const channels = channelData.length;
  const samples = channelData[0].length;
  const bitsPerSample = asFloat ? 32 : 16;
  const bytesPerSample = bitsPerSample / 8;
  const audioFormat = asFloat ? 3 : 1; // 3 = IEEE float, 1 = PCM

  const dataSize = samples * channels * bytesPerSample;
  const fileSize = 36 + dataSize;

  const buffer = new Uint8Array(44 + dataSize);
  const view = new DataView(buffer.buffer);
  let offset = 0;

  const writeString = (str: string, offset: number): void => {
    for (let i = 0; i < str.length; i++) {
      buffer[offset + i] = str.charCodeAt(i);
    }
  };

  // RIFF header
  writeString('RIFF', offset);
  offset += 4;
  view.setUint32(offset, fileSize, true);
  offset += 4;
  writeString('WAVE', offset);
  offset += 4;

  // fmt chunk
  writeString('fmt ', offset);
  offset += 4;
  view.setUint32(offset, 16, true); // fmt chunk size
  offset += 4;
  view.setUint16(offset, audioFormat, true);
  offset += 2;
  view.setUint16(offset, channels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, sampleRate * channels * bytesPerSample, true); // byte rate
  offset += 4;
  view.setUint16(offset, channels * bytesPerSample, true); // block align
  offset += 2;
  view.setUint16(offset, bitsPerSample, true);
  offset += 2;

  // data chunk
  writeString('data', offset);
  offset += 4;
  view.setUint32(offset, dataSize, true);
  offset += 4;

  // Write audio data (interleaved)
  for (let s = 0; s < samples; s++) {
    for (let c = 0; c < channels; c++) {
      const sample = channelData[c][s];
      if (asFloat) {
        view.setFloat32(offset, sample, true);
      } else {
        // Convert float [-1, 1] to int16 [-32768, 32767]
        const clampedSample = Math.max(-1, Math.min(1, sample));
        const int16Sample = Math.round(clampedSample * 32767);
        view.setInt16(offset, int16Sample, true);
      }
      offset += bytesPerSample;
    }
  }

  return buffer;
}export function planarize(channelData: Float32Array[]): Float32Array {
  const channels = channelData.length;
  const samples = channelData[0].length;

  if (channels === 1) {
    return channelData[0];
  } else {
    let isSequential = true;
    for (let c = 1; c < channels; c++) {
      if (channelData[c].buffer !== channelData[0].buffer ||
        channelData[c].byteOffset !== channelData[c - 1].byteOffset + channelData[c - 1].length * 4) {
        isSequential = false;
        break;
      }
    }

    if (isSequential) {
      // Channels are sequential views of the same buffer, create a single view
      return new Float32Array(channelData[0].buffer, channelData[0].byteOffset, channels * samples);
    } else {
      // Channels are separate buffers, concatenate them
      let data = new Float32Array(channels * samples);
      for (let c = 0; c < channels; c++) {
        data.set(channelData[c], c * samples);
      }
      return data;
    }
  }
}

