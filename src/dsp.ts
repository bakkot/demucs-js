export type Tensor = {
  data: Float32Array;
  shape: readonly number[];
};

export type ComplexTensor = {
  real: Tensor;
  imag: Tensor;
};

/**
 * Cooley-Tukey FFT algorithm
 */
export function fft(realInput: Float32Array, imagInput: Float32Array | null = null): { real: Float32Array; imag: Float32Array } {
  const n = realInput.length;

  // Check if power of 2
  if ((n & (n - 1)) !== 0) {
    throw new Error('FFT size must be power of 2');
  }

  const real = new Float32Array(realInput);
  const imag = imagInput ? new Float32Array(imagInput) : new Float32Array(n);

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  // Cooley-Tukey decimation-in-time
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;

    for (let i = 0; i < n; i += len) {
      let wr = 1,
        wi = 0;

      for (let j = 0; j < halfLen; j++) {
        const k = i + j;
        const l = k + halfLen;

        const tr = wr * real[l] - wi * imag[l];
        const ti = wr * imag[l] + wi * real[l];

        real[l] = real[k] - tr;
        imag[l] = imag[k] - ti;
        real[k] += tr;
        imag[k] += ti;

        const wtemp = wr;
        wr = wtemp * Math.cos(angle) - wi * Math.sin(angle);
        wi = wtemp * Math.sin(angle) + wi * Math.cos(angle);
      }
    }
  }

  return { real, imag };
}

/**
 * Inverse FFT
 */
export function ifft(realInput: Float32Array, imagInput: Float32Array): { real: Float32Array; imag: Float32Array } {
  const n = realInput.length;

  // IFFT = conj(FFT(conj(x))) / n
  // Conjugate input
  const imagConj = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    imagConj[i] = -imagInput[i];
  }

  // Forward FFT
  const result = fft(realInput, imagConj);

  // Conjugate and normalize output
  for (let i = 0; i < n; i++) {
    result.real[i] /= n;
    result.imag[i] = -result.imag[i] / n;
  }

  return result;
}

export function hannWindow(n: number): Float32Array {
  const window = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
  }
  return window;
}

/**
 * Pad array with reflect mode
 * Matches PyTorch's F.pad with mode='reflect'
 */
function padReflect(arr: Float32Array, padLeft: number, padRight: number): Float32Array {
  const length = arr.length;
  const newLength = length + padLeft + padRight;
  const result = new Float32Array(newLength);

  // Copy original data
  for (let i = 0; i < length; i++) {
    result[padLeft + i] = arr[i];
  }

  // Reflect left
  for (let i = 0; i < padLeft; i++) {
    const srcIdx = padLeft - i;
    result[i] = result[srcIdx];
  }

  // Reflect right
  for (let i = 0; i < padRight; i++) {
    const dstIdx = padLeft + length + i;
    const srcIdx = padLeft + length - 2 - i;
    result[dstIdx] = result[srcIdx];
  }

  return result;
}

/**
 * Pad1d - matches Python's pad1d function
 */
export function pad1d(x: Tensor, paddings: number[], mode: 'constant' | 'reflect' = 'constant'): Tensor {
  const length = x.shape[x.shape.length - 1];
  const [paddingLeft, paddingRight] = paddings;

  let xData = x.data;
  let xShape = x.shape;
  let actualPaddingLeft = paddingLeft;
  let actualPaddingRight = paddingRight;

  // Handle reflect mode with small input
  if (mode === 'reflect') {
    const maxPad = Math.max(paddingLeft, paddingRight);
    if (length <= maxPad) {
      const extraPad = maxPad - length + 1;
      const extraPadRight = Math.min(paddingRight, extraPad);
      const extraPadLeft = extraPad - extraPadRight;

      // First pad with zeros
      const shape = [...xShape];
      shape[shape.length - 1] = length + extraPadLeft + extraPadRight;
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const tempData = new Float32Array(totalSize);

      const outerSize = totalSize / shape[shape.length - 1];
      const newLength = shape[shape.length - 1];

      for (let i = 0; i < outerSize; i++) {
        for (let j = 0; j < length; j++) {
          tempData[i * newLength + extraPadLeft + j] = xData[i * length + j];
        }
      }

      xData = tempData;
      xShape = shape;
      actualPaddingLeft = paddingLeft - extraPadLeft;
      actualPaddingRight = paddingRight - extraPadRight;
    }
  }

  const currentLength = xShape[xShape.length - 1];
  const shape = [...xShape];
  shape[shape.length - 1] = currentLength + actualPaddingLeft + actualPaddingRight;

  const totalSize = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(totalSize);

  const outerSize = totalSize / shape[shape.length - 1];
  const outputLength = shape[shape.length - 1];

  for (let i = 0; i < outerSize; i++) {
    const srcOffset = i * currentLength;
    const dstOffset = i * outputLength;

    if (mode === 'constant') {
      // Copy data with zero padding
      for (let j = 0; j < currentLength; j++) {
        out[dstOffset + actualPaddingLeft + j] = xData[srcOffset + j];
      }
    } else if (mode === 'reflect') {
      // Extract slice for this outer dimension
      const slice = xData.slice(srcOffset, srcOffset + currentLength);
      const padded = padReflect(slice, actualPaddingLeft, actualPaddingRight);
      for (let j = 0; j < outputLength; j++) {
        out[dstOffset + j] = padded[j];
      }
    }
  }

  return { data: out, shape: shape };
}

/**
 * STFT - Short-Time Fourier Transform
 * Matches PyTorch's torch.stft behavior
 */
export function stft(x: Tensor, nFft: number, hopLength: number, window: Float32Array, normalized = true, center = true, padMode: 'constant' | 'reflect' = 'reflect'): ComplexTensor {
  // x is {data, shape} where shape is [batch, length]
  const [batch, length] = x.shape;

  let inputData = x.data;
  let inputLength = length;

  // Center padding
  if (center) {
    const pad = Math.floor(nFft / 2);
    const padded = pad1d(x, [pad, pad], padMode);
    inputData = padded.data;
    inputLength = padded.shape[padded.shape.length - 1];
  }

  // Calculate number of frames
  const numFrames = Math.floor((inputLength - nFft) / hopLength) + 1;
  const numFreqs = Math.floor(nFft / 2) + 1;

  // Output: [batch, freqs, frames] complex
  const realOut = new Float32Array(batch * numFreqs * numFrames);
  const imagOut = new Float32Array(batch * numFreqs * numFrames);

  // Normalization factor
  const norm = normalized ? 1.0 / Math.sqrt(nFft) : 1.0;

  for (let b = 0; b < batch; b++) {
    for (let frame = 0; frame < numFrames; frame++) {
      const frameStart = frame * hopLength;

      // Apply window and prepare for FFT
      const frameData = new Float32Array(nFft);
      for (let i = 0; i < nFft; i++) {
        const idx = b * inputLength + frameStart + i;
        frameData[i] = inputData[idx] * window[i] * norm;
      }

      // Compute FFT
      const { real, imag } = fft(frameData);

      // Store only positive frequencies (up to Nyquist)
      for (let freq = 0; freq < numFreqs; freq++) {
        const outIdx = b * numFreqs * numFrames + freq * numFrames + frame;
        realOut[outIdx] = real[freq];
        imagOut[outIdx] = imag[freq];
      }
    }
  }

  return {
    real: { data: realOut, shape: [batch, numFreqs, numFrames] },
    imag: { data: imagOut, shape: [batch, numFreqs, numFrames] },
  };
}

export function spectro(x: Tensor, nFft = 512, hopLength: number | null = null): ComplexTensor {
  if (hopLength === null) {
    hopLength = Math.floor(nFft / 4);
  }

  // x has shape [..., length]
  const originalShape = x.shape;
  const length = originalShape[originalShape.length - 1];

  // Reshape to [batch, length]
  const otherSize = x.data.length / length;
  const reshapedX = {
    data: x.data,
    shape: [otherSize, length],
  };

  const window = hannWindow(nFft);
  const result = stft(reshapedX, nFft, hopLength, window, true, true, 'reflect');

  // Reshape back to [..., freqs, frames]
  const [batch, freqs, frames] = result.real.shape;
  const outputShape = [...originalShape.slice(0, -1), freqs, frames];

  return {
    real: { data: result.real.data, shape: outputShape },
    imag: { data: result.imag.data, shape: outputShape },
  };
}

export function spec(x: Tensor): ComplexTensor {
  const hl = 1024;

  // le = ceil(x.shape[-1] / hl)
  const lastDimLength = x.shape[x.shape.length - 1];
  const le = Math.ceil(lastDimLength / hl);

  // pad = hl // 2 * 3
  const pad = Math.floor(hl / 2) * 3;

  // Pad input
  const paddingRight = pad + le * hl - lastDimLength;
  const paddedX = pad1d(x, [pad, paddingRight], 'reflect');

  // Compute spectrogram
  const z = spectro(paddedX, 4096, hl);

  // z[..., :-1, :] - remove last frequency bin
  const zShape = z.real.shape;
  const newFreqs = zShape[zShape.length - 2] - 1;
  const frames = zShape[zShape.length - 1];
  const outerSize = z.real.data.length / (zShape[zShape.length - 2] * frames);

  const newShape = [...zShape];
  newShape[newShape.length - 2] = newFreqs;

  const newSize = newShape.reduce((a, b) => a * b, 1);
  const newReal = new Float32Array(newSize);
  const newImag = new Float32Array(newSize);

  for (let i = 0; i < outerSize; i++) {
    for (let freq = 0; freq < newFreqs; freq++) {
      for (let frame = 0; frame < frames; frame++) {
        const oldIdx = i * zShape[zShape.length - 2] * frames + freq * frames + frame;
        const newIdx = i * newFreqs * frames + freq * frames + frame;
        newReal[newIdx] = z.real.data[oldIdx];
        newImag[newIdx] = z.imag.data[oldIdx];
      }
    }
  }

  // z[..., 2:2+le] - slice frames
  const slicedShape = [...newShape];
  slicedShape[slicedShape.length - 1] = le;

  const slicedSize = slicedShape.reduce((a, b) => a * b, 1);
  const slicedReal = new Float32Array(slicedSize);
  const slicedImag = new Float32Array(slicedSize);

  for (let i = 0; i < outerSize; i++) {
    for (let freq = 0; freq < newFreqs; freq++) {
      for (let frame = 0; frame < le; frame++) {
        const srcIdx = i * newFreqs * frames + freq * frames + (2 + frame);
        const dstIdx = i * newFreqs * le + freq * le + frame;
        slicedReal[dstIdx] = newReal[srcIdx];
        slicedImag[dstIdx] = newImag[srcIdx];
      }
    }
  }

  return {
    real: { data: slicedReal, shape: slicedShape },
    imag: { data: slicedImag, shape: slicedShape },
  };
}

export function magnitude(z: ComplexTensor): Tensor {
  // z is complex with shape [B, C, Fr, T]
  const [B, C, Fr, T] = z.real.shape;

  // view_as_real adds a dimension of size 2 at the end: [B, C, Fr, T, 2]
  // permute(0, 1, 4, 2, 3) -> [B, C, 2, Fr, T]
  // reshape to [B, C*2, Fr, T]

  const outShape = [B, C * 2, Fr, T];
  const outSize = B * C * 2 * Fr * T;
  const out = new Float32Array(outSize);

  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) {
      for (let fr = 0; fr < Fr; fr++) {
        for (let t = 0; t < T; t++) {
          const srcIdx = b * C * Fr * T + c * Fr * T + fr * T + t;

          // Real part goes to channel c*2
          const dstIdxReal = b * (C * 2) * Fr * T + c * 2 * Fr * T + fr * T + t;
          out[dstIdxReal] = z.real.data[srcIdx];

          // Imag part goes to channel c*2+1
          const dstIdxImag = b * (C * 2) * Fr * T + (c * 2 + 1) * Fr * T + fr * T + t;
          out[dstIdxImag] = z.imag.data[srcIdx];
        }
      }
    }
  }

  return { data: out, shape: outShape };
}

/**
 * ISTFT - Inverse Short-Time Fourier Transform
 * Matches PyTorch's torch.istft behavior
 */
export function istft(z: ComplexTensor, nFft: number, hopLength: number, window: Float32Array, normalized = true, length: number | null = null, center = true): Tensor {
  // z is complex with shape [batch, freqs, frames]
  const [batch, freqs, numFrames] = z.real.shape;

  if (2 * freqs - 2 !== nFft) {
    throw new Error(`Expected freqs = nFft/2 + 1, got freqs=${freqs}, nFft=${nFft}`);
  }

  // Calculate output length
  let outputLength;
  if (length !== null) {
    outputLength = length;
  } else {
    outputLength = nFft + (numFrames - 1) * hopLength;
  }

  // Account for center padding
  if (center) {
    outputLength -= nFft;
  }

  // Initialize output
  const output = new Float32Array(batch * outputLength);
  const windowSum = new Float32Array(batch * outputLength);

  // Normalization factor
  const norm = normalized ? Math.sqrt(nFft) : 1.0;

  for (let b = 0; b < batch; b++) {
    for (let frame = 0; frame < numFrames; frame++) {
      // Reconstruct full FFT spectrum (including negative frequencies)
      const fullReal = new Float32Array(nFft);
      const fullImag = new Float32Array(nFft);

      // Positive frequencies (0 to Nyquist)
      for (let freq = 0; freq < freqs; freq++) {
        const idx = b * freqs * numFrames + freq * numFrames + frame;
        fullReal[freq] = z.real.data[idx];
        fullImag[freq] = z.imag.data[idx];
      }

      // Negative frequencies (conjugate symmetry)
      for (let freq = freqs; freq < nFft; freq++) {
        const mirrorFreq = nFft - freq;
        fullReal[freq] = fullReal[mirrorFreq];
        fullImag[freq] = -fullImag[mirrorFreq];
      }

      // Inverse FFT
      const frameData = ifft(fullReal, fullImag);

      // Apply window and overlap-add
      const frameStart = frame * hopLength - (center ? nFft / 2 : 0);

      for (let i = 0; i < nFft; i++) {
        const outputIdx = frameStart + i;
        if (outputIdx >= 0 && outputIdx < outputLength) {
          const globalIdx = b * outputLength + outputIdx;
          output[globalIdx] += frameData.real[i] * window[i] * norm;
          windowSum[globalIdx] += window[i] * window[i];
        }
      }
    }
  }

  // Normalize by window sum
  for (let i = 0; i < output.length; i++) {
    if (windowSum[i] > 1e-8) {
      output[i] /= windowSum[i];
    }
  }

  return {
    data: output,
    shape: [batch, outputLength],
  };
}

export function ispectro(z: ComplexTensor, hopLength: number | null = null, length: number | null = null): Tensor {
  // z has shape [..., freqs, frames] with real and imag
  const originalShape = z.real.shape;
  const freqs = originalShape[originalShape.length - 2];
  const frames = originalShape[originalShape.length - 1];

  const nFft = 2 * freqs - 2;
  if (hopLength === null) {
    hopLength = Math.floor(nFft / 4);
  }

  // Reshape to [batch, freqs, frames]
  const otherSize = z.real.data.length / (freqs * frames);
  const reshapedZ = {
    real: { data: z.real.data, shape: [otherSize, freqs, frames] },
    imag: { data: z.imag.data, shape: [otherSize, freqs, frames] },
  };

  const window = hannWindow(nFft);
  const result = istft(reshapedZ, nFft, hopLength, window, true, length, true);

  // Reshape back to [..., length]
  const outputLength = result.shape[1];
  const outputShape = [...originalShape.slice(0, -2), outputLength];

  return {
    data: result.data,
    shape: outputShape,
  };
}

/**
 * Pad complex tensor on frequency or time dimensions
 */
function padComplex(z: ComplexTensor, padFreq: number[], padTime: number[]): ComplexTensor {
  const shape = z.real.shape;
  const ndim = shape.length;

  // padFreq is [left, right] for second-to-last dimension
  // padTime is [left, right] for last dimension
  const [padFreqLeft, padFreqRight] = padFreq;
  const [padTimeLeft, padTimeRight] = padTime;

  const oldFreqs = shape[ndim - 2];
  const oldFrames = shape[ndim - 1];
  const newFreqs = oldFreqs + padFreqLeft + padFreqRight;
  const newFrames = oldFrames + padTimeLeft + padTimeRight;

  const newShape = [...shape];
  newShape[ndim - 2] = newFreqs;
  newShape[ndim - 1] = newFrames;

  const totalSize = newShape.reduce((a, b) => a * b, 1);
  const newReal = new Float32Array(totalSize);
  const newImag = new Float32Array(totalSize);

  const outerSize = totalSize / (newFreqs * newFrames);

  for (let i = 0; i < outerSize; i++) {
    for (let freq = 0; freq < oldFreqs; freq++) {
      for (let frame = 0; frame < oldFrames; frame++) {
        const oldIdx = i * oldFreqs * oldFrames + freq * oldFrames + frame;
        const newIdx = i * newFreqs * newFrames + (freq + padFreqLeft) * newFrames + (frame + padTimeLeft);

        newReal[newIdx] = z.real.data[oldIdx];
        newImag[newIdx] = z.imag.data[oldIdx];
      }
    }
  }

  return {
    real: { data: newReal, shape: newShape },
    imag: { data: newImag, shape: newShape },
  };
}

export function ispec(z: ComplexTensor, length: number): Tensor {
  const hl = 1024;

  // z = F.pad(z, (0, 0, 0, 1)) - pad frequency dimension by 1 on the right
  let paddedZ = padComplex(z, [0, 1], [0, 0]);

  // z = F.pad(z, (2, 2)) - pad time dimension by 2 on each side
  paddedZ = padComplex(paddedZ, [0, 0], [2, 2]);

  // Compute parameters
  const pad = Math.floor(hl / 2) * 3;
  const le = hl * Math.ceil(length / hl) + 2 * pad;

  // Inverse spectrogram
  const x = ispectro(paddedZ, hl, le);

  // Crop to desired length: x[..., pad: pad + length]
  const shape = x.shape;
  const lastDim = shape.length - 1;
  const totalLength = shape[lastDim];

  const newShape = [...shape];
  newShape[lastDim] = length;

  const totalSize = newShape.reduce((a, b) => a * b, 1);
  const newData = new Float32Array(totalSize);

  const outerSize = totalSize / length;

  for (let i = 0; i < outerSize; i++) {
    for (let j = 0; j < length; j++) {
      const oldIdx = i * totalLength + pad + j;
      const newIdx = i * length + j;
      newData[newIdx] = x.data[oldIdx];
    }
  }

  return { data: newData, shape: newShape };
}
