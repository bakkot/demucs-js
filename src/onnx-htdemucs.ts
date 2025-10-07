import * as ort from 'onnxruntime-node';

import type { Tensor } from './dsp.ts';

// as far as I can tell this only matters on the web
ort.env.wasm.wasmPaths = './';

export class ONNXHTDemucs {
  readonly sources = ['drums', 'bass', 'other', 'vocals'] as const;
  readonly audioChannels = 2;
  readonly samplerate = 44100;
  readonly segment = 7.8;
  // @ts-expect-error these are initialized in the async initializer
  session: ort.InferenceSession;
  // @ts-expect-error
  inputNames: readonly string[];
  // @ts-expect-error
  outputNames: readonly string[];

  private constructor() {}

  static async init(modelWeights: ArrayBuffer): Promise<ONNXHTDemucs> {
    let instance = new ONNXHTDemucs();
    let opts = typeof self === 'undefined' || typeof self.location  === 'undefined' ? undefined : { executionProviders: ['webgpu', 'wasm'] };
    let session = await ort.InferenceSession.create(modelWeights, opts);
    instance.session = session;
    instance.inputNames = session.inputNames;
    instance.outputNames = session.outputNames;
    return instance;
  }

  validLength(length: number): number {
    const trainingLength = Math.floor(this.segment * this.samplerate);
    if (trainingLength < length) {
      throw new Error(`Given length ${length} is longer than training length ${trainingLength}`);
    }
    return trainingLength;
  }

  async forward(mix: Tensor, magspec: Tensor): Promise<{ outX: Tensor; outXt: Tensor }> {
    const mixTensor = new ort.Tensor('float32', mix.data, mix.shape);
    const magspecTensor = new ort.Tensor('float32', magspec.data, magspec.shape);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.inputNames[0]] = mixTensor;
    feeds[this.inputNames[1]] = magspecTensor;

    const results = await this.session!.run(feeds);

    const outX = results[this.outputNames[0]];
    const outXt = results[this.outputNames[1]];

    return {
      outX: { data: outX.data as Float32Array, shape: outX.dims },
      outXt: { data: outXt.data as Float32Array, shape: outXt.dims },
    };
  }
}
