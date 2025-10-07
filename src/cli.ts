#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { separateTracks } from './apply.ts';
import { ONNXHTDemucs } from './onnx-htdemucs.ts';
import { samplesToWav, wavToSamples, type RawAudio } from './wav-utils.ts';
import { encode as encodeMP3 } from './mp3-encoder.ts';

const { values, positionals } = parseArgs({
  options: {
    output: {
      type: 'string',
      short: 'o',
      default: './separated',
    },
    overlap: {
      type: 'string',
      default: '0.25',
    },
    mp3: {
      type: 'boolean',
    },
    help: {
      type: 'boolean',
    }
  },
  allowPositionals: true,
});

if (positionals.length !== 1 || values.help) {
  console.error('Usage: demucs <input.wav> [options]');
  console.error('');
  console.error('Arguments:');
  console.error('  <input.wav>      Path to the input WAV file');
  console.error('');
  console.error('Options:');
  console.error('  --help                  Print this message and exit');
  console.error('  --mp3                   Save tracks as .mp3 instead of .wav');
  console.error('  -o, --output <dir>      Output directory (default: ./separated)');
  console.error('  --overlap <float>       Overlap ratio for chunking (default: 0.25)');
  process.exit(values.help ? 0 : 1);
}

const inputPath = positionals[0];
const outputDir = values.output;
const overlap = parseFloat(values.overlap);

if (!fs.existsSync(inputPath)) {
  console.error(`Error: Input file not found: ${inputPath}`);
  process.exit(1);
}

console.log('Loading model...');
let weights = fs.readFileSync(path.join(import.meta.dirname, '..', 'htdemucs.onnx')).buffer;
const model = await ONNXHTDemucs.init(weights);

console.log('Reading and processing audio...');
const inputBuffer = new Uint8Array(fs.readFileSync(inputPath));

const rawAudio = wavToSamples(inputBuffer);

function progress(step: number, total: number) {
  console.log(`${step}/${total}`);
}
const tracks = await separateTracks(model, rawAudio, progress, overlap);

console.log('Writing output files...');
const inputBasename = path.basename(inputPath, path.extname(inputPath));
const fileOutDir = path.join(outputDir, inputBasename);
fs.mkdirSync(fileOutDir, { recursive: true });

for (const [trackName, samples] of Object.entries(tracks) as [string, RawAudio][]) {
  if (values.mp3) {
    let mp3 = await encodeMP3(samples);
    let outputPath = path.join(fileOutDir, `${trackName}.mp3`);
    fs.writeFileSync(outputPath, Buffer.from(mp3));
    console.log(`  Wrote: ${outputPath}`);
  } else {
    let wavBuffer = samplesToWav(samples.channelData, samples.sampleRate);
    let outputPath = path.join(fileOutDir, `${trackName}.wav`);
    fs.writeFileSync(outputPath, wavBuffer);
    console.log(`  Wrote: ${outputPath}`);
  }
}

console.log('Done!');
