/* eslint-disable no-restricted-globals */
const globalScope = self;

const inferenceState = {
  audioModel: null,
  audioModelPromise: null,
  videoModel: null,
  videoModelPromise: null,
  runtimeReady: false,
  loading: false,
  audioModelUrl: null,
  videoModelUrl: null,
  audioModelId: "yamnet-client-tfjs",
  videoModelId: "resnet-mc-client-tflite",
  wasmBaseUrl: "/static/tflite",
  classMap: [],
  classMapUrl: null,
  videoClassMap: [],
  videoClassMapUrl: null,
  videoFrameCount: 16,
  videoInputSize: 112,
};

function configureInference(payload = {}) {
  Object.assign(inferenceState, {
    audioModelUrl: payload.audioModelUrl || inferenceState.audioModelUrl,
    videoModelUrl: payload.videoModelUrl || inferenceState.videoModelUrl,
    audioModelId: payload.audioModelId || inferenceState.audioModelId,
    videoModelId: payload.videoModelId || inferenceState.videoModelId,
    wasmBaseUrl: payload.wasmBaseUrl || inferenceState.wasmBaseUrl,
    classMapUrl: payload.classMapUrl || inferenceState.classMapUrl,
    videoClassMapUrl: payload.videoClassMapUrl || inferenceState.videoClassMapUrl,
    videoFrameCount: payload.videoFrameCount || inferenceState.videoFrameCount,
    videoInputSize: payload.videoInputSize || inferenceState.videoInputSize,
  });
}

function toTimestampMs(value) {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return value.getTime();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function computeSegmentDurationMs(startTs, endTs) {
  const start = toTimestampMs(startTs);
  const end = toTimestampMs(endTs);
  if (typeof start === "number" && typeof end === "number") {
    return Math.max(Math.round(end - start), 0);
  }
  return null;
}

async function loadScriptWithFallback(urls) {
  for (const url of urls) {
    try {
      globalScope.importScripts(url);
      return;
    } catch (error) {
      console.warn("Inference worker failed to load script", url, error);
    }
  }
  throw new Error("All script sources failed to load");
}

async function ensureInferenceRuntime() {
  if (inferenceState.runtimeReady) {
    return;
  }
  if (inferenceState.loading) {
    while (inferenceState.loading) {
      // eslint-disable-next-line no-await-in-loop
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return;
  }
  inferenceState.loading = true;
  try {
    if (!globalScope.tf) {
      const tfUrls = [
        `${inferenceState.wasmBaseUrl}/tf.min.js`,
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
        "https://unpkg.com/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
      ];
      await loadScriptWithFallback(tfUrls);
    }
    if (globalScope.tf?.setBackend) {
      await globalScope.tf.setBackend("cpu");
      await globalScope.tf.ready();
    }
    if (!globalScope.tflite) {
      const base = inferenceState.wasmBaseUrl;
      await loadScriptWithFallback([`${base}/tf-tflite.min.js`]);
      await loadScriptWithFallback([`${base}/tflite_web_api_cc_simd.js`]);
      if (globalScope.tflite?.setWasmPath) {
        globalScope.tflite.setWasmPath(`${base}/`);
      }
    }
    if (!inferenceState.classMap.length && inferenceState.classMapUrl) {
      inferenceState.classMap = await loadClassMap();
    }
    if (!inferenceState.videoClassMap.length && inferenceState.videoClassMapUrl) {
      inferenceState.videoClassMap = await loadVideoClassMap();
    }
    inferenceState.runtimeReady = Boolean(globalScope.tf && globalScope.tflite);
    if (!inferenceState.runtimeReady) {
      throw new Error(
        `Inference runtime failed to initialize. TF: ${Boolean(globalScope.tf)} TFLite: ${Boolean(globalScope.tflite)}`,
      );
    }
  } finally {
    inferenceState.loading = false;
  }
}

async function fetchModelFromCache(url) {
  if (!url) {
    throw new Error("Model URL missing");
  }
  if (globalScope.caches) {
    const cache = await caches.open("buurt-models");
    const cached = await cache.match(url);
    if (cached) {
      return cached.arrayBuffer();
    }
    const response = await fetch(url, { cache: "force-cache" });
    await cache.put(url, response.clone());
    return response.arrayBuffer();
  }
  const response = await fetch(url, { cache: "force-cache" });
  return response.arrayBuffer();
}

async function loadAudioModel() {
  if (inferenceState.audioModel) {
    return inferenceState.audioModel;
  }
  if (inferenceState.audioModelPromise) {
    return inferenceState.audioModelPromise;
  }
  if (!inferenceState.runtimeReady || !globalScope.tflite) {
    throw new Error("Inference runtime not ready");
  }
  if (!inferenceState.audioModelUrl) {
    throw new Error("Audio model URL not configured");
  }

  const threads = Math.max(navigator.hardwareConcurrency / 2 || 1, 1);
  const load = (async () => {
    try {
      const model = await globalScope.tflite.loadTFLiteModel(
        inferenceState.audioModelUrl,
        { numThreads: threads },
      );
      inferenceState.audioModel = model;
      return model;
    } catch (primaryError) {
      console.error("Failed to load audio model", primaryError);
      const buffer = await fetchModelFromCache(inferenceState.audioModelUrl);
      const model = await globalScope.tflite.loadTFLiteModel(buffer, { numThreads: threads });
      inferenceState.audioModel = model;
      return model;
    } finally {
      inferenceState.audioModelPromise = null;
    }
  })();
  inferenceState.audioModelPromise = load;
  return load;
}

async function loadVideoModel() {
  if (inferenceState.videoModel) {
    return inferenceState.videoModel;
  }
  if (inferenceState.videoModelPromise) {
    return inferenceState.videoModelPromise;
  }
  if (!inferenceState.runtimeReady || !globalScope.tflite) {
    throw new Error("Inference runtime not ready");
  }
  if (!inferenceState.videoModelUrl) {
    throw new Error("Video model URL not configured");
  }

  const threads = Math.max(navigator.hardwareConcurrency / 2 || 1, 1);
  const load = (async () => {
    try {
      const model = await globalScope.tflite.loadTFLiteModel(
        inferenceState.videoModelUrl,
        { numThreads: threads },
      );
      inferenceState.videoModel = model;
      return model;
    } catch (primaryError) {
      console.error("Failed to load video model", primaryError);
      const buffer = await fetchModelFromCache(inferenceState.videoModelUrl);
      const model = await globalScope.tflite.loadTFLiteModel(buffer, { numThreads: threads });
      inferenceState.videoModel = model;
      return model;
    } finally {
      inferenceState.videoModelPromise = null;
    }
  })();
  inferenceState.videoModelPromise = load;
  return load;
}

function downsamplePcm(samples, sourceRate, targetRate = 16000) {
  if (!samples || !samples.length || !sourceRate) {
    return new Float32Array();
  }
  if (sourceRate === targetRate) {
    return samples.slice ? samples.slice() : new Float32Array(samples);
  }
  const ratio = sourceRate / targetRate;
  const newLength = Math.floor(samples.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = samples[Math.floor(i * ratio)];
  }
  return result;
}

async function runAudioInference(audioPayload) {
  if (!audioPayload?.pcm) {
    return [];
  }
  const model = await loadAudioModel();
  const start = performance.now();
  const mono = downsamplePcm(audioPayload.pcm, audioPayload.sampleRate || 48000, 16000);
  if (!mono.length) {
    return [];
  }
  const waveform = globalScope.tf.tensor1d(mono);
  const frameLength = 400;
  const frameStep = 160;
  const fftLength = 512;
  const numMelBins = 64;
  const patchFrames = 96;
  const patchHop = 48;
  const lowerEdgeHz = 125;
  const upperEdgeHz = 7500;

  const stft = globalScope.tf.signal.stft(
    waveform,
    frameLength,
    frameStep,
    fftLength,
    () => globalScope.tf.signal.hannWindow(frameLength),
  );
  const magnitude = globalScope.tf.abs(stft);
  const powerSpec = globalScope.tf.square(magnitude);
  const melMatrix = buildMelFilterbank(
    numMelBins,
    fftLength / 2 + 1,
    16000,
    lowerEdgeHz,
    upperEdgeHz,
  );
  const melSpec = globalScope.tf.matMul(powerSpec, melMatrix, false, true);
  const logMelSpec = globalScope.tf.log(melSpec.add(1e-6));

  const timeSteps = logMelSpec.shape[0] || 0;
  const padFrames = Math.max(patchFrames - timeSteps, 0);
  let framedInput = logMelSpec;
  let disposeFramedInput = false;
  if (padFrames > 0) {
    framedInput = globalScope.tf.pad(logMelSpec, [
      [0, padFrames],
      [0, 0],
    ]);
    disposeFramedInput = true;
  }

  const safeHop = Math.max(patchHop, 1);
  const slices = [];
  for (let idx = 0; idx + patchFrames <= framedInput.shape[0]; idx += safeHop) {
    slices.push(framedInput.slice([idx, 0], [patchFrames, numMelBins]));
  }
  if (!slices.length) {
    waveform.dispose();
    stft.dispose();
    magnitude.dispose();
    powerSpec.dispose();
    melMatrix.dispose();
    melSpec.dispose();
    logMelSpec.dispose();
    if (disposeFramedInput) {
      framedInput.dispose();
    }
    return [];
  }

  const patches = globalScope.tf.stack(slices);
  const numPatches = patches.shape[0];
  const mergedPatch = patches.mean(0);
  const inputTensor = mergedPatch.reshape([1, 1, patchFrames, numMelBins]);

  let output;
  try {
    output = model.predict(inputTensor);
  } finally {
    inputTensor.dispose();
    waveform.dispose();
    stft.dispose();
    magnitude.dispose();
    powerSpec.dispose();
    melMatrix.dispose();
    melSpec.dispose();
    logMelSpec.dispose();
    if (disposeFramedInput) {
      framedInput.dispose();
    }
    patches.dispose();
    mergedPatch.dispose();
    slices.forEach((slice) => slice.dispose());
  }

  const logits = await output.data();
  if (output.dispose) {
    output.dispose();
  }
  if (!logits?.length) {
    return [];
  }

  const numClasses = logits.length / numPatches;
  const classTotals = new Float32Array(numClasses);
  for (let i = 0; i < logits.length; i++) {
    classTotals[i % numClasses] += logits[i];
  }
  const avgLogits = Array.from(classTotals).map((value) => value / numPatches);
  const { maxIdx, maxProb } = softmaxTop1(avgLogits);
  const latencyMs = Math.max(Math.round(performance.now() - start), 1);
  const confidence = Math.max(Math.min(maxProb, 1), 0);
  const label = inferenceState.classMap[maxIdx] || `audio_class_${maxIdx}`;

  return [
    {
      class: label,
      confidence,
      timestamp: audioPayload.endTs.toISOString(),
      model_id: inferenceState.audioModelId,
      inference_time_ms: latencyMs,
      segment_duration_ms: audioPayload.segmentDurationMs ?? null,
      origin: "client",
    },
  ];
}

function normalizeVideoFrames(frameData, frameCount, frameSize, expectedCount) {
  const availableCount = Math.max(Math.floor(frameData.length / frameSize), 0);
  const initialCount = Math.max(
    frameCount || availableCount || expectedCount || 0,
    0,
  );
  if (!initialCount || availableCount === 0) {
    return null;
  }
  if (availableCount >= expectedCount) {
    const startIdx = availableCount - expectedCount;
    return frameData.subarray(
      startIdx * frameSize,
      startIdx * frameSize + expectedCount * frameSize,
    );
  }
  const copyCount = Math.min(availableCount, expectedCount);
  const padded = new Float32Array(expectedCount * frameSize);
  for (let i = 0; i < copyCount; i++) {
    padded.set(
      frameData.subarray(i * frameSize, (i + 1) * frameSize),
      i * frameSize,
    );
  }
  const repeatStart = Math.max(copyCount - 1, 0) * frameSize;
  const repeatSlice = frameData.subarray(
    repeatStart,
    repeatStart + frameSize,
  );
  for (let i = copyCount; i < expectedCount; i++) {
    padded.set(repeatSlice, i * frameSize);
  }
  return padded;
}

async function runVideoInference(videoPayload) {
  if (!videoPayload?.frames) {
    return [];
  }
  const model = await loadVideoModel();
  const frameData = videoPayload.frames instanceof Float32Array
    ? videoPayload.frames
    : new Float32Array(videoPayload.frames);
  const inputSize = Math.max(videoPayload.inputSize || inferenceState.videoInputSize, 1);
  const frameSize = inputSize * inputSize * 3;
  const targetCount = Math.max(inferenceState.videoFrameCount, 1);
  const normalizedFrames = normalizeVideoFrames(
    frameData,
    videoPayload.frameCount,
    frameSize,
    targetCount,
  );
  if (!normalizedFrames) {
    return [];
  }
  const tensor = globalScope.tf.tensor(
    normalizedFrames,
    [1, targetCount, inputSize, inputSize, 3],
    "float32",
  );

  const start = performance.now();
  let output;
  try {
    output = model.predict(tensor);
  } finally {
    tensor.dispose();
  }

  let logits;
  if (output?.dataSync) {
    logits = output.dataSync();
  } else if (output?.data) {
    logits = await output.data();
  } else if (ArrayBuffer.isView(output)) {
    logits = output;
  } else if (Array.isArray(output)) {
    logits = output;
  } else {
    console.warn("Video inference returned unexpected output", output);
    return [];
  }
  if (output?.dispose) {
    output.dispose();
  }
  if (!logits || logits.length === 0) {
    console.warn("Video inference produced empty logits");
    return [];
  }

  const { maxIdx, maxProb } = softmaxTop1(Array.from(logits));
  const label = inferenceState.videoClassMap[maxIdx] || `video_class_${maxIdx}`;
  const latencyMs = Math.max(Math.round(performance.now() - start), 1);
  const confidence = Math.max(Math.min(maxProb, 1), 0);

  return [
    {
      class: label,
      confidence,
      timestamp: videoPayload.endTs.toISOString(),
      model_id: inferenceState.videoModelId,
      inference_time_ms: latencyMs,
      segment_duration_ms: videoPayload.segmentDurationMs ?? null,
      origin: "client",
    },
  ];
}

async function loadClassMap() {
  if (!inferenceState.classMapUrl) {
    return [];
  }
  try {
    const response = await fetch(inferenceState.classMapUrl, { cache: "force-cache" });
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const text = await response.text();
    const lines = text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    lines.shift();
    const labels = [];
    for (const line of lines) {
      const parts = line.trim().split(",");
      if (parts.length >= 3) {
        labels.push(parts[2].replace(/(^\"|\"$)/g, "").trim());
      }
    }
    return labels;
  } catch (error) {
    console.warn("Failed to load class map", error);
    return [];
  }
}

async function loadVideoClassMap() {
  if (!inferenceState.videoClassMapUrl) {
    return [];
  }
  try {
    const response = await fetch(inferenceState.videoClassMapUrl, { cache: "force-cache" });
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const text = await response.text();
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
  } catch (error) {
    console.warn("Failed to load video class map", error);
    return [];
  }
}

function buildMelFilterbank(numMelBins, numSpectrogramBins, sampleRate, lowerEdgeHz, upperEdgeHz) {
  const hzToMel = (hz) => 1127 * Math.log(1 + hz / 700);
  const melToHz = (mel) => 700 * (Math.exp(mel / 1127) - 1);
  const lowerMel = hzToMel(lowerEdgeHz);
  const upperMel = hzToMel(upperEdgeHz);
  const melPoints = new Float32Array(numMelBins + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = lowerMel + ((upperMel - lowerMel) * i) / (numMelBins + 1);
  }
  const hzPoints = melPoints.map(melToHz);
  const binFrequencies = new Float32Array(numSpectrogramBins);
  const linearFreqs = sampleRate / 2 / (numSpectrogramBins - 1);
  for (let i = 0; i < numSpectrogramBins; i++) {
    binFrequencies[i] = i * linearFreqs;
  }
  const filterbank = [];
  for (let m = 0; m < numMelBins; m++) {
    const lower = hzPoints[m];
    const center = hzPoints[m + 1];
    const upper = hzPoints[m + 2];
    const weights = new Float32Array(numSpectrogramBins);
    for (let k = 0; k < numSpectrogramBins; k++) {
      const freq = binFrequencies[k];
      let weight = 0;
      if (freq >= lower && freq <= center) {
        weight = (freq - lower) / (center - lower + 1e-8);
      } else if (freq > center && freq <= upper) {
        weight = (upper - freq) / (upper - center + 1e-8);
      }
      weights[k] = weight;
    }
    filterbank.push(weights);
  }
  return globalScope.tf.tensor2d(filterbank, [numMelBins, numSpectrogramBins]);
}

function softmaxTop1(values) {
  let maxLogit = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > maxLogit) {
      maxLogit = values[i];
    }
  }
  let sumExp = 0;
  let maxIdx = 0;
  let maxProb = 0;
  for (let i = 0; i < values.length; i++) {
    const expVal = Math.exp(values[i] - maxLogit);
    sumExp += expVal;
    if (expVal > maxProb) {
      maxProb = expVal;
      maxIdx = i;
    }
  }
  if (sumExp > 0) {
    maxProb /= sumExp;
  }
  return { maxIdx, maxProb };
}

async function handleRunRequest(payload = {}) {
  await ensureInferenceRuntime();
  const detections = [];
  if (payload.audio?.pcm) {
    const audioData = new Float32Array(payload.audio.pcm);
    const audioStart = new Date(payload.audio.startTs || Date.now());
    const audioEnd = new Date(payload.audio.endTs || Date.now());
    const audioDuration = computeSegmentDurationMs(audioStart, audioEnd);
    const audioResult = await runAudioInference({
      pcm: audioData,
      sampleRate: payload.audio.sampleRate,
      startTs: audioStart,
      endTs: audioEnd,
      segmentDurationMs: payload.audio.segmentDurationMs ?? audioDuration,
    });
    detections.push(...audioResult);
  }
  if (payload.video?.frames) {
    const videoFrames = new Float32Array(payload.video.frames);
    const videoStart = new Date(payload.video.startTs || Date.now());
    const videoEnd = new Date(payload.video.endTs || Date.now());
    const videoDuration = computeSegmentDurationMs(videoStart, videoEnd);
    const videoResult = await runVideoInference({
      frames: videoFrames,
      frameCount: payload.video.frameCount,
      inputSize: payload.video.inputSize,
      endTs: videoEnd,
      segmentDurationMs: payload.video.segmentDurationMs ?? videoDuration,
    });
    detections.push(...videoResult);
  }
  return detections;
}

function respond(id, result) {
  globalScope.postMessage({ id, success: true, result });
}

function respondError(id, error) {
  const detail = error instanceof Error ? error.message : String(error);
  globalScope.postMessage({ id, success: false, error: detail });
}

globalScope.addEventListener("message", async (event) => {
  const { id, type, payload } = event.data || {};
  if (!id) {
    return;
  }
  try {
    switch (type) {
      case "INIT":
        configureInference(payload);
        respond(id, { ok: true });
        break;
      case "PRELOAD":
        await ensureInferenceRuntime();
        await Promise.allSettled([loadAudioModel(), loadVideoModel()]);
        respond(id, { ok: true });
        break;
      case "RUN_DETECTIONS":
        {
          const result = await handleRunRequest(payload);
          respond(id, result);
        }
        break;
      default:
        respond(id, []);
        break;
    }
  } catch (error) {
    respondError(id, error);
  }
});
