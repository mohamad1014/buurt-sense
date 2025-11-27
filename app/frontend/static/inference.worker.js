const workerState = {
  config: null,
  runtimeReady: false,
  loading: false,
  audioModel: null,
  audioModelPromise: null,
  videoModel: null,
  videoModelPromise: null,
  classMap: [],
  videoClassMap: [],
  videoModelSources: [],
  disabledVideoSources: new Set(),
};

self.onmessage = async (event) => {
  const { id, type, payload } = event.data || {};
  if (!type) {
    return;
  }
  try {
    let result = null;
    if (type === "init") {
      configureWorker(payload);
      await ensureInferenceRuntime();
      result = { ready: true };
    } else if (type === "warmup") {
      result = await warmupModels();
    } else if (type === "processSegment") {
      result = await processSegment(payload);
    } else {
      throw new Error(`Unknown inference worker message: ${type}`);
    }
    if (id) {
      self.postMessage({ id, success: true, result });
    }
  } catch (error) {
    if (id) {
      self.postMessage({ id, success: false, error: serializeError(error) });
    }
  }
};

function configureWorker(config) {
  workerState.config = config || {};
      workerState.videoModelSources = Array.isArray(config?.videoModelSources)
        ? config.videoModelSources.filter(Boolean)
        : [];
  workerState.videoModel = null;
  workerState.audioModel = null;
  workerState.runtimeReady = false;
  workerState.classMap = [];
  workerState.videoClassMap = [];
  workerState.disabledVideoSources = new Set();
}

async function processSegment(payload = {}) {
  await ensureInferenceRuntime();
  const detections = [];
  if (payload.audio?.pcm instanceof Float32Array) {
    const audioDetections = await runAudioInference(
      payload.audio.pcm,
      payload.audio.sampleRate,
      payload.endTs,
    );
    detections.push(...audioDetections);
  }
  if (payload.video?.data instanceof Float32Array) {
    ensureVideoModelLoading();
    const videoDetections = await runVideoInference(
      payload.video.data,
      payload.video.frameCount,
      payload.startTs,
      payload.endTs,
    );
    detections.push(...videoDetections);
  }
  return detections;
}

async function warmupModels() {
  await ensureInferenceRuntime();
  try {
    await loadAudioModel();
  } catch (error) {
    console.error("Audio warm-up failed", error);
  }
  try {
    await loadVideoModel();
  } catch (error) {
    console.error("Video warm-up failed", error);
  }
  return {
    audio_ready: Boolean(workerState.audioModel),
    video_ready: Boolean(workerState.videoModel),
  };
}

async function ensureInferenceRuntime() {
  if (workerState.runtimeReady) {
    return;
  }
  if (workerState.loading) {
    while (workerState.loading) {
      // eslint-disable-next-line no-await-in-loop
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return;
  }
  workerState.loading = true;
  try {
    if (!self.tf) {
      const base = workerState.config?.wasmBaseUrl || "/static/tflite";
      const tfUrls = [
        `${base}/tf.min.js`,
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
        "https://unpkg.com/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
      ];
      await loadScriptWithFallback(tfUrls);
    }

    if (self.tf?.setBackend) {
      await self.tf.setBackend("cpu");
      await self.tf.ready();
    }

    if (!self.tflite) {
      const base = workerState.config?.wasmBaseUrl || "/static/tflite";
      const normalizedBase = `${base.replace(/\/+$/, "")}/`;
      self.TFLITE_JS_DIR = normalizedBase;
      self.TFLITE_WASM_PATH = normalizedBase;
      const originalImportScripts = self.importScripts;
      const originalFetch = self.fetch.bind(self);
      self.importScripts = (...urls) => {
        const resolved = urls.map((url) => resolveWorkerUrl(url, normalizedBase));
        return originalImportScripts(...resolved);
      };
      self.fetch = (input, init) => {
        const target =
          typeof input === "string" ? input : input?.url || "";
        if (
          target &&
          target.endsWith("tflite_web_api_cc_simd.wasm") &&
          !target.startsWith("http")
        ) {
          input = `${normalizedBase}tflite_web_api_cc_simd.wasm`;
        }
        return originalFetch(input, init);
      };
      await loadScriptWithFallback([`${normalizedBase}tf-tflite.min.js`]);
      await loadScriptWithFallback([`${normalizedBase}tflite_web_api_cc_simd.js`]);
      if (self.tflite?.setWasmPath) {
        self.tflite.setWasmPath(normalizedBase);
      }
    }

    if (!workerState.classMap.length) {
      workerState.classMap = await loadClassMap();
    }
    if (!workerState.videoClassMap.length) {
      workerState.videoClassMap = await loadVideoClassMap();
    }

    workerState.runtimeReady = Boolean(self.tf && self.tflite);
    if (!workerState.runtimeReady) {
      throw new Error(
        `Inference runtime failed to initialize. TF: ${Boolean(
          self.tf,
        )}, TFLite: ${Boolean(self.tflite)}`,
      );
    }
  } finally {
    workerState.loading = false;
  }
}

async function runAudioInference(audioInput, sampleRate, endTs) {
  try {
    const model = await loadAudioModel();
    const rate = sampleRate || 48000;
    const mono = downsamplePcm(audioInput, rate, 16000);
    if (!mono.length) {
      return [];
    }

    const start = performance.now();
    const waveform = self.tf.tensor1d(mono);
    const frameLength = 400;
    const frameStep = 160;
    const fftLength = 512;
    const numMelBins = 64;
    const patchFrames = 96;
    const patchHop = 48;
    const lowerEdgeHz = 125;
    const upperEdgeHz = 7500;

    const stft = self.tf.signal.stft(
      waveform,
      frameLength,
      frameStep,
      fftLength,
      () => self.tf.signal.hannWindow(frameLength),
    );
    const magnitude = self.tf.abs(stft);
    const powerSpec = self.tf.square(magnitude);
    const melMatrix = buildMelFilterbank(
      numMelBins,
      fftLength / 2 + 1,
      16000,
      lowerEdgeHz,
      upperEdgeHz,
    );
    const melSpec = self.tf.matMul(powerSpec, melMatrix, false, true);
    const logMelSpec = self.tf.log(melSpec.add(1e-6));

    const timeSteps = logMelSpec.shape[0] || 0;
    const padFrames = Math.max(patchFrames - timeSteps, 0);
    let framedInput = logMelSpec;
    let disposeFramedInput = false;
    if (padFrames > 0) {
      framedInput = self.tf.pad(logMelSpec, [
        [0, padFrames],
        [0, 0],
      ]);
      disposeFramedInput = true;
    }

    const safeHop = Math.max(patchHop, 1);
    const slices = [];
    for (
      let startIndex = 0;
      startIndex + patchFrames <= framedInput.shape[0];
      startIndex += safeHop
    ) {
      slices.push(framedInput.slice([startIndex, 0], [patchFrames, numMelBins]));
    }
    if (!slices.length) {
      cleanupAudioTensors({
        waveform,
        stft,
        magnitude,
        powerSpec,
        melMatrix,
        melSpec,
        logMelSpec,
        framedInput,
        disposeFramedInput,
        slices,
      });
      return [];
    }

    const patches = self.tf.stack(slices);
    const numPatches = patches.shape[0];
    const mergedPatch = patches.mean(0);
    const inputTensor = mergedPatch.reshape([1, 1, patchFrames, numMelBins]);

    let output;
    try {
      output = model.predict(inputTensor);
    } finally {
      inputTensor.dispose();
      cleanupAudioTensors({
        waveform,
        stft,
        magnitude,
        powerSpec,
        melMatrix,
        melSpec,
        logMelSpec,
        framedInput,
        disposeFramedInput,
        patches,
        mergedPatch,
        slices,
      });
    }

    const logits = await output.data();
    if (output.dispose) {
      output.dispose();
    }
    if (!logits || !logits.length) {
      return [];
    }
    const numClasses = Math.max(logits.length / numPatches, 1);
    const classTotals = new Float32Array(numClasses);
    for (let i = 0; i < logits.length; i++) {
      classTotals[i % numClasses] += logits[i];
    }
    const avgLogits = Array.from(classTotals).map((v) => v / numPatches);
    const { maxIdx, maxProb } = softmaxTop1(avgLogits);
    const latencyMs = Math.max(Math.round(performance.now() - start), 1);
    const label =
      workerState.classMap[maxIdx] || `audio_class_${String(maxIdx)}`;

    return [
      {
        class: label,
        confidence: Math.max(Math.min(maxProb, 1), 0),
        timestamp: new Date(endTs || Date.now()).toISOString(),
        model_id: workerState.config?.audioModelId || "client-audio",
        inference_latency_ms: latencyMs,
        origin: "client",
      },
    ];
  } catch (error) {
    console.error("Audio inference failed", error);
    return [];
  }
}

function cleanupAudioTensors({
  waveform,
  stft,
  magnitude,
  powerSpec,
  melMatrix,
  melSpec,
  logMelSpec,
  framedInput,
  disposeFramedInput,
  patches,
  mergedPatch,
  slices,
}) {
  waveform?.dispose();
  stft?.dispose();
  magnitude?.dispose();
  powerSpec?.dispose();
  melMatrix?.dispose();
  melSpec?.dispose();
  logMelSpec?.dispose();
  if (disposeFramedInput) {
    framedInput?.dispose();
  }
  patches?.dispose();
  mergedPatch?.dispose();
  slices?.forEach((slice) => slice.dispose());
}

async function runVideoInference(videoData, frameCount, startTs, endTs) {
  if (!workerState.videoModel) {
    ensureVideoModelLoading();
    if (!workerState.videoModel) {
      return [];
    }
  }
  try {
    const model = await loadVideoModel();
    const expectedFrameCount = Math.max(workerState.config?.videoFrameCount || 16, 1);
    const inputSize = Math.max(workerState.config?.videoInputSize || 112, 1);
    const frames = Math.max(frameCount || expectedFrameCount, 1);
    const data = ensureVideoTensor(videoData, frames, expectedFrameCount, inputSize);
    if (!data) {
      return [];
    }
    const inputTensor = self.tf.tensor(
      data,
      [1, expectedFrameCount, inputSize, inputSize, 3],
      "float32",
    );
    const start = performance.now();
    let output;
    try {
      output = model.predict(inputTensor);
    } finally {
      inputTensor.dispose();
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
    } else if (output == null) {
      console.warn("Video inference returned no output tensor");
      return [];
    } else {
      console.warn("Video inference returned unexpected output", output);
      return [];
    }
    if (output?.dispose) {
      output.dispose();
    }
    if (!logits || !logits.length) {
      return [];
    }
    const { maxIdx, maxProb } = softmaxTop1(Array.from(logits));
    const label =
      workerState.videoClassMap[maxIdx] || `video_class_${String(maxIdx)}`;
    const latencyMs = Math.max(Math.round(performance.now() - start), 1);
    return [
      {
        class: label,
        confidence: Math.max(Math.min(maxProb, 1), 0),
        timestamp: new Date(endTs || Date.now()).toISOString(),
        model_id: workerState.config?.videoModelId || "client-video",
        inference_latency_ms: latencyMs,
        origin: "client",
      },
    ];
  } catch (error) {
    console.error("Video inference failed", error);
    return [];
  }
}

function ensureVideoTensor(rawData, rawFrameCount, expectedFrameCount, inputSize) {
  if (!(rawData instanceof Float32Array)) {
    return null;
  }
  const frameSize = inputSize * inputSize * 3;
  const expectedLength = expectedFrameCount * frameSize;
  if (rawData.length === expectedLength && rawFrameCount === expectedFrameCount) {
    return rawData;
  }
  if (rawData.length < frameSize) {
    return null;
  }
  const tensorData = new Float32Array(expectedLength);
  const availableFrames = Math.max(Math.floor(rawData.length / frameSize), 1);
  const usableFrames = Math.min(availableFrames, expectedFrameCount);
  for (let i = 0; i < usableFrames; i++) {
    tensorData.set(
      rawData.subarray(i * frameSize, (i + 1) * frameSize),
      i * frameSize,
    );
  }
  const lastFrameOffset = (usableFrames - 1) * frameSize;
  const repeatSlice = rawData.subarray(lastFrameOffset, lastFrameOffset + frameSize);
  for (let i = usableFrames; i < expectedFrameCount; i++) {
    tensorData.set(repeatSlice, i * frameSize);
  }
  return tensorData;
}

async function loadAudioModel() {
  if (workerState.audioModel) {
    return workerState.audioModel;
  }
  if (workerState.audioModelPromise) {
    return workerState.audioModelPromise;
  }
  if (!workerState.runtimeReady || !self.tflite) {
    throw new Error("Inference runtime not ready");
  }
  if (!workerState.config?.audioModelUrl) {
    throw new Error("Audio model URL not configured");
  }
  const load = (async () => {
    const threads = Math.max((self.navigator?.hardwareConcurrency || 4) / 2, 1);
    try {
      const model = await self.tflite.loadTFLiteModel(
        workerState.config.audioModelUrl,
        { numThreads: threads },
      );
      workerState.audioModel = model;
      return model;
    } catch (primaryError) {
      console.error("Failed to load audio model from URL", primaryError);
      const buffer = await fetchModelFromCache(workerState.config.audioModelUrl);
      const model = await self.tflite.loadTFLiteModel(buffer, {
        numThreads: threads,
      });
      workerState.audioModel = model;
      return model;
    }
  })()
    .catch((error) => {
      workerState.audioModel = null;
      throw error;
    })
    .finally(() => {
      workerState.audioModelPromise = null;
    });
  workerState.audioModelPromise = load;
  return load;
}

async function loadVideoModel() {
  if (workerState.videoModel) {
    return workerState.videoModel;
  }
  if (workerState.videoModelPromise) {
    return workerState.videoModelPromise;
  }
  if (!workerState.runtimeReady || !self.tflite) {
    throw new Error("Inference runtime not ready");
  }
  const sources = workerState.videoModelSources.filter(
    (url) => url && !workerState.disabledVideoSources.has(url),
  );
  if (!sources.length) {
    throw new Error("Video model URL not configured");
  }
  const load = (async () => {
    const threads = Math.max((self.navigator?.hardwareConcurrency || 4) / 2, 1);
    let lastError = null;
    for (const url of sources) {
      try {
        const model = await self.tflite.loadTFLiteModel(url, {
          numThreads: threads,
        });
        workerState.videoModel = model;
        return model;
      } catch (primaryError) {
        console.warn("Failed to load video model from URL", url, primaryError);
        lastError = primaryError;
        workerState.disabledVideoSources.add(url);
        try {
          const buffer = await fetchModelFromCache(url);
          const model = await self.tflite.loadTFLiteModel(buffer, {
            numThreads: threads,
          });
          workerState.videoModel = model;
          return model;
        } catch (fallbackError) {
          console.error("Failed to load video model from cache", url, fallbackError);
          workerState.disabledVideoSources.add(url);
          lastError = fallbackError;
        }
      }
    }
    throw lastError || new Error("Video model URL not configured");
  })()
    .catch((error) => {
      workerState.videoModel = null;
      throw error;
    })
    .finally(() => {
      workerState.videoModelPromise = null;
    });
  workerState.videoModelPromise = load;
  return load;
}

function ensureVideoModelLoading() {
  if (workerState.videoModel || workerState.videoModelPromise) {
    return;
  }
  loadVideoModel().catch((error) => {
    console.error("Video model preload failed", error);
  });
}

function downsamplePcm(samples, sourceRate, targetRate = 16000) {
  if (!(samples instanceof Float32Array) || !samples.length || !sourceRate) {
    return new Float32Array();
  }
  if (sourceRate === targetRate) {
    return samples.slice();
  }
  const ratio = sourceRate / targetRate;
  const newLength = Math.floor(samples.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = samples[Math.floor(i * ratio)];
  }
  return result;
}

async function loadClassMap() {
  try {
    const response = await fetch(workerState.config?.classMapUrl, {
      cache: "force-cache",
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch class map: ${response.statusText}`);
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
  try {
    const response = await fetch(workerState.config?.videoClassMapUrl, {
      cache: "force-cache",
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch video class map: ${response.statusText}`);
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

function buildMelFilterbank(
  numMelBins,
  numSpectrogramBins,
  sampleRate,
  lowerEdgeHz,
  upperEdgeHz,
) {
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
  const linearFreqs = (sampleRate / 2) / (numSpectrogramBins - 1);
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

  return self.tf.tensor2d(filterbank, [numMelBins, numSpectrogramBins]);
}

async function fetchModelFromCache(url) {
  if (self.caches) {
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

async function loadScriptWithFallback(urls) {
  for (const url of urls) {
    try {
      importScripts(url);
      return;
    } catch (error) {
      console.warn("Script load failed, trying next URL", url, error);
    }
  }
  throw new Error("All script sources failed to load");
}

function resolveWorkerUrl(url, base) {
  if (!url || /^[a-zA-Z]+:/.test(url) || url.startsWith("/")) {
    return url;
  }
  if (url.startsWith("./") || url.startsWith("../")) {
    return new URL(url, `${base}/`).toString();
  }
  return `${base.replace(/\/+$/, "")}/${url}`;
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

function serializeError(error) {
  return {
    message: error?.message || String(error),
    stack: error?.stack || null,
  };
}
