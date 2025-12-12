/* eslint-disable no-restricted-globals */
const globalScope = self;

// SmolVLM Worker state
const smolvlmState = {
  modelId: null,
  model: null,
  processor: null,
  rawImageCtor: null,
  modelPromise: null,
  runtimePromise: null,
  device: null,
  loaded: false,
  initializing: false,
};

// Constants
const TRANSFORMERS_WASM_BASE = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/";
const SMOLVLM_DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-256M-Instruct";
const SMOLVLM_VIDEO_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";
const SMOLVLM_OPTIONS = {
  frameCount: 8,
  maxNewTokens: 192,
};

function normalizeSmolVlmText(text) {
  if (text == null) {
    return null;
  }
  let cleaned = typeof text === "string" ? text : String(text);
  cleaned = cleaned.trim();
  if (!cleaned) {
    return null;
  }
  const assistantMarker = "Assistant:";
  const markerIndex = cleaned.lastIndexOf(assistantMarker);
  if (markerIndex >= 0) {
    cleaned = cleaned.slice(markerIndex + assistantMarker.length);
  }
  cleaned = cleaned.replace(/^\s*assistant\s*:\s*/i, "");
  const userMarkerIndex = cleaned.indexOf("User:");
  if (userMarkerIndex > 0) {
    cleaned = cleaned.slice(0, userMarkerIndex);
  }
  cleaned = cleaned.trim();
  return cleaned || null;
}

// Message handlers
globalScope.addEventListener("message", async (event) => {
  const { id, type, payload } = event.data || {};
  
  try {
    switch (type) {
      case "INIT":
        await initWorker();
        respond(id, { ok: true, loaded: smolvlmState.loaded });
        break;
        
      case "PRELOAD":
        await preloadModel(payload?.modelId || SMOLVLM_DEFAULT_MODEL);
        respond(id, { ok: true, loaded: smolvlmState.loaded });
        break;
        
      case "DESCRIBE_FRAMES":
        const result = await describeFrames(payload);
        respond(id, result);
        break;
        
      case "GET_STATUS":
        respond(id, {
          ok: true,
          loaded: smolvlmState.loaded,
          modelId: smolvlmState.modelId,
          device: smolvlmState.device,
        });
        break;
        
      case "CLEAR_MODEL":
        await clearModel();
        respond(id, { ok: true });
        break;
        
      default:
        respond(id, { ok: false, error: `Unknown message type: ${type}` });
        break;
    }
  } catch (error) {
    respondError(id, error);
  }
});

function respond(id, result) {
  globalScope.postMessage({ id, success: true, result });
}

function respondError(id, error) {
  const detail = error instanceof Error ? error.message : String(error);
  globalScope.postMessage({ id, success: false, error: detail });
}

async function initWorker() {
  if (smolvlmState.initialized) {
    return;
  }
  smolvlmState.initialized = true;
  
  // Load transformers.js runtime
  await loadTransformersRuntime();
}

async function loadTransformersRuntime() {
  if (smolvlmState.runtimePromise) {
    return smolvlmState.runtimePromise;
  }
  
  const load = async () => {
    if (globalScope.transformers) {
      return globalScope.transformers;
    }
    
    const module = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1");
    const runtime = module?.default ?? module;
    
    if (runtime?.env) {
      runtime.env.allowLocalModels = false;
      runtime.env.allowRemoteModels = true;
      runtime.env.useCache = false;
      runtime.env.useBrowserCache = false;
      runtime.env.useFSCache = false;
      
      if (runtime.env.backends?.onnx?.wasm) {
        const wasm = runtime.env.backends.onnx.wasm;
        wasm.wasmPaths = TRANSFORMERS_WASM_BASE;
        wasm.proxy = false;
        wasm.numThreads = 1;
      }
    }
    
    return runtime;
  };
  
  smolvlmState.runtimePromise = load().catch((error) => {
    smolvlmState.runtimePromise = null;
    throw error;
  });
  
  return smolvlmState.runtimePromise;
}

async function preloadModel(modelId = SMOLVLM_DEFAULT_MODEL) {
  const targetModelId = modelId || SMOLVLM_DEFAULT_MODEL;
  
  if (smolvlmState.model && smolvlmState.processor && smolvlmState.modelId === targetModelId) {
    return {
      model: smolvlmState.model,
      processor: smolvlmState.processor,
      rawImageCtor: smolvlmState.rawImageCtor,
      device: smolvlmState.device,
    };
  }
  
  if (smolvlmState.modelPromise && smolvlmState.modelId === targetModelId) {
    return smolvlmState.modelPromise;
  }
  
  const runtime = await loadTransformersRuntime();
  const { AutoProcessor, AutoModelForVision2Seq, RawImage } = runtime;
  
  const userAgent = globalScope.navigator?.userAgent || "";
  const targetDevice = /\bAndroid\b/i.test(userAgent) ? "wasm" : "webgpu";
  
  const loadForDevice = async (device) => {
    const processor = await AutoProcessor.from_pretrained(targetModelId);
    const model = await AutoModelForVision2Seq.from_pretrained(targetModelId, {
      device,
      dtype: "fp32",
    });
    
    // Configure processor for memory efficiency
    if (processor?.image_processor) {
      const img = processor.image_processor;
      img.do_image_splitting = false;
      if (img.config) {
        img.config.do_image_splitting = false;
        img.config.do_image_splitting_layers = [];
        img.config.image_size = { height: 224, width: 224 };
        img.config.size = { height: 224, width: 224 };
      }
    }
    
    return { processor, model, rawImageCtor: RawImage, device };
  };
  
  const promise = (async () => {
    try {
      return await loadForDevice(targetDevice);
    } catch (primaryError) {
      console.warn("WebGPU failed for SmolVLM2; retrying with WASM", primaryError);
      return loadForDevice("wasm");
    }
  })();
  
  smolvlmState.modelId = targetModelId;
  smolvlmState.modelPromise = promise
    .then((result) => {
      smolvlmState.model = result.model;
      smolvlmState.processor = result.processor;
      smolvlmState.rawImageCtor = result.rawImageCtor;
      smolvlmState.device = result.device;
      smolvlmState.loaded = true;
      smolvlmState.modelPromise = null;
      return result;
    })
    .catch((error) => {
      smolvlmState.modelPromise = null;
      smolvlmState.modelId = null;
      smolvlmState.loaded = false;
      throw error;
    });
  
  return smolvlmState.modelPromise;
}

async function describeFrames(payload) {
  const { frames, modelId, prompt } = payload;
  
  if (!frames || !frames.length) {
    return { text: null, latencyMs: null, device: null };
  }
  
  let modelArtifacts;
  
  try {
    modelArtifacts = await preloadModel(modelId || SMOLVLM_DEFAULT_MODEL);
    
    if (!modelArtifacts?.processor || !modelArtifacts?.model) {
      return { text: null, latencyMs: null, device: modelArtifacts?.device ?? null };
    }
    
    const getInputTokenCount = (inputIds) => {
      if (!inputIds) {
        return null;
      }
      if (Array.isArray(inputIds)) {
        const first = inputIds[0];
        return Array.isArray(first) ? first.length : inputIds.length;
      }
      if (ArrayBuffer.isView(inputIds)) {
        return inputIds.length;
      }
      const dims = inputIds.dims || inputIds.shape;
      if (Array.isArray(dims) && dims.length >= 2) {
        return dims[dims.length - 1];
      }
      const data = inputIds.data;
      if (ArrayBuffer.isView(data)) {
        return data.length;
      }
      return null;
    };

    const getFirstSequenceTokenIds = (sequences) => {
      if (!sequences) {
        return null;
      }
      if (Array.isArray(sequences)) {
        const first = sequences[0];
        return Array.isArray(first) ? first : sequences;
      }
      if (ArrayBuffer.isView(sequences)) {
        return sequences;
      }
      const data = sequences.data;
      if (ArrayBuffer.isView(data)) {
        return data;
      }
      if (typeof sequences.tolist === "function") {
        try {
          const list = sequences.tolist();
          if (Array.isArray(list)) {
            const first = list[0];
            return Array.isArray(first) ? first : list;
          }
        } catch {
          // ignore tensor conversion failures
        }
      }
      return null;
    };

    const decodeTokenIds = (tokenIds, tokenizer) => {
      if (!tokenIds || !tokenizer) {
        return null;
      }
      const ids = Array.isArray(tokenIds) ? tokenIds : Array.from(tokenIds);
      if (typeof tokenizer.decode === "function") {
        return tokenizer.decode(ids, { skip_special_tokens: true });
      }
      if (typeof tokenizer.batch_decode === "function") {
        const decoded = tokenizer.batch_decode([ids], { skip_special_tokens: true });
        return Array.isArray(decoded) ? decoded[0] : decoded;
      }
      return null;
    };

    let scratchCanvas = null;
    let scratchCtx = null;
    let scratchWidth = 0;
    let scratchHeight = 0;
    const ensureScratchCanvas = (width, height) => {
      if (
        scratchCanvas &&
        scratchCtx &&
        scratchWidth === width &&
        scratchHeight === height
      ) {
        return scratchCtx;
      }
      if (typeof OffscreenCanvas === "undefined") {
        return null;
      }
      try {
        scratchCanvas = new OffscreenCanvas(width, height);
        scratchCtx = scratchCanvas.getContext("2d");
        scratchWidth = width;
        scratchHeight = height;
        return scratchCtx;
      } catch {
        scratchCanvas = null;
        scratchCtx = null;
        scratchWidth = 0;
        scratchHeight = 0;
        return null;
      }
    };

    // Convert frame data to RawImage objects
    const processedFrames = [];
    for (const frameData of frames) {
      try {
        const { width, height, data } = frameData;
        if (!width || !height || !data) continue;
        
        // Create ImageData from frame data
        const imageData = new ImageData(
          new Uint8ClampedArray(data),
          width,
          height
        );
        
        // Convert to RawImage
        let rawImage;
        if (typeof modelArtifacts.rawImageCtor.fromCanvas === "function") {
          const ctx = ensureScratchCanvas(width, height);
          if (ctx && scratchCanvas) {
            ctx.putImageData(imageData, 0, 0);
            rawImage = modelArtifacts.rawImageCtor.fromCanvas(scratchCanvas);
          }
        }
        if (!rawImage && typeof modelArtifacts.rawImageCtor.fromImageData === "function") {
          rawImage = modelArtifacts.rawImageCtor.fromImageData(imageData);
        }
        
        if (rawImage) {
          processedFrames.push(rawImage);
        }
      } catch (error) {
        console.warn("Failed to process frame", error);
      }
    }
    
    if (!processedFrames.length) {
      return { text: null, latencyMs: null, device: modelArtifacts.device };
    }
    
    const finalFrames = selectFrames(processedFrames, payload?.frameCount ?? SMOLVLM_OPTIONS.frameCount);
    
    const buildInputs = async (frameList) => {
      const messages = [
        {
          role: "user",
          content: [
            ...frameList.map(() => ({ type: "image" })),
            { type: "text", text: prompt ?? "Describe what happens in the video." },
          ],
        },
      ];
      const templated = modelArtifacts.processor.apply_chat_template(
        messages,
        { add_generation_prompt: true, tokenize: false },
      );
      return modelArtifacts.processor(
        templated,
        frameList,
        {
          return_tensors: "np",
          return_row_col_info: false,
          do_image_splitting: false,
          image_size: { height: 224, width: 224 },
        },
      );
    };
    
    let inputs;
    try {
      inputs = await buildInputs(finalFrames);
    } catch (error) {
      console.warn("SmolVLM2 processor failed; retrying with a single frame", error);
      const fallbackFrame = finalFrames[Math.floor(finalFrames.length / 2)] ?? finalFrames[0];
      inputs = await buildInputs([fallbackFrame]);
    }
    
    const runGeneration = async (inputTensors) => {
      const start = performance.now();
      const generation = await modelArtifacts.model.generate({
        ...inputTensors,
        max_new_tokens: SMOLVLM_OPTIONS.maxNewTokens,
      });
      const latencyMs = Math.max(Math.round(performance.now() - start), 1);
      const sequences = generation?.sequences ?? generation;
      const tokenIds = getFirstSequenceTokenIds(sequences);
      const inputTokenCount = getInputTokenCount(inputTensors?.input_ids);
      const generatedTokenIds =
        tokenIds &&
        Number.isFinite(inputTokenCount) &&
        inputTokenCount > 0 &&
        tokenIds.length > inputTokenCount
          ? tokenIds.slice(inputTokenCount)
          : tokenIds;
      const summary = decodeTokenIds(generatedTokenIds, modelArtifacts.processor.tokenizer);
      return { text: normalizeSmolVlmText(summary), latencyMs };
    };
    
    try {
      const { text, latencyMs } = await runGeneration(inputs);
      return {
        text,
        latencyMs,
        device: modelArtifacts.device,
      };
    } catch (generationError) {
      const message = generationError?.message || String(generationError);
      const tokenFeatureMismatch = message.includes("Number of tokens and features do not match");
      if (tokenFeatureMismatch && finalFrames.length > 1) {
        console.warn("SmolVLM2 token/feature mismatch; retrying with a single frame", generationError);
        const fallbackFrame = finalFrames[Math.floor(finalFrames.length / 2)] ?? finalFrames[0];
        try {
          const singleInputs = await buildInputs([fallbackFrame]);
          const { text, latencyMs } = await runGeneration(singleInputs);
          return {
            text,
            latencyMs,
            device: modelArtifacts.device,
          };
        } catch (retryError) {
          console.warn("SmolVLM2 retry after mismatch failed", retryError);
        }
      }
      console.warn("SmolVLM2 generation failed", generationError);
      return {
        text: null,
        latencyMs: null,
        device: modelArtifacts.device,
        error: message,
      };
    }
  } catch (fatalError) {
    console.warn("SmolVLM2 describeFrames failed", fatalError);
    return {
      text: null,
      latencyMs: null,
      device: modelArtifacts?.device ?? null,
      error: fatalError?.message || String(fatalError),
    };
  }
}

function selectFrames(frameList, targetCount = SMOLVLM_OPTIONS.frameCount) {
  if (!frameList?.length) {
    return [];
  }
  
  const count = Math.max(Math.min(targetCount || frameList.length, frameList.length), 1);
  if (count === frameList.length) {
    return frameList;
  }
  
  const selected = [];
  const step = frameList.length / count;
  for (let i = 0; i < count; i += 1) {
    const index = Math.floor(i * step);
    selected.push(frameList[index]);
  }
  
  return selected;
}

async function clearModel() {
  // Clear model resources but keep runtime loaded
  smolvlmState.model = null;
  smolvlmState.processor = null;
  smolvlmState.rawImageCtor = null;
  smolvlmState.modelId = null;
  smolvlmState.modelPromise = null;
  smolvlmState.loaded = false;
}
