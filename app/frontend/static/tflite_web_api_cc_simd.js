// Shim file to keep TensorFlow Lite's relative import happy when running
// inside a Web Worker. The real implementation lives under /static/tflite/.
importScripts("/static/tflite/tflite_web_api_cc_simd.js");
