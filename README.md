# mlr-ex08 — leaf classifier on your phone

**Open the page here: <https://256995.github.io/mlr-ex08/>**

Scan the QR code at the bottom of the page with your phone camera to open it.

A single `index.html` that classifies leaf photographs with the MobileNetV3
network exported from the course notebook — running entirely in the browser of
your phone, through [ONNX Runtime Web] loaded from a CDN.

## What the page does

1. **Live rear camera.** *Start rear camera* opens the back lens
   (`facingMode: "environment"`) and classifies a new frame about every
   250 ms. A square frame is drawn over the video: that square is exactly what
   the model receives.
2. **Photo upload.** If the camera is not available (desktop, permissions,
   an old browser), upload any picture instead — the same preprocessing and the
   same network are used.
3. **Every class probability, live.** `apple`, `cherry`, `chestnut` and `maple`
   are each shown as a bar and a percentage, with the winning class highlighted.
4. **A built-in check.** On load the page runs `check_image.jpg` through its
   own preprocessing and prints a table comparing what it got with
   `check_expected.json` from the notebook. Largest difference ≤ 0.001 → **PASS**.

## The preprocessing (matches `model_info.json` exactly)

`model_info.json` says how the model must be fed:

```json
{"classes": ["apple", "cherry", "chestnut", "maple"], "input_size": 96,
 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
```

In `index.html`, `classifySource()` is the notebook's
`transforms.Compose([ToTensor(), Normalize(mean, std)])` applied to the centre
of the picture, step by step:

| step | notebook | `index.html` |
| --- | --- | --- |
| crop | images were square-cropped when the data was packed | `Math.min(width, height)`, centred: `cropX = (width - square) / 2` |
| resize | already 96 × 96 | `drawImage(…, square, square, 0, 0, 96, 96)` with `imageSmoothingQuality = "high"` |
| 0–1 | `ToTensor()` divides by 255 | `rgba[p] / 255` |
| standardize | `Normalize(mean, std)` | `(rgba[p] / 255 - info.mean[c]) / info.std[c]` |
| layout | `ToTensor()` moves channels first | three consecutive planes R, G, B → `float32` tensor of shape `1 × 3 × 96 × 96` |

The exported network outputs **logits** (`output_names=["logits"]`; softmax was
inside `nn.CrossEntropyLoss` during training), so the page applies `softmax()`
itself before showing probabilities.

## Nothing leaves the phone

The page requests only these URLs:

- `index.html`, `leaf_model.onnx`, `model_info.json`, `check_image.jpg`,
  `check_expected.json` — from this GitHub Pages site;
- `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js`,
  `ort-wasm-simd-threaded.mjs` and `ort-wasm-simd-threaded.wasm` — the
  inference engine itself: 11 MB of WebAssembly, fetched once and then cached.

No picture, thumbnail or pixel is ever sent anywhere; `POST` requests: none.
The only way pixels could leave would be a network call carrying them, and
there is none — classification runs on the device's WebAssembly.

## iOS Safari notes

- `<video playsinline>` + a *Start camera* button, because iPhone only starts a
  camera after a real tap, and only inline rather than in the fullscreen player.
- `ort.env.wasm.numThreads = 1` — GitHub Pages cannot send COOP/COEP headers,
  so `SharedArrayBuffer` (and therefore threaded WASM) is unavailable; asking
  for threads makes Safari fail to start the runtime.
- The runtime version is pinned by two measurements, written down in
  `index.html`: it has to be newer than 1.17 (this model is exported with ONNX
  **IR version 10**, which 1.17 refuses — it aborts with a bare number and no
  message), and it stays on **1.19.2** because that release loads the 11 MB
  `ort-wasm-simd-threaded.wasm`, while 1.22 and newer hard-code the WebGPU
  "jsep" binary at 22 MB. Half the download on a phone.
- With `numThreads = 1` the runtime starts even though the page is *not*
  cross-origin isolated — checked in a real browser with `crossOriginIsolated`
  reported as `false`, which is exactly what GitHub Pages serves.
- HTTPS is required for camera access, which GitHub Pages provides.

## Files

| file | what it is |
| --- | --- |
| `index.html` | the whole page: HTML, CSS, plain JavaScript, QR code encoder |
| `leaf_model.onnx` | the trained MobileNetV3 (6.3 MB) with `model_info` metadata |
| `model_info.json` | classes, input size, channel mean and std |
| `check_image.jpg` | one 96 × 96 validation photo |
| `check_expected.json` | the probabilities the notebook's PyTorch model gives it |

The QR code at the bottom of the page is drawn module by module by the
JavaScript in `index.html` (byte mode, error correction level M, ISO/IEC 18004)
— no QR image service is contacted, so nothing about the URL is sent to anyone.

## For the teacher

`MLRteacher` is invited as a collaborator on this repository. The hand-in files
`REVIEW.md` (your check of the page, with a phone screenshot) and `REPORT.md`
(Task 4) are written by the student, not by the agent, so they are not here yet.

[ONNX Runtime Web]: https://onnxruntime.ai/docs/tutorials/web/
