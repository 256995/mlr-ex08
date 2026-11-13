# better_classifier — the Task 4 page and the experiments behind it

A second leaf-classifying page, next to the Task 2 one, running a model this folder
trained. Nothing here replaces `../index.html`; open this page at
`better_classifier/index.html` and the old one stays exactly as it was.

- **`RESULTS.md`** — every number: the variant table, what each change brought, whether it
  is larger than the noise, and the before/after of the two model files.
- **`TRIED_AND_NOT_USED.md`** — what was tried and thrown away, with the measurement that
  threw it away.
- **`REPORT.md` is deliberately missing**: the notebook requires it to be the student's own
  writing, so it is not written here.

## What the page is

The Task 2 page, changed as little as possible:

| | |
| --- | --- |
| model | `leaf_model_v2.onnx` + its `model_info.json`, `check_image.jpg`, `check_expected.json` |
| QR code | encodes the address the page was actually loaded from, so it opens the page you are looking at (GitHub Pages, a laptop on the same wifi, …); `PUBLISHED_URL` is only the fallback for a page opened from disk |
| new behaviour | one line under the bars when the top probability is under `UNSURE_BELOW = 0.6`: it says which two classes are close and tells you to get closer |
| everything else | unchanged: preprocessing from `model_info.json`, softmax on the logits, one wasm thread, `proxy = false`, pinned `onnxruntime-web@1.19.2`, nothing leaves the phone |

## Files

| file | what it is |
| --- | --- |
| `index.html` | the page |
| `train_leaves.py` | the experiments: baseline + one change per variant, camera-like test set, ONNX export |
| `measure_onnx.py` | measures any `.onnx` with the page's own preprocessing and one thread |
| `quantize_model.py` | the 8-bit experiment (kept; it failed, see `TRIED_AND_NOT_USED.md`) |
| `strip_export_metadata.py` | run after `--export`: removes the exporter's per-node stack traces, which quote the exporting machine's paths |
| `run_experiments.sh` | the whole matrix in five parallel streams, as one command |
| `leaf_model_v2.onnx` | variant `full`, seed 0, last epoch — 6.25 MB (see below) |
| `model_info.json` | classes, `input_size` 96, the mean/std the model was trained with |
| `check_image.jpg`, `check_expected.json` | the built-in check: a camera-like photo whose answer is *not* decisive (0.802 / 0.169), so the check actually checks |
| `pyproject.toml` | uv: torch, torchvision, onnx, onnxscript, onnxruntime, pillow; ruff in the dev group |

### Why the shipped `.onnx` is 6.25 MB and `--export` writes 6.38 MB

`torch.onnx.export` (torch 2.15, opset 20) attaches a `pkg.torch.onnx.stack_trace` entry to
**every one of the 122 nodes**, and those Python tracebacks contain the absolute path of the
machine that did the exporting — the interpreter inside `.venv`, the username, the project
folder. A model file that is about to sit in a public repository should not carry that, so
those 122 entries are deleted by `strip_export_metadata.py`, which is what was run on the
shipped file (its output is byte-for-byte the file in this folder). Nothing else was touched:
122 nodes with the same names, op types, inputs, outputs and attributes, 110 initializers with
byte-identical weights, `onnx.checker` passes, and inference on `check_image.jpg` is
**bit-for-bit identical** (logits `2.490443, 0.934786, -1.606177, -1.468719`). Every number in
`RESULTS.md` was reproduced on the stripped file: validation 0.875, camera-like 0.900, the same
confidence bands, 1.0 ms per classification, largest difference from `check_expected.json`
1.79e-07, and the page's own browser check still PASS at 9.06e-8.

## Running it

uv only, from this folder. This machine's console is cp1250 and `python -I` ignores
`PYTHONUTF8`, so runs that print anything non-ASCII need `-X utf8`:

```bash
uv sync --group dev                                  # once
export TORCH_HOME=../_dev/torch_cache                # keep the pretrained weights inside the project

uv run python -I -X utf8 train_leaves.py --variant baseline --seed 0     # one run, ~2 minutes
uv run python -I -X utf8 train_leaves.py --variant full --seeds 3        # three seeds
uv run python -I -X utf8 train_leaves.py --summarise                     # the table from ../_dev/runs
uv run python -I -X utf8 train_leaves.py --variant full --seed 0 --export
uv run python -I -X utf8 strip_export_metadata.py leaf_model_v2.onnx      # before shipping it
uv run python -I -X utf8 measure_onnx.py leaf_model_v2.onnx              # any .onnx, incl. ../leaf_model.onnx
uv run --group dev ruff check . && uv run --group dev ruff format --check .
```

The pictures are not in the folder: `train_leaves.py` expects the course's historical
archive (175 + 30 photos per species) unpacked in `../_dev/leaves_data/leaves/{train,val}`,
from <https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip>.
The camera-like set is derived from `val` once, with seed 0, into
`../_dev/leaves_data/leaves_cam`, and is never used to choose a model. Delete that folder
only if you mean to rebuild it — the whole table depends on those being the same pictures.

## Checking the page for real

`?selftest=1` proves the pure logic only. The camera loop needs real time over CDP, so:

```bash
# 1. serve the repository root (the page lives in a subfolder of it)
uv run --no-project python -m http.server 8731 --bind 127.0.0.1 &
# 2. headless Chrome with a fake rear camera, profile kept inside the project
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --remote-debugging-port=9222 \
    --user-data-dir="$PWD/_dev/chrome_profile" --use-fake-device-for-media-stream \
    --use-fake-ui-for-media-stream --no-first-run about:blank &
# 3. drive it: built-in check, the not-sure line, the QR, the live loop, every request made
uv run --no-project python -I -X utf8 _dev/test_better_page.py 9222 http://127.0.0.1:8731/better_classifier/
```

Last run of that, on 2026-11-12: check **PASS at 9.06e-8** (limit 0.001), live loop
"cherry — 0.7685 · 24 ms per frame", the not-sure line shown for a 0.31/0.25 split and
hidden for a 0.94 answer, both uploads classified (`photo_small.jpg` 1600×1200 → 98.14 %,
`photo_huge.jpg` 8000×6000 → 98.03 %, canvas capped to `1024x768` both times), QR text equal
to the address served (29 modules, mask 7), **no POSTs and no data: URLs**, no console
errors. Full output in `../_dev/better_report.json`, screenshots `../_dev/better_shot.png`
and `../_dev/better_shot_footer.png`. The upload tests need
`../_dev/photo_small.jpg` (any normal photo) and `../_dev/photo_huge.jpg` (over 16 million
pixels) — both are there now, rebuilt from `_dev/leaves_data/leaves_cam/apple/apple_000.jpg`:

```bash
uv run --no-project --with pillow python -I -X utf8 -c \
  "from PIL import Image; im=Image.open('_dev/leaves_data/leaves_cam/apple/apple_000.jpg').convert('RGB'); \
   im.resize((1600,1200)).save('_dev/photo_small.jpg'); im.resize((8000,6000)).save('_dev/photo_huge.jpg')"
```

The QR grid alone is verified independently of the page's own encoder: it is painted as a PNG
and decoded by zxing-cpp, and rebuilt by the `qrcode` library **forced to the page's version
and mask** — left free, that library picks its own mask, and a different mask flips about half
the data modules, which is what once made this comparison report "528 modules differ" about a
code that decoded perfectly.

```bash
uv run --no-project --with pillow --with zxing-cpp --with qrcode python -I -X utf8 \
    _dev/qr_check.py _dev/qr_served.json          # <expected-url> is an optional second argument
```

Both of the codes the page can draw were checked this way (`_dev/qr_served.json` and
`_dev/qr_published.json`, dumped from the live page with `JSON.stringify(window.__qr)` and
`QR.draw(…, PUBLISHED_URL, 8)`): **0 modules differ out of 841** (version 3, mask 7, the
localhost address) and **0 out of 1089** (version 4, mask 0, the published address), and zxing
reads each as exactly its own text.

## Deploying it

Publish this folder as `better_classifier/` next to the Task 2 page, then open the page on a
phone and scan the code on its screen. The QR encodes whatever address the page was loaded
from, so the other way works too: `python -m http.server` on a laptop and the phone on the
same wifi (the camera needs `https://` or `localhost`, so for a LAN address the camera will
say so and the photo upload is what works). `PUBLISHED_URL` is what a page opened from disk
falls back to — until this folder is actually published, that URL is a 404 and the QR built
from it opens nothing. Checked over the network on 2026-11-12:
`https://256995.github.io/mlr-ex08/` answers **200**,
`https://256995.github.io/mlr-ex08/better_classifier/` answers **404** — the Task 2 page is
live, this folder is not. Publishing is an ordinary push: this copy has `origin` at
`github.com/256995/mlr-ex08`, its `main` is level with `origin/main`, and the folder stays
where it is in the repository — the Pages address is exactly `…/mlr-ex08/better_classifier/`.

## Fixed after the first phone test (2026-11-12)

Reported from a phone: photos did not load and the QR did not work.

- **The QR pointed at a page that had never been published** — a correct, scannable code
  for a 404. Now the address comes from `location.href` (see above). The encoder itself was
  checked and is fine: 0 modules differ from a reference encoder at the same version, level
  and mask (`RESULTS.md`).
- **A model that failed to load was reported as "still loading" — forever.** Uploading a
  photo then said *"The model is still loading — try again in a moment"* even though the
  download had already failed, which is exactly what "the page does not load photos" looks
  like from the outside. `loadModel()` now records the reason (`modelError`), and three
  places state it: the upload, the camera loop, and the verdict line — *"The check cannot
  run: the model did not load (Failed to fetch)."* — because *Run the check* stays disabled
  for the rest of the page's life and a disabled button explains nothing on its own.
- **An uploaded photo is no longer kept at full resolution.** The page allocated a canvas at
  the photo's own size; Safari on iOS refuses canvases over ~16 million pixels and hands
  back a blank one, so a photo from a 48 MP camera could not be classified at all. The
  snapshot is capped at `MAX_SNAPSHOT = 1024` — the model reads one 96×96 centre crop either
  way. The built-in check is unaffected: `check_image.jpg` is 96×96 and is drawn at 1:1.

Verified afterwards in headless Chrome (`_dev/test_better_page.py`, which now also uploads
two photos through `DOM.setFileInputFiles`): check still PASS at 9.06e-8, a 1600×1200 and an
8000×6000 photo both classify (`1024x768` canvas, probabilities matching each other),
the live camera loop unchanged at 24 ms/frame, QR text equal to the address served, still
0 POSTs and no console errors. **Not verified on an iPhone** — the canvas limit is Safari
documented behaviour, exercised here on desktop Chrome, which tolerates 48 MP happily.

### The three addresses a phone can actually open it at

All three served to headless Chrome on this machine, 2026-11-12, and all three are what a
scan of the code on the screen would give you:

| opened at | model | QR encodes | camera |
| --- | --- | --- | --- |
| `http://127.0.0.1:8731/better_classifier/` | loads, check PASS | that address (v3, mask 7) | works (localhost is secure) |
| `http://192.168.0.157:8732/better_classifier/` | loads, check PASS | that address | refuses, and says: *"The camera needs HTTPS (or localhost) — this page was opened at http:, where browsers do not hand out a camera. The photo upload still works."* |
| `file:///…/better_classifier/index.html` | fails (`fetch` of `model_info.json` is refused), and the page now says so in both places | falls back to `PUBLISHED_URL` (v4, mask 0) | — |

The LAN row is the one that matters for the "run it from a laptop" route in *Deploying it*,
and it is the only context where the `isSecureContext` message had ever been reached —
`file://` counts as a secure context in Chrome, so it does not trigger it. In the LAN context
both uploads classified (98.14 % / 98.03 %, canvas `1024x768`), and the only hosts contacted
were `192.168.0.157:8732` and `cdn.jsdelivr.net`.

**A page opened from `file://` cannot work, and now explains why rather than appearing
frozen.** If the phone report came from opening the HTML file itself, that is the whole
answer: the model has to be served over http.
