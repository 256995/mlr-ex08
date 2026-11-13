# Tried, measured, not used

Everything that was attempted for this folder and did **not** end up in
`leaf_model_v2.onnx` or in `index.html`, with the number that decided it. Accuracy columns
are always: validation (120 photos) / **camera-like (120 photos, seed 0, last epoch)** — the
second one is the column that decides, see `RESULTS.md`. "Noise" means a difference of
about 0.02 or less, which one photo in 120 and one new seed can produce by themselves.

## From the notebook's list of candidates

**Frozen backbone — transfer learning with only the new last layer trained.**
0.781 / 0.650, three seeds, worse than the baseline (0.900 / 0.739) on *both* columns.
With 175 photos per class the ImageNet features are useful but clearly not tuned to leaves,
and one linear layer on top cannot fix that. The notebook suggests comparing it against
fine-tuning; the answer here is fine-tuning, by 0.09 on the camera-like set.

**The Task 3 network from random weights (`LeafNet`).** 0.883 / **0.600**, one seed.
Its validation number looks almost as good as MobileNet's, and its camera-like number is
the worst in the whole table: 0.28 *below* its own validation column, where the shipped
model is 0.02 above its own. Shipping it would also mean a new `model_info.json` and a
fresh PyTorch-to-ONNX check, for a network 0.30 worse where it matters. Kept in
`train_leaves.py` as the `scratch` row, because it is the clearest demonstration of why
requirement 2 exists.

**AdamW with weight decay 1e-4.** 0.887 / 0.762, two seeds. No gain over plain Adam
(0.900 / 0.739) that would survive a third seed, and augmentation alone beats it by 0.09.
One more knob to explain in the report for nothing measured.

**Colour jitter as a single change.** `augment` 0.856 -> `colour` 0.850, three seeds. That
0.006 is less than one validation photo, and it sits inside `augment`'s own seed spread of
+-0.038, so as a single change on 15 epochs colour jitter is worth nothing measurable.
**Used, but not as the notebook suggests trying it**: inside the full stack (30 epochs,
cosine, label smoothing) removing colour jitter costs 0.086 (`augment_cosine` 0.800 vs
`combined` 0.886), on all three seeds. On its own 0.000, in the combination 0.086 — anyone
reading only the single-change table would cut it and be wrong to.

**Keeping the weights of the best epoch.** Measured, not assumed: the best-epoch
checkpoints (best validation loss, epochs 17/18/25 of 30 for the three seeds of `full`)
average 0.897 on the camera-like set against 0.911 for the last epoch. On seed 1 the best
epoch is 0.041 *worse* on the camera-like set while being 0.008 better on validation.
`train_leaves.py` still saves both (`<name>.pt` and `<name>-best.pt`) and `--weights best`
still works, but the page ships the last epoch — the only checkpoint the camera-like column
ever measured.

**Dropout / global average pooling instead of flattening.** Nothing to add. The pretrained
head is already `Linear(576,1024) -> Hardswish -> Dropout(p=0.2) -> Linear(1024,4)`, and
MobileNetV3 pools spatially down to 1x1 before the head, so "add dropout" and "use global
average pooling" describe features the shipped model already has. GAP-instead-of-flatten
would only apply to `LeafNet`, which is not shipped. Not measured as a variant; checked by
printing `model.classifier` and the pooling layer.

**More data — last year's photographs.** The biggest single effect visible in these numbers,
and the one row that is *not* a fair comparison, so it is not in the table. The model the
phone runs today was trained by the notebook on the group's own ZIP and scores 0.450 here;
the same recipe retrained on the historical archive (`baseline`) scores 0.739. +0.29, but
that gap mixes "700 photos instead of the group's" with "different students, different
phones, different light" — it is not one change. The notebook's actual suggestion, *adding*
the historical photos to the group's own, could not be measured in this copy at all: only
the historical archive is here, the group ZIP is not.

## On the export / phone side

**8-bit weights (dynamic quantization).** The most tempting idea in the folder — the phone
downloads the model before it can classify anything, so a quarter of the bytes is a quarter
of the waiting. It fails three separate ways:

1. `quantize_dynamic` cannot read the file this torch exports at all:
   `onnx.onnx_cpp2py_export.shape_inference.InferenceError: [ShapeInferenceError] Inferred
   shape and existing shape differ in dimension 0: (576) vs (1024)`.
2. Exporting the *same weights* with the legacy exporter (`dynamo=False`) gives a graph it
   can quantize, 6.10 MB -> 1.69 MB. So I measured that file: validation **0.250**,
   camera-like **0.258** — chance for four classes, from a network that scored 0.875 / 0.900
   in 32-bit. To repeat it:

   ```bash
   uv run python -I -X utf8 -c "
   import sys; sys.path.insert(0, '.'); import torch
   from train_leaves import BASE, VARIANTS, RUNS, SIZE, build_network
   m = build_network(BASE | VARIANTS['full'])
   m.load_state_dict(torch.load(RUNS / 'full-seed0.pt', map_location='cpu')); m.eval()
   torch.onnx.export(m, (torch.zeros(1,3,SIZE,SIZE),), '../_dev/v2_legacy.onnx',
                     input_names=['image'], output_names=['logits'], dynamo=False)"
   uv run python -I -X utf8 quantize_model.py ../_dev/v2_legacy.onnx
   uv run python -I -X utf8 measure_onnx.py ../_dev/v2_legacy_int8.onnx
   ```
3. It is also 7.4 ms per classification instead of 1.0 ms, and its `check_image.jpg`
   probabilities are off by **0.68**, so it would fail the page's own built-in check
   (limit 0.001) before it ever reached a leaf.

`quantize_model.py` is kept so the experiment can be repeated; `measure_onnx.py` is what
shows it is chance. (Quantization-aware training is the way to get 8-bit weights that still
work, and it is far outside "keep it simple".)

**Test-time flip averaging** (classify the picture and its mirror, average the
probabilities). Free accuracy is free, usually — here it is not. The `cam + flip` column:
`full` 0.911 -> 0.900, `combined` 0.886 -> 0.906, `augment` 0.856 -> 0.842, `baseline`
unchanged. Two variants up, two down, all inside +-0.02, and it doubles the work per frame
on a phone that is already juggling a camera stream. A toggle for it was built into the
page, measured, and taken out again.

**WebGPU instead of the wasm runtime.** The bottleneck is not the model: on a real phone the
page classifies in about 71 ms per frame (`README.md`), and on desktop the exported model
needs 1.0 ms with one CPU thread. The ONNX Runtime Web GPU build needs a different, larger
binary (about 22 MB against 11 MB of wasm) and a context with cross-origin isolation turned
on, which GitHub Pages does not give. A bigger download for a speed-up nobody needs.

**More threads inside wasm** (`ort.env.wasm.numThreads = 2+`). Not an improvement to try but
one to avoid: without COOP/COEP headers (absent on GitHub Pages) and in iOS Safari, multi
threaded wasm either fails or crashes, which is why the Task 2 page pins one thread. Left
exactly as it was.

**A different ONNX Runtime Web version.** The version stays pinned at 1.19.2. Anything
older than 1.18 cannot read the `ir_version 10` graph that torch 2.15 exports; newer builds
fetch a different (larger, "jsep") wasm binary. 1.19.2 is what was verified to load both the
old and the new file, in the browser, with the same probabilities as PyTorch.

## Deliberately not tried

- **A bigger input than 96x96.** `model_info.json` carries `input_size`, so the page would
  follow automatically — but the whole Task 2 point is that the page matches the notebook's
  model exactly, and the exported check numbers are defined at 96. It is also quadratic cost
  on the phone for a model whose error is now dominated by leaf appearance, not resolution.
- **A bigger network** (`mobilenet_v3_large`, ResNet): doubles the download on a phone
  connection, for a problem this side of overfitting on 700 photos.
- **Mixup / CutMix, stochastic depth, two-stage fine-tuning, SAM.** Each is a paragraph to
  explain in a report whose rule is "you must be able to read every change and explain it".
  The four changes that stayed are all one line each.
- **A rewrite of the page** into modules, a framework, classes or helper layers: excluded in
  advance by the notebook's own red line. The new page is the Task 2 page plus a different
  model file, a different URL for the QR, an "I am not sure" line, and text saying what
  changed.
- **Retraining on the union of both datasets**, and **fine-tuning the shipped model further**
  on the historical photos — neither is possible without the group's ZIP in this copy.

## What went wrong while measuring (not model ideas, but part of the record)

- `--seed K` **did nothing** for several hours: `main()` looped over `range(args.seeds)` and
  never read `args.seed`, so `--seed 1` silently trained seed 0. Three parallel "seeds" of
  `augment_cosine` turned out to be three copies of seed 0, all writing the same
  `<variant>-seed0.json`. Caught because three supposedly different runs printed identical
  epoch lines; fixed in `main()`, the wasted runs deleted, the row retrained properly. Any
  row of the table that came from `--seeds N` (all of them) was never affected.
- Run files were first named without the seed, so consecutive seeds overwrote one another.
  Now `<variant>-seed<k>.json`, and `--summarise` reads only those files.
- `python -I` **ignores `PYTHONUTF8`** (`-I` implies `-E`), so `torch.onnx.export`'s progress
  line, which contains a check-mark emoji, died with `UnicodeEncodeError` on this cp1250
  console. The runs in this folder use `python -I -X utf8`, which sets it in spite of `-I`.
- One ad-hoc command was run without `TORCH_HOME=../_dev/torch_cache`, which made torchvision
  re-download the pretrained weights into the user cache under the home directory — outside
  this project, which the working rules forbid. Reported to the user rather than quietly undone.
