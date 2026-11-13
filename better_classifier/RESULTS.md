# What each change measured

Numbers from `train_leaves.py --summarise` and `measure_onnx.py`, run on this machine on
2026-11-12. Every run is recorded in `../_dev/runs/<variant>-seed<k>.json`, so any number
here can be traced to the file it came from. This is the raw material for `REPORT.md`,
which the student writes — this file is not that report.

## How it was measured (the rules the notebook sets)

- **Fixed baseline**: the recipe of Task 2 exactly as the notebook writes it —
  `mobilenet_v3_small(weights="DEFAULT")` with `classifier[3]` replaced, the whole network
  fine-tuned, Adam `lr=1e-3`, batch 32, `CrossEntropyLoss`, 15 epochs, no augmentation.
- **One change per variant**, everything else inherited from the baseline
  (`VARIANTS` in `train_leaves.py` is literally "baseline plus these settings").
- **Same training pictures, fixed seed.** Every variant trains on the same 700 photos;
  `torch.manual_seed(seed)` and a seeded shuffling `Generator`. Important rows run with
  three seeds, and the table shows the spread.
- **Two test sets, and only one of them is honest.**
  - *validation* = the 120 validation photos, measured on the **last** epoch.
  - *camera-like* = the same 120 photos, each randomly cropped to 50–100 %, rotated by up
    to 25°, recoloured (brightness/contrast/saturation 0.4, hue 0.15) and saved —
    **generated once with seed 0** by `camera_like_set()`, never used to choose anything.
    This is requirement 2 of Task 4, and it is the column that decides.
- **Noise.** One validation photo is worth 1/120 = 0.008. The seed spread of a row is
  typically ±0.02–0.05, so a difference of about 0.02 or less between two rows is not a
  result, it is the same result twice. The `+-` in the table is half the distance between
  the best and the worst seed.
- The `s/epoch` column is **not** a measurement: the runs went in parallel streams on a
  shared laptop, so it mostly records how many neighbours each run had.

## The table

32 runs, `uv run python train_leaves.py --summarise`:

```
variant        runs      validation     camera-like  cam + flip  s/epoch
adamw             2   0.887 +-0.029   0.762 +-0.038       0.775      6.3
augment           3   0.908 +-0.008   0.856 +-0.038       0.842     26.8
augment_cosine    3   0.919 +-0.013   0.800 +-0.021       0.808      8.4
baseline          3   0.900 +-0.017   0.739 +-0.054       0.739     26.3
colour            3   0.894 +-0.017   0.850 +-0.000       0.850     18.7
combined          3   0.906 +-0.004   0.886 +-0.004       0.906     18.6
cosine            3   0.922 +-0.004   0.775 +-0.029       0.783     12.5
frozen            3   0.781 +-0.025   0.650 +-0.025       0.653     12.9
full              3   0.892 +-0.021   0.911 +-0.013       0.900     13.3
scratch           1     0.883           0.600             0.608      5.3
smooth            2   0.917 +-0.008   0.863 +-0.046       0.850      7.0
```

| variant | what it is, against the baseline |
| --- | --- |
| `baseline` | the Task 2 recipe, as the phone runs it today |
| `augment` | + geometric augmentation: `RandomResizedCrop(96, scale=0.6-1.0)`, horizontal and vertical flip, `RandomRotation(20)` |
| `colour` | `augment` + `ColorJitter(0.3, 0.3, 0.3, 0.1)` |
| `cosine` | + `CosineAnnealingLR` and 30 epochs instead of 15 |
| `smooth` | + `CrossEntropyLoss(label_smoothing=0.05)` |
| `adamw` | AdamW with weight decay 1e-4 |
| `frozen` | only `classifier[3]` trained, everything else frozen |
| `scratch` | the Task 3 network (`LeafNet`) from random weights, 30 epochs, augmented |
| `augment_cosine` | `augment` and `cosine` together, without colour jitter |
| `combined` | augmentation + colour jitter + cosine/30 epochs |
| `full` | `combined` + label smoothing 0.05 — **the model this folder ships** |

## The two columns disagree, and that is the finding

`cosine` has the **best validation accuracy of everything tried** (0.922) and is only
fourth on the camera-like set (0.775). `scratch` looks respectable on validation (0.883)
and collapses to 0.600 when the same photos arrive cropped, rotated and recoloured. If
only the validation column had been measured, the choice would have been `cosine` — and
the phone would have got a model 0.14 worse than the one shipped.

The single changes also do not add up. Written as a 2x2 of augmentation against the
longer cosine schedule (camera-like, last epoch, three seeds each):

|  | 15 epochs, plain lr | 30 epochs, cosine |
| --- | --- | --- |
| **no augmentation** | 0.739 `baseline` | 0.775 `cosine` |
| **augmentation** | 0.856 `augment` | 0.800 `augment_cosine` |

On its own, the cosine schedule *hurts* an augmented run by 0.056. Colour jitter on top of
augmentation alone is worth nothing (0.850 vs 0.856). Yet the full stack — augmentation +
colour + cosine + label smoothing — reaches **0.911**, the best number in the table and
0.055 above augmentation alone. Regularisation only pays once the network is training long
enough and hard enough to overfit; measured one at a time, on 15 epochs, each of those
changes looks pointless. That is the honest reason the combination is shipped and not the
prettiest single row.

### Compared seed by seed, not just by mean

The same run with the same seed shares its shuffling and its augmentation draws, so the
honest comparison is pair by pair (camera-like, last epoch):

| seed | `baseline` | `augment` | `augment_cosine` | `combined` | `full` |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.683 | 0.842 | 0.817 | 0.883 | **0.900** |
| 1 | 0.742 | 0.825 | 0.808 | 0.892 | **0.908** |
| 2 | 0.792 | 0.900 | 0.775 | 0.883 | **0.925** |

`full` is ahead of every variant it is stacked from, on all three seeds. That is a
stronger statement than its margin over `combined` in the means (0.911 vs 0.886 = 0.025,
about three photos): the size of the gap is still within what a fourth seed could move,
but its sign is not. `combined` would be a defensible second choice, and it is one word in
`--variant` away.

## What the phone does today, and what this page does

Both files measured by `measure_onnx.py`, one thread, the page's own preprocessing,
on an idle machine:

| | `leaf_model.onnx` (the page today) | `leaf_model_v2.onnx` (this folder) |
| --- | --- | --- |
| validation (120 photos) | 0.500 | 0.875 |
| **camera-like (120 photos)** | **0.450** | **0.900** |
| file size | 6.35 MB | 6.25 MB (export metadata stripped — see `README.md`) |
| one classification, 1 thread | 1.0 ms | 1.0 ms |
| agrees with PyTorch on `check_image.jpg` | 2.8e-10 | 1.8e-07 |

Same network, same input size, same download, same speed — the difference is only how it
was trained. `leaf_model_v2.onnx` is variant `full`, seed 0, **last** epoch; its seed's own
camera-like number is 0.900 and the three-seed mean of the variant is 0.911 +-0.013.

The old model's 0.450 is not a contradiction of the notebook's own 0.724: that 0.724 was
on the *group's* validation photos, this is on last year's photos, cropped and recoloured.

### And it knows when it does not know

The old model was confidently wrong, which is the worst thing a page like this can be. How
often it was right, grouped by the probability it printed, on the camera-like set:

| top probability | page today | this model |
| --- | --- | --- |
| under 0.40 | 3 photos, right 0.00 | 0 photos |
| 0.40–0.60 | 27 photos, right 0.30 | 9 photos, right 0.56 |
| 0.60–0.80 | 19 photos, right 0.47 | 12 photos, right 0.75 |
| 0.80–0.95 | 26 photos, right 0.46 | 38 photos, right 0.89 |
| 0.95 and above | 45 photos, **right 0.56** | 61 photos, **right 0.98** |

Today, a "99 % chestnut" frame is wrong almost half the time. Now the confident answers
are right 98 % of the time and only 9 of 120 photos are below 0.60 at all.

`UNSURE_BELOW = 0.6` in `index.html` comes from the first row of that right-hand column:
under 0.60 the model is right about as often as a coin, so the page says so
("Not sure — cherry is only 31 % …") instead of showing a bar chart as if it were a
decision. It speaks on 8 of 120 photos. The stricter 0.80 would also cover the 0.60–0.80
band, where the model is right 3 times out of 4 — a reasonable different choice, and one
line to change.

## Not verified, said out loud

- **Resolved: the QR address is live now.** It was not — on 2026-11-12
  `https://256995.github.io/mlr-ex08/better_classifier/` answered **404** while the Task 2 page
  above it answered **200**, which is the whole explanation of "the QR does not work". The
  folder was published on 2026-11-13 (commit `f9ea03a`, pushed to `github.com/256995/mlr-ex08`
  from this copy, staying a subfolder of the repo root), the address answers **200**, and the
  page was then tested *from that address* rather than from a local server — see `README.md`.
  An earlier version of this file, and the working rules one folder up in `CLAUDE.md`, said
  this copy has no git remote: it has had `origin` there all along, with `main` level with
  `origin/main`, which is why publishing was one ordinary push.
- **The QR code draws a correct code, and it used to point at a page that does not exist.**
  Both codes the page can draw were checked with `_dev/qr_check.py` on 2026-11-12, each
  painted as a PNG, decoded by zxing-cpp, and rebuilt by a reference encoder (`qrcode`)
  forced to the page's own version, level and mask:
  the served address gives **0 modules differing out of 841** (version 3, mask 7) and the
  published fallback address **0 out of 1089** (version 4, mask 0); zxing reads each as
  exactly its own text. The 15 format bits read back out of the grid also decode as error
  correction M with the mask the page claims.
  The "528 modules differ" line that stood in this file before was the *checker's* fault, not
  the page's: `qr_check.py` left the reference library free to pick its own mask, and a
  different mask flips about half the data modules. `qr_check.py` now forces
  `makeImpl(False, mask)` instead, and re-running it on the same grids reports 0 — that
  difference is the whole story of the bug that was never in `index.html`.
  What *was* wrong in the page is the address in it — see the next bullet.
- **The QR encoded a published URL that has never been published**, so every phone that
  scanned it got a 404, and the page loaded from anywhere else could not be reached by
  scanning at all. `PAGE_URL` is now the address the browser actually loaded
  (`location.href`, minus query and fragment), with the published URL only as the fallback
  for a page opened straight from disk. Verified in the browser from both ends of that rule:
  served from `http://127.0.0.1:8731/better_classifier/` the drawn code encodes exactly that
  (29 modules, mask 7), and drawn by the published page it encodes the GitHub Pages address
  (33 modules, mask 0) — which since 2026-11-13 is a page that exists.
- **The browser check ran on Chrome for Windows with a fake camera device**
  (`_dev/test_better_page.py`, PASS at 9.06e-8, live bars at 24 ms/frame, both photo uploads
  classified with the canvas capped at 1024, 0 POSTs, no console errors), from four
  addresses: the published GitHub Pages URL (camera granted, 17 ms/frame), `localhost` (camera
  allowed), a LAN IP (camera refused with the reason shown, uploads still fine) and `file://`
  (model cannot load, page says why, QR falls back). It has
  not been opened on the iPhone, and `?selftest`-style in-page checks cannot prove the
  Safari/wasm path — nor the ~16-million-pixel canvas limit the photo fix works around, which
  desktop Chrome does not enforce.
- **Both columns are last year's photographs.** Requirement 2 allows the camera-like set
  *or* last year's photos. Here they are the same photos: the training data in this copy
  *is* the historical archive, so the camera-like set is a fixed distortion of the 120
  validation photos and there is no second, untouched domain to fall back on. The group's
  own photos are not in this folder, so nothing here measures "how good is it on leaves my
  group photographed" — the one thing that comparison would add.
- **No `REPORT.md` is written here.** The notebook requires it to be the student's own
  writing.
