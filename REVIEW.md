Review checking the requirements:

1. It runs on my phone: Yes, it does. The picture is in the repository.
![Leaf classified live on the phone — apple 66.02%, chestnut 20.67%, cherry 9.11%, maple 4.20%, 71 ms per frame](https://256995.github.io/mlr-ex08/phone_live_classification.jpg)

2. The same numbers as the notebook: The numbers are the same to at least three decimals.
The real values are taken from the page internal check, as the app itself does not show that many decimals.
check_expected:
{
 "apple": 1.2304545293773117e-07,
 "cherry": 0.999981164932251,
 "chestnut": 1.8114320482709445e-05,
 "maple": 5.882645268684428e-07
}
check real:
{
 "apple": 1.230e-7,
 "cherry": 0.999981,
 "chestnut": 1.811e-5,
 "maple": 5.883e-7
}

3. The preprocessing matching my model: Yes, it is done in function ClassifySource in index.html starting at line 210.

4. Nothing leaves the pohone: 
All the pages the page reaches for are justified (needed to run) and nothing leaves the phone.
┌───────────────────────────────────────┬───────────────────────────────────────────────────────┐
│                  URL                  │                          why                          │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/mlr-ex08/index.html                 │ the page itself                                       │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/model_info.json (295 B)             │ classes, 96, mean, std                                │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/leaf_model.onnx (6,348,543 B)       │ the network, downloaded once and cached               │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/check_expected.json (499 B)         │ the numbers for the built-in check                    │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/check_image.jpg?cachebust=… (3.1    │ the check photo; the query string stops a cached copy │
│ KB)                                   │  making the check a no-op                             │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ cdn.jsdelivr.net/…/ort.min.js         │ ONNX Runtime Web — the inference engine               │
│                                       │ (index.html:146)                                      │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/ort-wasm-simd-threaded.mjs (10.9    │ its JS glue                                           │
│ KB)                                   │                                                       │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ …/ort-wasm-simd-threaded.wasm         │ the engine itself: 11,018,731 B raw, 2,461,966 B over │
│                                       │  the wire (Brotli), cached afterwards                 │
├───────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ https://256995.github.io/favicon.ico  │ Chrome asks for it on its own — the page never        │
│                                       │ requests it                                           │
└───────────────────────────────────────┴───────────────────────────────────────────────────────┘