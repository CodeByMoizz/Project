# State

## Browser model

The second, independent, browser-side classifier is our own CNN on log-mel
spectrograms. Teachable Machine is gone.

`gtm_model/` holds three files and nothing else:

| file | what |
|---|---|
| `model.json` | tfjs LayersModel topology |
| `group1-shard1of1.bin` | weights, one shard |
| `metadata.json` | labels in encoder order, feature params, version |

Version string: `gtm_model_final`.

### Feature parameters

Same values in Python and JavaScript.

| | |
|---|---|
| sample_rate | 16000 |
| window | 1.0 s = 16000 samples |
| pad_to | 16384 samples |
| frame_length | 1024 |
| frame_step | 512 |
| fft_length | 1024 |
| mel_bins | 64 |
| fmin / fmax | 20 / 8000 |
| log | log(mel + 1e-6) |
| normalise | per example: subtract mean, divide by std |
| input shape | [31, 64, 1] |

16000 samples through a 1024/512 STFT give 30 frames. The window is zero padded
to 16384 on both sides, which gives exactly 31.

### Aggregation

Mean over the 1-second windows of a clip or segment, then argmax. Chosen on the
validation split only:

| policy | val accuracy | val macro F1 |
|---|---|---|
| max over windows | 0.8039 | 0.7978 |
| mean over windows | 0.8214 | 0.8153 |
| mean of top-3 windows | 0.8148 | 0.8084 |

Applied in `gtm.js` for both the live and the uploaded path.

### Files

| file | what |
|---|---|
| `feature_extraction/logmel.py` | features, Python |
| `static/js/logmel.js` | features, JavaScript |
| `src/browser_cnn.py` | dataset, model, training |
| `notebooks/BrowserCNN.ipynb` | Colab training |
| `static/js/gtm.js` | loads the model, runs inference, aggregates |
| `tests/test_gtm_label_check.py` | label check tests |

### Results

Feature parity, 5 clips, Python vs JavaScript:

| backend | worst mean abs diff | worst correlation |
|---|---|---|
| cpu | 3.7e-5 | 1.000000 |
| webgl | 1.7e-3 | 0.999976 |

Model parity after conversion, 5 test clips, worst max abs difference
2.287e-4. Gate was 1e-2.

Test split, 445 clips, Python, mean over windows:

| metric | value |
|---|---|
| accuracy | 75.73% |
| macro F1 | 0.7555 |

Same 445 clips through the app's own browser path:

| metric | Python | browser |
|---|---|---|
| accuracy | 75.73% | 75.51% |
| macro F1 | 0.7555 | 0.7533 |

One clip differs out of 445.

Thresholds: accuracy FAIL, macro F1 FAIL, glass_breaking and aggression PASS,
gunshot / panic_scream / person_asking_help FAIL.

Weakest class is `background_noise` at 0.4222 recall, and `machinery_fault`
takes 70 predictions for 45 true clips.

### Notes

- Keras 3 writes `batch_shape` and a `DTypePolicy` dict into the tfjs export.
  tfjs-layers reads neither, so `model.json` is rewritten after conversion to
  `batch_input_shape` and `dtype: "float32"`. Weights untouched.
- The discarded models are in `~/Desktop/sonicsentinel_archive/`, outside the
  project: the Keras source, the SavedModel zip, `gtm.zip`, and all three
  Teachable Machine exports.
- `gtm.zip` held a 108 MB scikit-learn RandomForest and a third Teachable
  Machine export. The RandomForest cannot run in a browser, so it cannot fill
  this role. That TM export also has two labels that fail the mapping,
  `Mechanical Fault` and `Panic screaming`.
- TensorFlow 2.16.2 and tensorflowjs 4.22.0 are installed in the `sonic` conda
  env. TensorFlow pulled numpy down to 1.26.4.
