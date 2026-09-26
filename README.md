# SonicSentinel AI

A Flask web application that detects sound events in uploaded audio clips and in live
microphone input. Every audio segment is classified twice, independently: once by a Python
machine-learning model on the server, and once by a Google Teachable Machine audio model
running in the browser. The two results are compared, checked against configurable alert
rules, and stored as a sound event with a severity, an alert status and a manual-review
decision.

Supported sound categories: Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn,
Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help, Background Noise.

This is a competition prototype for controlled and ethical testing. It is not a certified
emergency-response or law-enforcement system.

## Prerequisites

- Python 3.11 or newer (developed on 3.13)
- FFmpeg, needed by librosa to read MP3, M4A and OGG files
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: install from ffmpeg.org and add it to PATH
- On macOS, XGBoost needs OpenMP locally: `brew install libomp`. Without it `import
  xgboost` fails with "libxgboost.dylib could not be loaded". This only affects this
  machine — Colab, where the models are trained, does not need it.
- A modern browser with microphone support for live monitoring

## Installation

```bash
git clone <repository-url>
cd Project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create the database tables:

```bash
python database/database.py
```

Optionally set a real secret key and, if you host the GTM model instead of copying it in:

```bash
export SONICSENTINEL_SECRET_KEY="choose-a-long-random-value"
export GTM_MODEL_URL="https://teachablemachine.withgoogle.com/models/XXXXXXX/"
```

## Running the application

```bash
python app.py
```

Open http://127.0.0.1:5000 and register an account. **The first account created
automatically becomes the administrator.** Later accounts get the role chosen on the
registration form.

The application starts and runs with no trained models present. Every page works, uploads
are validated, preprocessed, segmented and stored, and the interface says clearly that the
model is not loaded instead of showing an error page.

## Running the tests

```bash
python -m pytest tests -q
```

## How to use it

| Page | What it does |
|---|---|
| Dashboard | Upload one file, or several for a batch. Shows metadata, waveform, spectrogram, both models' top-three scores, agreement status, confidence difference, top-two margin, audio quality, severity, alert status and the recommended action. |
| Live Monitor | Starts the microphone after you allow it, captures fixed windows, sends each to the server for the Python model, and runs the GTM model in the browser on the same stream. Shows microphone status, both results and the live comparison. |
| Event History | Search and filter stored events by audio ID, filename, category, date range, confidence range, severity, quality, review status and user. |
| Alerts | Acknowledge, dismiss or escalate active alerts, and see the full alert history. |
| Manual Review | Queue of uncertain and conflicting events. Reviewers can play the audio, confirm or correct the class, add comments and override the automatic result. The original model outputs are always kept. |
| Reports | Detection analytics, and the saved training metrics for each model: accuracy, precision, recall, F1, macro F1, per-class results, critical-class recall, confusion matrix and noise robustness. |
| Admin | Confidence and margin thresholds, the repeated-detection window, retention, the loaded alert rules, monitoring notices and the audit trail. |

Roles: `user` uploads and views; `reviewer` also records review decisions; `operator` and
`maintenance` handle alerts; `admin` can do everything, change thresholds and export data.

## Preparing the dataset

Collect the recordings yourself — at least 3,000 original clips, around 300 per class — and
put them in one folder per class:

```
audio_dataset/raw/machinery_fault/
audio_dataset/raw/glass_breaking/
audio_dataset/raw/alarm_siren/
audio_dataset/raw/vehicle_horn/
audio_dataset/raw/animal_sound/
audio_dataset/raw/gunshot/
audio_dataset/raw/panic_scream/
audio_dataset/raw/aggression/
audio_dataset/raw/person_asking_help/
audio_dataset/raw/background_noise/
```

Then build the splits:

```bash
python data_preparation/prepare_dataset.py
```

This validates every file, splits the data 70/15/15 with stratification, writes
`data/metadata/dataset_metadata.csv` and `data/metadata/dataset_statistics.csv`, and warns
about class imbalance and short class counts.

Two rules are enforced automatically, because breaking either one inflates the graded
scores:

- every segment of one recording stays in the same split
- for classes containing speech, every clip from one speaker stays in the same split

Grouping is taken from the filenames: a recording is the filename with any `_seg0` or
`__aug_1` suffix removed, and a speaker is a `spk3_` or `speaker3_` prefix. To describe the
recordings explicitly instead, create `data/metadata/source_metadata.csv` with the columns
`filename, speaker_id, recording_id, source, recording_environment, recording_device,
source_distance`.

Then augment the training split only:

```bash
python augmentation/augmentation.py
```

This writes `audio_dataset/augmented_training/` plus
`data/metadata/augmentation_metadata.csv`, applying eight augmentation types: noise, time
shift, pitch shift, time stretch, volume, reverberation, distance simulation and recording
device simulation. Augmented clips stay in the training split and are never counted as
original recordings.

## Training the Python models (Google Colab)

Training runs in Colab, not locally. The notebooks import the project's own
feature-extraction code from Drive, so the features used in training are exactly the ones
the app uses at inference time.

### 1. Upload to Drive

Upload the whole project folder to your Drive, so it looks like this:

```
MyDrive/SonicSentinel/
  app.py
  config/
  feature_extraction/
  audio_preprocessing/
  src/
  audio_dataset/
    training/<class>/*.wav
    augmented_training/<class>/*.wav
    validation/<class>/*.wav
    testing/<class>/*.wav
  data/metadata/
  python_models/
  reports/
```

The code folders are required — the notebooks import from them. Run
`prepare_dataset.py` and `augmentation.py` locally first, then upload
`audio_dataset/` with the splits already built.

You do not need to upload `.venv/`, `database/`, `templates/` or `static/`.

The prepared splits are about 850 MB (training 190 MB, augmented training 578 MB, validation
42 MB, testing 40 MB), so allow time for the upload. You do not need to upload
`audio_dataset/raw/` — the notebooks only read the four split folders.

### 2. Run the notebooks

Open each one from Drive in Colab and run the cells top to bottom:

| Order | Notebook | Notes |
|---|---|---|
| 1 | `notebooks/LogisticRegression.ipynb` | Fastest. Run it first to confirm the data loads. |
| 2 | `notebooks/Random_Forest.ipynb` | CPU. |
| 3 | `notebooks/XGBoost.ipynb` | Set Runtime > Change runtime type > T4 GPU first. |
| 4 | `notebooks/SVM.ipynb` | Slowest, because it needs probability estimates. |

Any order works; they are independent. Run all four to compare them.

In every notebook, change one line only:

```python
PROJECT_PATH = '/content/drive/MyDrive/SonicSentinel'
```

Every other path is derived from it. If Colab shows a "Restart runtime" button after the
install cell, click it and carry on from the Drive mount cell — do not run the install
again.

Each notebook prints the split sizes and per-class counts before the long step, so check
those numbers look right before leaving it running. Extracted features are cached to
`data/metadata/*_features.csv` on Drive; if the runtime disconnects, run the training cell
again and it reuses the cache instead of re-extracting everything.

Method, the same in all four: trained on the training split only, hyperparameters selected
on the **validation** split, and the test split scored once at the end and never tuned
against. Where a full grid would take hours, 40 combinations are sampled at random instead
and the notebook says so.

### 3. Download the results

Each notebook writes four files into your Drive. The last cell lists them.

| Notebook | Writes to `python_models/` | Writes to `reports/` |
|---|---|---|
| LogisticRegression | `logistic_regression_model.pkl`, `logistic_regression_scaler.pkl`, `label_encoder.pkl` | `logistic_regression_metrics.json` |
| Random_Forest | `random_forest_model.pkl`, `random_forest_scaler.pkl`, `label_encoder.pkl` | `random_forest_metrics.json` |
| XGBoost | `xgboost_model.pkl`, `xgboost_scaler.pkl`, `label_encoder.pkl` | `xgboost_metrics.json` |
| SVM | `svm_model.pkl`, `svm_scaler.pkl`, `label_encoder.pkl` | `svm_metrics.json` |

Download them and put them in the matching folders in the local project:

- the `.pkl` files go in `python_models/`
- the `_metrics.json` files go in `reports/`

`label_encoder.pkl` is shared, so the last notebook you run simply overwrites it with an
identical file. Keep one copy.

**Prepare the dataset once, then run all four notebooks against that same prepared data.**
`label_encoder.pkl` is shared and last-writer-wins. If you re-prepare the dataset between
notebook runs, a later notebook can write an encoder whose class ordering differs from the
one an earlier model was trained against. Nothing would fail — the app would load the model
and the newer encoder together, and every prediction would be silently mislabelled. If you
do re-prepare the dataset, re-run all four notebooks.

### 4. Pick the winner

Start the app and open the Reports page — it lists every model that has a metrics file,
with accuracy, macro F1, precision and recall. Then set one line in `config/config.py`:

```python
ACTIVE_PYTHON_MODEL = "random_forest"
```

Valid values: `random_forest`, `svm`, `xgboost`, `logistic_regression`. No other file names
a model. Restart the app afterwards.

`notebooks/eda.ipynb` explores the raw audio and is not part of training. It is written for
local use.

## The Google Teachable Machine model

Teachable Machine audio projects export to TensorFlow.js, so the GTM model runs in the
browser beside the microphone capture. Its result is posted back to the server, which
stores it and compares it with the Python prediction.

1. Create an Audio Project at https://teachablemachine.withgoogle.com/
2. Use the same ten class names as the Python model, plus Teachable Machine's own
   Background Noise class. Spelling like "Alarm or Siren" or "alarm_siren" is fine, the
   server maps the names back onto the canonical ids.
3. Train the samples only from the recordings in `audio_dataset/training/`.
4. Export the model as **TensorFlow.js**, then either:
   - download it and copy `model.json`, `metadata.json` and the `.bin` weight files into
     `gtm_model/`, or
   - upload it and set `GTM_MODEL_URL` to the hosted URL.

The two models never see each other's output: the Python model runs on the server from the
audio alone, and the GTM model runs in the browser from the microphone stream alone. The
Python prediction is never sent to the browser before the GTM result is posted.

## Alert rules

All rules live in `alert_rules/alert_rules.json` — no thresholds are written into the code.
Each category configures its minimum confidence, top-two margin, required consecutive
detections, model-agreement requirement, minimum audio quality, severity, recommended
action, manual-review condition and escalation condition. The file is re-read on every
detection, so editing it takes effect without a restart. Adding a new category is a change
to this file plus `CLASSES` in `config/config.py`.

## Project structure

```
app.py                       Flask application and all routes
config/config.py             active model, thresholds, class list, paths
alert_rules/alert_rules.json configurable alert rules
audio_preprocessing/         validation.py and preprocessing.py
feature_extraction/          features.py, the single shared feature path
augmentation/                augmentation.py, training split only
data_preparation/            prepare_dataset.py, splits and metadata
src/
  analysis.py                ties the whole prediction path together
  python_model.py            loads the active model, guarded
  comparison.py              compares the two models
  rules.py                   reads and applies alert_rules.json
  decision.py                the final sound event decision
  quality.py                 audio quality analysis
  visuals.py                 waveform and spectrogram images
  audio_metadata.py          metadata, hashing, near-duplicate fingerprint
  metrics.py                 reads the saved training metrics
  training.py                shared training pipeline for the notebooks
database/database.py         schema and all queries
templates/                   the nine pages
static/                      style.css and plain JavaScript
notebooks/                   EDA and the four training notebooks
python_models/               trained models go here
gtm_model/                   exported Teachable Machine model goes here
reports/                     saved metrics files
audio_dataset/               raw audio and the generated splits
sample_audio/                a few clips for a quick manual test
tests/                       pytest suite
documentation/               project report outline
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "The model is not trained yet" | Train a model and put the files in `python_models/` with the names above. |
| "No GTM model found" | Copy the TensorFlow.js export into `gtm_model/` or set `GTM_MODEL_URL`. |
| MP3 or M4A upload fails to decode | Install FFmpeg and restart. |
| `libxgboost.dylib could not be loaded` on macOS | `brew install libomp`. |
| Microphone does nothing | Browsers only allow microphone access on `localhost` or over HTTPS, and permission must be granted. |
| Reports page shows no metrics | The metrics files are missing from `reports/`; rerun the training notebook. |
| Database errors after changing the schema | Delete `database/sonicsentinel.db` and run `python database/database.py`. |
