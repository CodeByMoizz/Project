# AI Tool Usage Declaration

SonicSentinel AI — NextWave AI and ML, Aptech TechWiz 7
Team: MSG-SlytherinCoders

No AI tool is used at runtime. The final sound classification is produced by
the Python classification model, the browser-side model and the alert rule
engine. No generative-AI API is called during prediction.

---

## 1. Unit tests

- **Tool:** ChatGPT
- **Purpose:** writing the automated test suite
- **Assistance requested:** test cases for audio file validation, feature
  extraction, preprocessing, audio quality assessment and the alert rule
  engine, including boundary and negative cases such as silent, corrupt,
  empty and unsupported files
- **Files affected:** `tests/`
- **Team modifications:** replaced generated fixtures with real clips from the
  project dataset; added cases for duplicate uploads and clipped audio; removed
  tests that asserted on implementation details rather than behaviour
- **Testing completed:** full suite run locally against the application; all
  tests passing before each change was accepted
- **Verified by:** Zarwasham, Moiz

## 2. Diagrams

- **Tool:** ChatGPT
- **Purpose:** producing the diagrams required by the project report
- **Assistance requested:** data flow, use case, activity, sequence and
  decision flow diagrams describing the upload, prediction, comparison and
  alerting paths
- **Files affected:** `documentation/`, project report
- **Team modifications:** corrected the sequence diagram so the browser-side
  model is shown predicting independently in the client rather than on the
  server; updated the decision flow to match the implemented rule engine,
  including the manual-review branch
- **Testing completed:** each diagram walked against the running application
  and the corresponding source modules
- **Verified by:** Zarwasham

## 3. User interface bug fixes

- **Tool:** ChatGPT
- **Purpose:** diagnosing and fixing front-end defects
- **Assistance requested:** fixes for layout problems on smaller screens,
  result fields not refreshing after an analysis completed, and inconsistent
  display of confidence values
- **Files affected:** `templates/`, `static/css/`, `static/js/`
- **Team modifications:** rewrote the suggested fixes to match the existing
  page structure instead of introducing a new layout system; kept the styling
  consistent across pages
- **Testing completed:** every page checked in the browser at desktop and
  mobile widths after each change
- **Verified by:** Alishba

## 4. Environment and dependency problems

- **Tool:** ChatGPT
- **Purpose:** resolving environment setup and dependency conflicts
- **Assistance requested:** diagnosing a failing llvmlite build during conda
  environment creation, a missing soundfile package, XGBoost failing to load
  without libomp on macOS, and NumPy version conflicts introduced by
  TensorFlow
- **Files affected:** `requirements.txt`, conda environment configuration
- **Team modifications:** pinned the NumPy version so the training and serving
  environments match; documented the libomp requirement in the README rather
  than adding it as a Python dependency
- **Testing completed:** environment rebuilt from scratch and the application
  run end to end; full test suite passing afterwards
- **Verified by:** Alishba, Moiz, Zarwasham

## 5. Documentation and blog

- **Tool:** Gemini
- **Purpose:** improving the wording and structure of the project report and
  the technical blog
- **Assistance requested:** language and readability improvements to text
  written by the team, and suggestions on section ordering
- **Files affected:** project report, technical blog
- **Team modifications:** rewrote sections where the suggested phrasing
  overstated results; kept the team's own account of the difficulties
  encountered
- **Testing completed:** every figure quoted in the report and blog checked
  against the metrics files in `reports/` and the confusion matrices they were
  computed from
- **Verified by:** Moiz, Zarwasham, Sarim

## 6. Machine learning work

- **Tool:** ChatGPT
- **Purpose:** assistance with model selection, feature extraction and
  evaluation methodology
- **Assistance requested:** which acoustic features suit environmental sound
  classification; how to compare candidate classifiers fairly; how to evaluate
  per-class and critical-class performance
- **Files affected:** `src/`, `feature_extraction/`, `notebooks/`
- **Team modifications:** chose the final feature set and model after running
  the comparison ourselves; selected the final model on validation macro F1
  rather than accuracy, because accuracy hid weak performance on the smaller
  critical classes
- **Testing completed:** all candidate models evaluated on the held-out testing
  split, which was never used for training or model selection; every reported
  metric recomputed directly from its confusion matrix and confirmed to match
- **Verified by:** Moiz, Sarim


---

No AI-generated code was submitted without review, modification and testing by
the team. The final classification, confidence scores and alert decisions are
produced entirely by the project's own models and rule engine.
