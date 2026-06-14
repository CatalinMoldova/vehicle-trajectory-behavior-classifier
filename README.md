
# Abu Dhabi Traffic Flow Vehicle Behavior Classification
**Deep Learning for Traffic State Analysis and Flow Prediction**

## Overview

This repository contains research code and data pipelines for classifying vehicle behaviors in simulated Abu Dhabi traffic using deep learning. Our methodology is based on actual vehicle trajectories and behavioral labels derived from detailed Aimsun micro-simulation output, supporting advanced research into traffic dynamics and the automation potential in Abu Dhabi’s arterial networks.

Two principal model training strategies are provided:
1. **Continuous Learning Classifier**: A scalable classifier that supports incremental updates, allowing the integration of new trajectory samples as they become available. This enables ongoing refinement and adaptation as traffic patterns and datasets evolve.
2. **Rate-Limited Fast Trainer**: An optimized trainer designed specifically for large-scale data ingestion from NYU’s Box storage, ensuring compliance with API rate limits (to prevent HTTP 429 errors) while maximizing throughput, robustness, and speed.

Both models utilize a sophisticated **CNN-LSTM hybrid architecture** that ingests time-series vehicle speed and position data, engineered to capture temporal and spatial patterns symptomatic of driver behavior categories (aggressive, cooperative, and normal).

## Reproducible Setup (Virtual Environment)

Use a project-local virtual environment so dependencies are isolated and reproducible. The `.venv/` directory is gitignored and is **not** committed.

**Option A — automated setup (recommended):**

```bash
git clone https://github.com/CatalinMoldova/Vehicle-Classifier.git
cd Vehicle-Classifier
bash scripts/setup_env.sh
source .venv/bin/activate
```

**Option B — manual setup:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
python scripts/generate_sample_data.py
```

**Run the pipeline:**

```bash
python -m vehicle_behavior.train --config configs/default.yaml --model logistic_regression
python -m vehicle_behavior.evaluate --model artifacts/logistic_regression.joblib
python -m vehicle_behavior.predict --input data/sample/sample_trajectories.csv --model artifacts/logistic_regression.joblib
pytest
```

In Cursor/VS Code, select the interpreter at `.venv/bin/python` when prompted (or click **Create** on the virtual-environment notification).

## Data Access Requirements

**Box API Authentication**  
To access the vehicle trajectory data hosted on NYU Box, you must supply a `key.json` JWT credential file, as generated via the Box developer console for your app.  
**Instructions:**
1. Place your `key.json` credential in the root project directory (where the notebook runs).
2. Inspect the authentication block and adjust the path if needed.

**Example:**
```python
auth = JWTAuth.from_settings_file('key.json')
client = Client(auth)
```

**Do not share your key.json file**.

## Google Drive Integration

All models and supporting files are persisted for experiment reproducibility and efficient collaboration.

- **Create a Google Drive shortcut** for a directory named `Test_Output` (case-sensitive), e.g.:
  ```
  /content/drive/MyDrive/Test_Output/
  ```
- This directory should be used to **store experiment artifacts**, including:
  - **Label encoder object:** `label_encoder.joblib`
  - **Processed file hash log:** `processed.json` (ensuring no duplicate data ingestion)
  - **Trained model artifact:** `traffic_model.keras` (Tensorflow/Keras SavedModel format)

Example snippet to mount Drive and set output location:
```python
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = '/content/drive/MyDrive/Test_Output'
```

## Model Architecture Description

The main model is a **hybrid Convolutional Neural Network–Long Short-Term Memory (CNN-LSTM) model**, optimally designed for sequential traffic data:

- **Inputs:**
  - Vehicle **speed** time series (shape: N x sequence_length x 1)
  - Vehicle **position** time series (shape: N x sequence_length x 2)
- **Feature Engineering:**
  - Internal computation of acceleration (first differences in speed) and lateral movement (frame-to-frame displacement)
- **Network Structure:**
  - **CNN branch:** Multiple `Conv1D` layers extract local temporal patterns, followed by batch normalization and pooling.
  - **LSTM branch:** Recursively processes sequence data, capturing driver behavior trends.
  - **Feature fusion:** Concatenation across temporal and spatial representations.
  - **Dense layers:** ReLU activations for high-level decoding, with dropout and batch normalization to prevent overfitting.
  - **Output:** Softmax classification across behavior labels (`aggressive`, `cooperative`, `normal`).

All model parameters, including input shapes, sequence lengths, and optimizer settings, are specified in the notebook and are customizable.

## Repository Structure

- `Cleaned_classification_Pipeline_24_Jul-2.ipynb`: The main notebook containing:
  - Data streaming and cleaning routines
  - Model building (see "build_speed_position_model_fixed")
  - Training loops for both incremental learning (CL-system) and rate-limited batch learning (Fast Trainer)
  - Evaluation metrics (accuracy, F1-score, detailed reports)
- Supporting scripts and functions for:
  - Feature processing
  - Data stratification and validation
  - NYU Box and Drive integration

## Operational Notes

- **NYU Box Rate Limiting:**  
  The Fast Trainer aggressively manages Box API calls (defaulting to 6 requests/sec with exponential backoff) and provides auto-resume and checkpointing to protect against accidental interruptions or throttling.
- **Incremental Learning:**  
  The Continuous Learning Classifier can be run on rolling datasets and supports online (lifelong) learning for traffic state modeling.
- **Experiment Logging:**  
  All training statistics, class distributions, and batch progress are logged for review. Model performance should be reported in terms of F1-macro, F1-weighted, and class-wise confusion matrices.

## Usage Guidelines

### Setup
1. Copy your `key.json` file into the working directory.
2. Create a shortcut or actual directory `Test_Output` in your `Google Drive > MyDrive`.
3. Open and execute the notebook, following all prompts in sequence.
4. All artifacts will be synchronized to `/Test_Output`.

### Training
- **Continuous (CL) training:**  
  Incrementally adds new records, retraining or fine-tuning the model after new data is labeled.
- **Rate-limited batch training:**  
  Processes files in bulk, stratifies datasets, builds fresh models, and resumes automatically from failed states.

### Model Extension
- For new datasets or additional vehicle behavior labels, adjust the `extract_vehicle_labels` mapping function.
- To extend sequence features (e.g., add heading, road context), expand the feature engineering modules.

## Abu Dhabi Research Context

This platform supports high-resolution traffic modeling and behavior recognition, enabling machine learning-driven inference and forecast of flow regimes in complex urban environments typical to Abu Dhabi.  
Future work will integrate real-world sensor datasets from Abu Dhabi's ITS network.

## Contact

For research collaboration or technical issues, please contact  
**Catalin Botezat**  
Abu Dhabi Traffic Flow Modeling Research  
cb5330@nyu.edu
