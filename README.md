# Multimodal Deepfake Detection and Explainable Content Verification

## 1. Project Overview

This project aims to detect misleading or manipulated content using multimodal analysis of text and images.

It analyzes textual and visual information to identify potential inconsistencies and provides an explainable prediction to help users understand the result.

## 2. Key Features

- Text-only analysis
- Image-only analysis
- Combined text and image analysis
- Real, Fake, and Uncertain prediction categories
- Confidence score
- Text-image mismatch scoring
- Analysis history
- Individual analysis result retrieval
- PDF report generation
- Input validation and error handling

## 3. Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| API Testing | Pytest, FastAPI TestClient |
| Database | SQLite, SQLAlchemy |
| Data Validation | Pydantic |
| Image Processing | Pillow |
| Report Generation | PDF generation library |
| Version Control | Git and GitHub |

## 4. Project Structure

```text
multimodal-deepfake-detection/
│
├── app/
│   ├── models/
│   ├── routers/
│   ├── schemas/
│   ├── services/
│   ├── database.py
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── predictions/
│   └── sanity_set/
│
├── evaluation/
│   ├── metrics.py
│   ├── mismatch.py
│   └── export.py
│
├── tests/
│
├── requirements.txt
├── DATASET.md
└── README.md
```

## 5. Backend Setup

### Prerequisites

- Python 3.10 or above
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/nainsikk/multimodal-deepfake-detection.git
```

Navigate into the project directory:

```bash
cd multimodal-deepfake-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Backend

```bash
uvicorn app.main:app --reload
```

The API server runs at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Backend health check |
| POST | `/analyze` | Analyze text and/or image |
| GET | `/history` | Retrieve analysis history |
| GET | `/result/{analysis_id}` | Retrieve a specific result |
| GET | `/report/{analysis_id}` | Generate a PDF report |
| GET | `/sanity` | Run a basic sanity check |

## 7. Evaluation

The evaluation module supports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- PR-AUC

The mismatch scoring module measures differences between text and image fake probabilities.

Evaluation reports can be exported in JSON format.

## 8. Testing

Run the complete automated test suite:

```bash
python -m pytest -v
```

The backend currently has **46 passing automated tests**.

## 9. Current Development Status

- FastAPI backend implemented.
- SQLite database integration implemented.
- Text-only, image-only, and combined API modes implemented.
- Evaluation metrics implemented and unit tested.
- Mismatch scoring implemented and unit tested.
- Automated API tests implemented.
- Real trained model integration and actual dataset-based evaluation are pending.

## 10. Team Project

This is a collaborative academic project focused on multimodal deepfake detection and explainable content verification.