# Nexus: Movie Recommendation System

**Nexus** is a Streamlit-based application for recommending movies using content features and NLP preprocessing.

---

## Highlights

* Streamlit app entry point: `Nexus.py`
* Preprocessing and analysis via Jupyter notebooks: `preprocessing_1.ipynb`, `preprocessing_2.ipynb`, `preprocessing_3.ipynb`, `sentiment_analysis.ipynb`, `cast_sentiment_analysis.ipynb`
* Model and transform artifacts: `nlp_model.pkl`, `tranform.pkl`
* Datasets and caches under `datasets/`, including `.tmdb_genres_cache.json`
* Static assets in `static/`
* Dependencies listed in `requirements.txt`

---

## Repository Structure

```
Nexus/
├── Nexus.py                        # Main Streamlit application
├── datasets/                       # Prepared datasets and auxiliary files
│   └── .tmdb_genres_cache.json     # Cached genre data
├── static/                         # Static UI assets
├── preprocessing_1.ipynb           # Data cleaning step 1
├── preprocessing_2.ipynb           # Feature extraction step 2
├── preprocessing_3.ipynb           # Final preprocessing step 3
├── sentiment_analysis.ipynb        # Sentiment analysis on movie overviews
├── cast_sentiment_analysis.ipynb   # Sentiment analysis on cast bios
├── nlp_model.pkl                   # Trained NLP recommendation model
├── tranform.pkl                    # Data transformation pipeline
├── requirements.txt                # Python dependencies
```

---

## Setup (Local)

### Prerequisites

* Python 3.11 or higher
* pip installed

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Nexus
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:

   ```bash
   streamlit run Nexus.py
   ```

Once started, open the app in your browser at:

```
http://localhost:8501
```

---

## Usage

* Start the app locally and interact through the Streamlit interface.
* The app loads preprocessed data and model artifacts from paths referenced in `Nexus.py`.
* Explore recommendations, filter by genre, and view sentiment-based insights.

---

## Data & Artifacts

* **Datasets:** Located in `datasets/`
* **Model & Transform Files:**

  * `nlp_model.pkl` – Trained NLP recommendation model
  * `tranform.pkl` – Transformation pipeline

---

## Notebooks Overview

| Notebook                        | Description                            |
| ------------------------------- | -------------------------------------- |
| `preprocessing_1.ipynb`         | Initial data cleaning and formatting   |
| `preprocessing_2.ipynb`         | Feature extraction and selection       |
| `preprocessing_3.ipynb`         | Final data preparation                 |
| `sentiment_analysis.ipynb`      | Sentiment scoring on movie overviews   |
| `cast_sentiment_analysis.ipynb` | Sentiment analysis on cast biographies |

These notebooks document the preprocessing, exploration, and feature engineering processes used in model development.

---

## Deployment

* The application is live at: [https://nexus-mov.streamlit.app/](https://nexus-mov.streamlit.app/)
* For local testing and modification:

  ```bash
  streamlit run Nexus.py
  ```
