# Nexus – NLP-Powered Movie Recommendation System

Nexus is a web application that recommends movies using content features and natural language processing. It goes beyond simple collaborative filtering by using NLP to analyze movie overviews and cast biographies, combining these insights to provide accurate and personalized movie suggestions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nexus-mov.streamlit.app/)

> [!NOTE]
> **Knowledge Cutoff:** The dataset used to build this recommendation system contains movies released up to **2024**. Searches for movies released after this period may not be found in the database.

## What makes it interesting

Nexus leverages machine learning and text analysis to find deeper connections between films:

1. **NLP Preprocessing**: Notebooks document advanced text preprocessing and sentiment analysis on movie overviews and cast biographies to derive rich combined features.
2. **Content-Based Filtering**: Recommends movies based on combined content features using Cosine Similarity, effectively mitigating the cold-start problem.
3. **Interactive UI**: Built with Streamlit, providing a fast, reactive web interface connected to the TMDB API for live movie posters and details.
4. **Pre-trained Artifacts**: Uses serialized data transformation pipelines for fast, real-time recommendation inference.

## Features

- **Movie Recommendations**: Get tailored movie suggestions based on a selected title using content similarity.
- **Detailed Movie Information**: View TMDB-powered insights including movie posters, ratings, release dates, and genres.
- **Top Cast Profiles**: Explore the top cast members for any selected movie with live profile images.
- **Transparent Data Pipeline**: Fully documented Jupyter notebooks detailing the entire data cleaning, feature extraction, and sentiment scoring processes.

## Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core language |
| **Web Framework** | Streamlit | Clean, reactive user interface |
| **Machine Learning** | Scikit-learn, NLTK | NLP and recommendation modeling |
| **Data Processing** | Pandas, NumPy | Data cleaning and feature engineering |
| **Ops & Dev** | uv, pytest, GitHub Actions | Fast dependency management, testing, and CI |
| **Deployment** | Streamlit Community Cloud | Live application hosting |

## Directory Structure

```text
Nexus/
├── .github/
│   └── workflows/                  # GitHub Actions CI pipelines
├── static/                         # Static UI assets
├── Nexus.py                        # Main Streamlit application entry point
├── data/                           # Prepared datasets and auxiliary files
│   └── .tmdb_genres_cache.json     # Cached genre data
├── models/
│   ├── nlp_model.pkl               # Trained NLP recommendation model
│   └── tranform.pkl                # Data transformation pipeline
├── notebooks/                      # Data cleaning and feature engineering
│   ├── cast_sentiment_analysis.ipynb
│   ├── preprocessing_1.ipynb
│   ├── preprocessing_2.ipynb
│   ├── preprocessing_3.ipynb
│   └── sentiment_analysis.ipynb
├── tests/                          # Unit tests
├── pyproject.toml                  # uv project configuration
├── requirements.txt                # Legacy Python dependencies
├── uv.lock                         # Locked dependency versions
└── README.md                       # Project documentation
```

## Getting Started

### 1. Setup

Clone the repository to your local machine:

```bash
git clone <your-repo-url>
cd Nexus
```

### 2. Environment & Dependencies

This project uses `uv` for fast package management.

Install `uv` (if not already installed) and sync the project:

```bash
uv sync --all-extras
```

*(Alternatively, you can use `pip install -r requirements.txt` if you prefer standard virtual environments).*

### 3. Run

Start the Streamlit application using `uv`:

```bash
uv run streamlit run Nexus.py
```

Once started, open the app in your browser at:
`http://localhost:8501`

## Usage

* **Interactive App**: Start the app locally and interact through the clean Streamlit interface.
* **Instant Predictions**: The app loads preprocessed data and model artifacts automatically to provide fast recommendations.
* **Explore**: Enter a movie name to explore recommendations, view movie overviews, and see top cast profiles.
