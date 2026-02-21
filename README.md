# 🚨 Chicago Crime Analysis & Prediction

> A full end-to-end Big Data and Machine Learning project on ~1.92 million crime records from Chicago (2001–2004), built with PySpark, XGBoost, and Streamlit.

---

## 👥 Team Members

| Name |
|------|
| Reem Al-Zouhby |
| Sourour Hammoud |
| Mariam Marhaba |
| Majdeline Allawa |

> Supervised by **Dr. Rola Naja**  
> Lebanese University — Faculty of Engineering, M2 IOT & Smart Systems

---

## 📌 Project Overview

This project covers the complete data science pipeline:

- 🧹 **Data Cleaning** — handling missing values, duplicates, type casting using PySpark
- 📊 **Exploratory Data Analysis (EDA)** — spatial maps, temporal charts, heatmaps
- 🤖 **Machine Learning** — Random Forest (first attempt) → XGBoost (final model)
- 🌐 **Dashboard** — interactive Streamlit app with live crime prediction and Folium maps

---

## 📁 Repository Structure

```
Chicago_crime_spark_ML/
│
├── preprocessing.py          # Data cleaning and preprocessing pipeline (PySpark)
├── xgboost_model.py          # Feature engineering, model training, and evaluation
├── streamlit_app.py          # Interactive Streamlit dashboard
├── Chicago_Crimes_2001_to_2004.csv  # Raw dataset (download separately)
├── crime_xgboost_clean1.json # Saved trained XGBoost model
├── indexer_mappings.json     # Saved StringIndexer label mappings
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📂 File Details

### `preprocessing.py`

Everything related to loading and cleaning the raw CSV:

- Loads the dataset with PySpark (`inferSchema=False`)
- Replaces `"NULL"` strings and empty cells with proper nulls
- Removes duplicate rows and stray header rows
- **Missing value strategy:**
  - `Location Description` (16 missing) → drop rows
  - `District` (2 missing) → drop rows
  - `Community Area` & `Ward` (~36% missing) → **replace with median** using `approxQuantile`
  - Coordinates (~30,692 missing) → drop rows
- Casts all columns to correct types (`DoubleType`, `IntegerType`, `Boolean`)
- Parses the `Date` column and extracts: `Year`, `Month`, `Hour`, `DayOfWeek`
- Groups the 30+ primary crime types into **4 macro-categories**:

| Category | Crime Types Included |
|----------|----------------------|
| `PROPERTY_CRIME` | Theft, Burglary, Motor Vehicle Theft, Criminal Damage, Criminal Trespass |
| `VIOLENT_CRIME` | Battery, Assault, Robbery, Homicide, **Weapons Violation** |
| `DRUG_CRIME` | Narcotics, Other Narcotic Violation |
| `OTHER` | All remaining types |

> **Note:** Weapons Violation was merged into `VIOLENT_CRIME` (instead of being a separate class) because it had too few samples (~12K) and is logically related to violent offences.

- Generates all EDA visualisations (bar charts, heatmaps, trend lines, correlation matrix)
- Produces interactive Folium maps (crime type map, crime category map)

---

### `xgboost_model.py`

The full ML pipeline from feature engineering to model evaluation:

- **Data split:** time-based split — train on 2001–2003, test on 2004 (no leakage)
- **Feature engineering (18 features):**
  - Temporal: `Year`, `Month`, `Hour`, `DayOfWeek`
  - Cyclical hour encoding: `Hour_sin`, `Hour_cos`
  - Weekend flag: `IsWeekend`
  - Spatial: `Latitude`, `Longitude`, `District`, `Beat`, `Ward`, `Community Area`
  - Distance from Chicago city centre: `Distance_from_center`
  - Density features (train-only): `District_Crime_Count`, `Beat_Crime_Count`, `Community_Crime_Count`
  - Encoded string: `Location_Index` (via PySpark `StringIndexer`)
- **Class balancing:** sample weights computed per class, capped at 5.0 to prevent dominance
- **Model:** XGBoost with `multi:softprob`, `max_depth=7`, `learning_rate=0.07`, early stopping (200 rounds)
- **Evaluation:** Accuracy, weighted F1-score, per-class classification report
- Saves the trained model as `crime_xgboost_clean1.json`
- Saves label mappings as `indexer_mappings.json`

---

### `streamlit_app.py`

A 5-page interactive web dashboard:

| Page | Description |
|------|-------------|
| 📊 Overview | Key metrics, crime category definitions, model performance chart |
| 📈 Visualisations | All EDA charts viewable individually or as a gallery |
| 🔮 Crime Prediction | Input form → live XGBoost prediction with probability chart |
| 📍 Interactive Maps | Embedded Folium HTML maps (by type and by category) |
| 📉 Trends Analysis | Plotly charts of yearly and categorical crime trends |

---

## 🧪 Results

| Model | Classes | Accuracy | F1 (weighted) |
|-------|---------|----------|----------------|
| Random Forest (initial) | 20+ raw types | 8% | Very low |
| Random Forest (improved) | 20+ raw types | 34% | Low |
| **XGBoost (final)** | **4 categories** | **51.88%** | **0.531** |

### Per-class breakdown (XGBoost):

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| PROPERTY_CRIME | 0.74 | 0.54 | 0.62 | 176,227 |
| VIOLENT_CRIME | 0.51 | 0.43 | 0.47 | 112,621 |
| OTHER | 0.31 | 0.51 | 0.39 | 52,007 |
| DRUG_CRIME | 0.39 | 0.69 | 0.50 | 44,036 |

### Key Observations:

- **Location_Index** is the strongest predictor by a wide margin (gain 37.1 — 3× any other feature)
- **DRUG_CRIME** has high recall (69%) but low precision (39%) — the model over-predicts this class
- **PROPERTY_CRIME** has the best precision (74%) but misses ~46% of actual cases
- **VIOLENT_CRIME** is the hardest class to separate with both precision and recall below 55%

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/reemzouhby/Chicago_crime_spark_ML.git
cd Chicago_crime_spark_ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `Chicago_Crimes_2001_to_2004.csv` from [Kaggle](https://www.kaggle.com/currie32/crimes-in-chicago) and place it in the project root folder.

### 4. Run preprocessing and EDA
```bash
python preprocessing.py
```
This will clean the data and generate all charts and map files.

### 5. Train the XGBoost model
```bash
python xgboost_model.py
```
This will train the model and save it as `crime_xgboost_clean1.json` and `indexer_mappings.json`.

### 6. Launch the dashboard
```bash
streamlit run streamlit_app.py
```

---

## 📦 Requirements

```
pyspark
xgboost
scikit-learn
streamlit
folium
matplotlib
seaborn
plotly
pandas
numpy
Pillow
```

---

## 📊 Dataset

- **Source:** [Kaggle — Crimes in Chicago](https://www.kaggle.com/currie32/crimes-in-chicago)
- **Records:** ~1.92 million
- **Columns:** 22
- **Period:** 2001–2004
- **Train set:** 2001–2003 (~1,425,334 rows)
- **Test set:** 2004 (~384,891 rows)

---

## 🔗 Links

- 📁 **GitHub Repository:** [https://github.com/reemzouhby/Chicago_crime_spark_ML](https://github.com/reemzouhby/Chicago_crime_spark_ML)
- 📄 **Dataset:** [Kaggle](https://www.kaggle.com/currie32/crimes-in-chicago)
