# 🚨 Chicago Crime Analysis & Prediction

> A full end-to-end Big Data and Machine Learning project on 1.85 million crime records from Chicago (2001–2004), built with PySpark, XGBoost, and Streamlit.

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
│
├── xgboost_model.py          # Feature engineering, model training, and evaluation
│
├── streamlit_app.py          # Interactive Streamlit dashboard
│
├── Chicago_Crimes_2001_to_2004.csv   # Raw dataset (download separately)
│
├── crime_xgboost_clean.json  # Saved trained XGBoost model
│
├── requirements.txt          # Python dependencies
│
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
- Groups the 30+ primary crime types into **5 macro-categories**:

| Category | Crime Types Included |
|----------|----------------------|
| `PROPERTY_CRIME` | Theft, Burglary, Motor Vehicle Theft, Criminal Damage, Criminal Trespass |
| `VIOLENT_CRIME` | Battery, Assault, Robbery, Homicide |
| `DRUG_CRIME` | Narcotics, Other Narcotic Violation |
| `WEAPONS_CRIME` | Weapons Violation |
| `OTHER` | All remaining types |

- Generates all EDA visualisations (bar charts, heatmaps, trend lines, correlation matrix)
- Produces 3 interactive Folium maps (heatmap, crime type map, district map)

---

### `xgboost_model.py`
The full ML pipeline from feature engineering to model evaluation:

- **Data split:** 80/20 train/test split *before* any feature engineering (no leakage)
- **Feature engineering (20 features):**
  - Temporal: `Year`, `Month`, `Hour`, `DayOfWeek`
  - Cyclical hour encoding: `Hour_sin`, `Hour_cos`
  - Weekend flag: `IsWeekend`
  - Spatial: `Latitude`, `Longitude`, `District`, `Beat`, `Ward`, `Community Area`
  - Distance from Chicago city centre: `Distance_from_center`
  - Density features (train-only): `District_Crime_Count`, `Beat_Crime_Count`, `Community_Crime_Count`
  - Encoded strings: `Location_Index`, `Description_Index` (via PySpark `StringIndexer`)
  - `Arrest_Flag`
- **Class balancing:** oversampling/undersampling each category to a target count
- **Model:** XGBoost with `multi:softprob`, `max_depth=8`, `learning_rate=0.05`, early stopping
- **Evaluation:** Accuracy, weighted F1-score, per-class classification report
- Saves the trained model as `crime_xgboost_clean.json`

---

### `streamlit_app.py`
A 5-page interactive web dashboard:

| Page | Description |
|------|-------------|
| 📊 Overview | Key metrics, crime category definitions, model performance table |
| 📈 Visualisations | All EDA charts viewable individually or as a gallery |
| 🔮 Crime Prediction | Input form → live XGBoost prediction with probability chart |
| 📍 Interactive Maps | Embedded Folium HTML maps |
| 📉 Trends Analysis | Plotly charts of yearly and categorical crime trends |

---

## 🧪 Results

| Model | Classes | Accuracy | F1 (weighted) |
|-------|---------|----------|----------------|
| Random Forest (initial) | 20+ raw types | 8% | Very low |
| Random Forest (improved) | 20+ raw types | 34% | Low |
| **XGBoost (final)** | **5 categories** | **97.4%** | **0.974** |

### Per-class breakdown (XGBoost):

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| DRUG_CRIME | 0.98 | 0.98 | 0.98 |
| OTHER | 0.90 | 0.97 | 0.93 |
| PROPERTY_CRIME | 1.00 | 0.98 | 0.99 |
| VIOLENT_CRIME | 0.98 | 0.97 | 0.97 |
| WEAPONS_CRIME | 0.70 | 0.99 | 0.82 |

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
This will train the model and save it as `crime_xgboost_clean.json`.

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
- **Records:** ~1.85 million
- **Columns:** 22
- **Period:** 2001–2004

---

## 🔗 Links

- 📁 **GitHub Repository:** [https://github.com/reemzouhby/Chicago_crime_spark_ML](https://github.com/reemzouhby/Chicago_crime_spark_ML)
- 📄 **Dataset:** [Kaggle](https://www.kaggle.com/currie32/crimes-in-chicago)
