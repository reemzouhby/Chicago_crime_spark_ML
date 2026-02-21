import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, year, month, dayofweek, hour,
    sin, cos, radians, sqrt, pow as spark_pow,
    count as spark_count, lit
)
from pyspark.sql.functions import try_to_timestamp
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import StringIndexer

from sklearn.metrics import accuracy_score, f1_score, classification_report

# -------------------------------------------------------------------
# SPARK INITIALIZATION
# -------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Crime_XGBoost_Improved") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.storage.blockManagerSlaveTimeoutMs", "800000") \
    .config("spark.shuffle.io.maxRetries", "10") \
    .config("spark.shuffle.io.retryWait", "60s") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "8") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark initialized")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = spark.read.csv(
    r"C:\Users\Omen\PycharmProjects\lab3\Chicago_project\Chicago_Crimes_2001_to_2004.csv",
    header=True,
    inferSchema=False
).drop("_c0")

df = df.dropDuplicates()

for column in df.columns:
    df = df.withColumn(
        column,
        when((col(column) == "NULL") | (col(column) == ""), None)
        .otherwise(col(column))
    )

df = df.dropna(subset=[
    'Location Description', 'District',
    'Latitude', 'Longitude', 'Primary Type'
])

numeric_cols = {
    'District': DoubleType(),
    'Ward': DoubleType(),
    'Community Area': DoubleType(),
    'Latitude': DoubleType(),
    'Longitude': DoubleType(),
    'Year': IntegerType(),
    'Beat': IntegerType()
}
for col_name, col_type in numeric_cols.items():
    df = df.withColumn(col_name, col(col_name).cast(col_type))

# -------------------------------------------------------------------
# TIMESTAMP PARSING
# -------------------------------------------------------------------
df = df.withColumn("Date", try_to_timestamp(col("Date"), lit("MM/dd/yyyy hh:mm:ss a")))
df = df.dropna(subset=["Date"])

df = df.withColumn("Year",      year("Date"))
df = df.withColumn("Month",     month("Date"))
df = df.withColumn("Hour",      hour("Date"))
df = df.withColumn("DayOfWeek", dayofweek("Date"))

# -------------------------------------------------------------------
# ARREST FLAG
# -------------------------------------------------------------------
df = df.withColumn("Arrest_Flag",
    when(col("Arrest") == "true", 1.0).otherwise(0.0))

# -------------------------------------------------------------------
# CRIME CATEGORIES
# FIX: WEAPONS_CRIME merged into VIOLENT_CRIME
#      Reason: only 12,591 samples (too rare) and logically related
#              to violent crime — both involve physical threat/harm
#              This gives the model 4 clean balanced classes
# -------------------------------------------------------------------
df = df.withColumn("Crime_Category",
    when(col("Primary Type").isin(
        ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT",
         "CRIMINAL DAMAGE", "CRIMINAL TRESPASS"]),
        "PROPERTY_CRIME")
    .when(col("Primary Type").isin(
        ["BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE",
         "WEAPONS VIOLATION"]),
        "VIOLENT_CRIME")
    .when(col("Primary Type").isin(
        ["NARCOTICS", "OTHER NARCOTIC VIOLATION"]),
        "DRUG_CRIME")
    .otherwise("OTHER")
)

# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
df = df.withColumn("Hour_sin", sin(radians(col("Hour") * 15)))
df = df.withColumn("Hour_cos", cos(radians(col("Hour") * 15)))
df = df.withColumn("IsWeekend",
    when(col("DayOfWeek").isin([1, 7]), 1.0).otherwise(0.0))

chicago_lat = 41.8781
chicago_lon = -87.6298
df = df.withColumn(
    "Distance_from_center",
    sqrt(
        spark_pow(col("Latitude") - chicago_lat, 2) +
        spark_pow(col("Longitude") - chicago_lon, 2)
    ) * 111
)

# -------------------------------------------------------------------
# TIME-BASED SPLIT
# -------------------------------------------------------------------
from pyspark.storagelevel import StorageLevel

train_df = df.filter(col("Year") <= 2003)
test_df  = df.filter(col("Year") == 2004)

train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df  = test_df.persist(StorageLevel.MEMORY_AND_DISK)

print(f"Train rows (2001-2003): {train_df.count()}")
print(f"Test  rows (2004):      {test_df.count()}")

# -------------------------------------------------------------------
# STRING ENCODINGS 
# -------------------------------------------------------------------
loc_indexer = StringIndexer(
    inputCol="Location Description",
    outputCol="Location_Index",
    handleInvalid="keep"
)
loc_model = loc_indexer.fit(train_df)
train_df  = loc_model.transform(train_df)
test_df   = loc_model.transform(test_df)

indexer_mappings = {"location_labels": loc_model.labels}
with open("indexer_mappings.json", "w") as f:
    json.dump(indexer_mappings, f, indent=2)
print("Saved indexer_mappings.json")

# -------------------------------------------------------------------
# DENSITY FEATURES (train only)
# -------------------------------------------------------------------
district_counts = train_df.groupBy("District") \
    .agg(spark_count("*").alias("District_Crime_Count"))
train_df = train_df.join(district_counts, "District", "left")
test_df  = test_df.join(district_counts,  "District", "left")
train_df = train_df.fillna({"District_Crime_Count": 0})
test_df  = test_df.fillna({"District_Crime_Count": 0})

beat_counts = train_df.groupBy("Beat") \
    .agg(spark_count("*").alias("Beat_Crime_Count"))
train_df = train_df.join(beat_counts, "Beat", "left")
test_df  = test_df.join(beat_counts,  "Beat", "left")
train_df = train_df.fillna({"Beat_Crime_Count": 0})
test_df  = test_df.fillna({"Beat_Crime_Count": 0})

community_counts = train_df.groupBy("Community Area") \
    .agg(spark_count("*").alias("Community_Crime_Count"))
train_df = train_df.join(community_counts, "Community Area", "left")
test_df  = test_df.join(community_counts,  "Community Area", "left")
train_df = train_df.fillna({"Community_Crime_Count": 0})
test_df  = test_df.fillna({"Community_Crime_Count": 0})

train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df  = test_df.persist(StorageLevel.MEMORY_AND_DISK)

# -------------------------------------------------------------------
# FEATURE LIST
# -------------------------------------------------------------------
feature_columns = [
    'District', 'Community Area', 'Year', 'Month', 'Hour', 'DayOfWeek',
    'Latitude', 'Longitude', 'Beat', 'Ward',
    'Hour_sin', 'Hour_cos', 'IsWeekend',
    'Distance_from_center',
    'District_Crime_Count', 'Beat_Crime_Count', 'Community_Crime_Count',
    'Location_Index'
]

# -------------------------------------------------------------------
# LABEL ENCODING
# -------------------------------------------------------------------
labelIndexer  = StringIndexer(inputCol="Crime_Category", outputCol="label")
label_model   = labelIndexer.fit(train_df)
train_labeled = label_model.transform(train_df)
test_labeled  = label_model.transform(test_df)

print("\nLabel mapping:")
for i, name in enumerate(label_model.labels):
    print(f"  {i} -> {name}")

indexer_mappings["label_labels"] = label_model.labels
with open("indexer_mappings.json", "w") as f:
    json.dump(indexer_mappings, f, indent=2)
print("Updated indexer_mappings.json with label mapping")

# -------------------------------------------------------------------
# CONVERT TO PANDAS
# -------------------------------------------------------------------
print("\nConverting train to Pandas...")
train_pd = train_labeled.select(feature_columns + ['label']) \
    .repartition(4).toPandas()

print("Converting test to Pandas...")
test_pd = test_labeled.select(feature_columns + ['label']) \
    .repartition(4).toPandas()

train_pd = train_pd.fillna(0)
test_pd  = test_pd.fillna(0)

X_train = train_pd[feature_columns].values.astype(np.float32)
y_train = train_pd['label'].values.astype(np.int32)
X_test  = test_pd[feature_columns].values.astype(np.float32)
y_test  = test_pd['label'].values.astype(np.int32)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

spark.catalog.clearCache()

# -------------------------------------------------------------------
# CLASS WEIGHTS
# -------------------------------------------------------------------
class_counts_arr = np.bincount(y_train)
total            = len(y_train)
n_classes        = len(label_model.labels)

class_weights = total / (n_classes * class_counts_arr)

# Cap maximum weight at 5.0 — prevents any class from dominating
class_weights = np.clip(class_weights, a_min=0.1, a_max=5.0)

sample_weights = np.array([class_weights[label] for label in y_train],
                           dtype=np.float32)

print("\nClass weights applied (capped at 5.0):")
for i, name in enumerate(label_model.labels):
    print(f"  {name}: {class_weights[i]:.4f}")

# -------------------------------------------------------------------
# TRAIN XGBOOST
# -------------------------------------------------------------------
print("\nTraining XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train,
                     weight=sample_weights,
                     feature_names=feature_columns)
dtest  = xgb.DMatrix(X_test, label=y_test,
                     feature_names=feature_columns)

params = {
    'objective':        'multi:softprob',
    'num_class':        n_classes,
    'max_depth':        7,
    'learning_rate':    0.07,
    'subsample':        0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 5,
    'reg_alpha':        0.05,
    'reg_lambda':       0.5,
    'eval_metric':      'mlogloss',
    'seed':             42,
    'nthread':          -1,
    'tree_method':      'hist',
    'grow_policy':      'lossguide',
    'max_leaves':       63,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=200,
    verbose_eval=50
)

# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------
y_pred = np.argmax(model.predict(dtest), axis=1)

print("\n===== RESULTS =====")
print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_model.labels))

# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------
importance    = model.get_score(importance_type='gain')
importance_df = pd.DataFrame(
    list(importance.items()), columns=['Feature', 'Importance']
).sort_values('Importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(importance_df.head(10).to_string(index=False))

# -------------------------------------------------------------------
# SAVE
# -------------------------------------------------------------------
model.save_model("crime_xgboost_clean1.json")
print("\nModel saved to crime_xgboost_clean.json")
print("Mappings saved to indexer_mappings.json")

spark.stop()
