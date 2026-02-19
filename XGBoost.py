import numpy as np
import pandas as pd
import xgboost as xgb
import os

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
    "Chicago_Crimes_2001_to_2004.csv",
    header=True,
    inferSchema=False
).drop("_c0")

df = df.dropDuplicates()

# Clean nulls
for column in df.columns:
    df = df.withColumn(
        column,
        when((col(column) == "NULL") | (col(column) == ""), None)
        .otherwise(col(column))
    )

df = df.dropna(subset=[
    'Location Description', 'District',
    'Latitude', 'Longitude', 'Primary Type', 'Description'
])

# Cast numeric columns
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
# SAFE TIMESTAMP PARSING
# -------------------------------------------------------------------
df = df.withColumn("Date", try_to_timestamp(col("Date"), lit("MM/dd/yyyy hh:mm:ss a")))
df = df.dropna(subset=["Date"])

# Time features
df = df.withColumn("Year", year("Date"))
df = df.withColumn("Month", month("Date"))
df = df.withColumn("Hour", hour("Date"))
df = df.withColumn("DayOfWeek", dayofweek("Date"))

# -------------------------------------------------------------------
# ARREST FLAG
# -------------------------------------------------------------------
df = df.withColumn("Arrest_Flag",
    when(col("Arrest") == "true", 1.0).otherwise(0.0))

# -------------------------------------------------------------------
# CRIME CATEGORIES
# -------------------------------------------------------------------
df = df.withColumn("Crime_Category",
    when(col("Primary Type").isin(
        ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT",
         "CRIMINAL DAMAGE", "CRIMINAL TRESPASS"]),
        "PROPERTY_CRIME")
    .when(col("Primary Type").isin(
        ["BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE"]),
        "VIOLENT_CRIME")
    .when(col("Primary Type").isin(
        ["NARCOTICS", "OTHER NARCOTIC VIOLATION"]),
        "DRUG_CRIME")
    .when(col("Primary Type") == "WEAPONS VIOLATION",
        "WEAPONS_CRIME")
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
# SPLIT FIRST
# -------------------------------------------------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Use persist instead of checkpoint (avoids Windows Hadoop native lib issue)
from pyspark.storagelevel import StorageLevel
train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

# Force materialization
print(f"Train raw count: {train_df.count()}")
print(f"Test raw count: {test_df.count()}")

# -------------------------------------------------------------------
# STRING ENCODINGS (fit on train only)
# -------------------------------------------------------------------
loc_indexer = StringIndexer(
    inputCol="Location Description",
    outputCol="Location_Index",
    handleInvalid="keep"
)
loc_model = loc_indexer.fit(train_df)
train_df = loc_model.transform(train_df)
test_df = loc_model.transform(test_df)

desc_indexer = StringIndexer(
    inputCol="Description",
    outputCol="Description_Index",
    handleInvalid="keep"
)
desc_model = desc_indexer.fit(train_df)
train_df = desc_model.transform(train_df)
test_df = desc_model.transform(test_df)

# -------------------------------------------------------------------
# DENSITY FEATURES (train only)
# -------------------------------------------------------------------
district_counts = train_df.groupBy("District") \
    .agg(spark_count("*").alias("District_Crime_Count"))
train_df = train_df.join(district_counts, "District", "left")
test_df = test_df.join(district_counts, "District", "left")
train_df = train_df.fillna({"District_Crime_Count": 0})
test_df = test_df.fillna({"District_Crime_Count": 0})

beat_counts = train_df.groupBy("Beat") \
    .agg(spark_count("*").alias("Beat_Crime_Count"))
train_df = train_df.join(beat_counts, "Beat", "left")
test_df = test_df.join(beat_counts, "Beat", "left")
train_df = train_df.fillna({"Beat_Crime_Count": 0})
test_df = test_df.fillna({"Beat_Crime_Count": 0})

community_counts = train_df.groupBy("Community Area") \
    .agg(spark_count("*").alias("Community_Crime_Count"))
train_df = train_df.join(community_counts, "Community Area", "left")
test_df = test_df.join(community_counts, "Community Area", "left")
train_df = train_df.fillna({"Community_Crime_Count": 0})
test_df = test_df.fillna({"Community_Crime_Count": 0})

# Persist after joins
train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

# -------------------------------------------------------------------
# BALANCE TRAIN
# -------------------------------------------------------------------
class_counts = train_df.groupBy("Crime_Category").count().collect()
class_dict = {row['Crime_Category']: row['count'] for row in class_counts}

print("\nClass distribution before balancing:")
for k, v in sorted(class_dict.items()):
    print(f"  {k}: {v}")

min_count = min(class_dict.values())
target = min(int(min_count * 5), 150000)
print(f"\nBalancing target per class: {target}")

balanced = []
for crime_type, count in class_dict.items():
    subset = train_df.filter(col("Crime_Category") == crime_type)
    fraction = target / count
    sampled = subset.sample(
        withReplacement=(count < target),
        fraction=min(fraction * 1.1, 1.0) if count >= target else fraction,
        seed=42
    )
    balanced.append(sampled.limit(target))

train_balanced = balanced[0]
for i in range(1, len(balanced)):
    train_balanced = train_balanced.union(balanced[i])

train_balanced = train_balanced.persist(StorageLevel.MEMORY_AND_DISK)
print(f"Balanced train size: {train_balanced.count()}")

# -------------------------------------------------------------------
# FEATURE LIST
# -------------------------------------------------------------------
feature_columns = [
    'District', 'Community Area', 'Year', 'Month', 'Hour', 'DayOfWeek',
    'Latitude', 'Longitude', 'Beat', 'Ward',
    'Hour_sin', 'Hour_cos', 'IsWeekend',
    'Distance_from_center',
    'District_Crime_Count', 'Beat_Crime_Count', 'Community_Crime_Count',
    'Location_Index', 'Arrest_Flag',
    'Description_Index'
]

# -------------------------------------------------------------------
# LABEL ENCODING
# -------------------------------------------------------------------
labelIndexer = StringIndexer(inputCol="Crime_Category", outputCol="label")
label_model = labelIndexer.fit(train_balanced)
train_labeled = label_model.transform(train_balanced)
test_labeled = label_model.transform(test_df)

print("\nLabel mapping:")
for i, name in enumerate(label_model.labels):
    print(f"  {i} -> {name}")

# -------------------------------------------------------------------
# CONVERT TO PANDAS
# -------------------------------------------------------------------
print("\nConverting train to Pandas...")
train_pd = train_labeled.select(feature_columns + ['label']) \
    .repartition(4) \
    .toPandas()

print("Converting test to Pandas...")
test_pd = test_labeled.select(feature_columns + ['label']) \
    .repartition(4) \
    .toPandas()

train_pd = train_pd.fillna(0)
test_pd = test_pd.fillna(0)

X_train = train_pd[feature_columns].values.astype(np.float32)
y_train = train_pd['label'].values.astype(np.int32)
X_test = test_pd[feature_columns].values.astype(np.float32)
y_test = test_pd['label'].values.astype(np.int32)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Free Spark memory
spark.catalog.clearCache()

# -------------------------------------------------------------------
# TRAIN XGBOOST
# -------------------------------------------------------------------
print("\nTraining XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

params = {
    'objective': 'multi:softprob',
    'num_class': len(label_model.labels),
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'eval_metric': 'mlogloss',
    'seed': 42,
    'nthread': -1
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=30,
    verbose_eval=25
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
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame(
    list(importance.items()),
    columns=['Feature', 'Importance']
).sort_values('Importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(importance_df.head(10).to_string(index=False))

# -------------------------------------------------------------------
# SAVE
# -------------------------------------------------------------------
model.save_model("crime_xgboost_clean.json")
print("\nModel saved to crime_xgboost_clean.json")

spark.stop()