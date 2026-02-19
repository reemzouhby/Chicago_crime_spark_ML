import os
import sys
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, regexp_replace , avg
from pyspark.sql.types import IntegerType, DoubleType, BooleanType

spark = SparkSession.builder.appName('Spark Playground').getOrCreate()
sc = spark.sparkContext

# Load in the data, using a dataframe
# header=true --> first row is column name
df = spark.read.csv('Chicago_Crimes_2001_to_2004.csv', header=True, inferSchema=False).drop("_c0")

print("INITIAL DATA OVERVIEW")
print("_" * 80)
df.printSchema()  # check types
df.show(5)  # preview first 5 rows
print(f"Initial row count: {df.count()}")

# Count nulls, empty strings, and "NULL" strings
print("\n" + "=" * 80)
print("NULL/EMPTY VALUE COUNTS (Before Cleaning)")
print("=" * 80)
null_counts = df.select([
    sum(when(col(c).isNull() | (col(c) == "") | (col(c) == "NULL"), 1).otherwise(0)).alias(c)
    for c in df.columns
])
null_counts.show()

# Calculate percentage of missing values for each column
print("\n" + "=" * 80)
print("PERCENTAGE OF MISSING VALUES")
print("=" * 80)
total_rows = df.count()
for column in df.columns:
    null_count = df.filter(col(column).isNull() | (col(column) == "") | (col(column) == "NULL")).count()
    percentage = (null_count / total_rows) * 100
    print(f"{column}: {percentage:.2f}% missing ({null_count}/{total_rows})")

# Drop duplicates
print("\n"+"_" * 80)
print("REMOVING DUPLICATES")
print("_" * 80)
initial_count = df.count()
df = df.dropDuplicates()
duplicate_count = initial_count - df.count()
print(f"Removed {duplicate_count} duplicate rows")
print(f"Remaining rows: {df.count()}")

# Replace "NULL" strings with none

for column in df.columns:
    df = df.withColumn(
        column,
        when((col(column) == "NULL") | (col(column) == ""), None).otherwise(col(column))
    )

#  Remove any header rows that got mixed into the data
before_header_removal = df.count()

# Filter out rows where ID column contains the string "ID" (the header)
df = df.filter(col("ID") != "ID") # if contain any Word id instead of nb
# here removal if inside row contains name of header
after_header_removal = df.count()
header_rows_removed = before_header_removal - after_header_removal
print(f"Removed {header_rows_removed} header rows that were mixed into data")
print(f"Remaining rows: {after_header_removal}")

# Drop rows with missing location description or district (small percentage)
print("\n" + "_" * 80)
print("DROPPING ROWS WITH MISSING LOCATION DESCRIPTION OR DISTRICT")
print("_" * 80)
before_drop = df.count()
df = df.dropna(subset=['Location Description', 'District'])
after_drop = df.count()
print(f"Dropped {before_drop - after_drop} rows")
print(f"Remaining rows: {after_drop}")

# Drop rows with missing coordinates (large percentage ~30%)
print("\n" + "=" * 80)
print("DROPPING ROWS WITH MISSING COORDINATES")
print("=" * 80)
before_drop = df.count()
df = df.dropna(subset=['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude'])
after_drop = df.count()
print(f"Dropped {before_drop - after_drop} rows")
print(f"Remaining rows: {after_drop}")

# CRITICAL: CAST DATA TYPES BEFORE COMPUTING STATISTICS
print("\n" + "_" * 80)
print("CASTING COLUMNS TO APPROPRIATE DATA TYPES")
print("_" * 80)

# Cast numeric columns
numeric_columns = {
    'District': DoubleType(),
    'Ward': DoubleType(),
    'Community Area': DoubleType(),
    'X Coordinate': DoubleType(),
    'Y Coordinate': DoubleType(),
    'Latitude': DoubleType(),
    'Longitude': DoubleType(),
    'Year': IntegerType(),
    'Beat': IntegerType()
}

for col_name, col_type in numeric_columns.items():
    df = df.withColumn(col_name, col(col_name).cast(col_type))

# Cast boolean columns
boolean_columns = ['Arrest', 'Domestic']
for col_name in boolean_columns:
    df = df.withColumn(
        col_name,
        when(col(col_name) == 'True', True)
        .when(col(col_name) == 'False', False)
        .otherwise(None)
    )

print("Data types after casting:")
df.printSchema()

# Handle Community Area - fill missing values with median (36% missing - too much to drop
print("\n" + "-" * 80)
print("FILLING MISSING COMMUNITY AREA VALUES WITH MEDIAN")
print("-" * 80)

# Check how many are missing
missing_community = df.filter(col("Community Area").isNull()).count()
total_after_coord_drop = df.count()
missing_percentage = (missing_community / total_after_coord_drop) * 100
print(f"Missing Community Area values: {missing_community} ({missing_percentage:.2f}%)")
print("This is too much to drop, so we'll impute with median")


    # Calculate median from valid values (already cast to DoubleType)
median_community2 = df.filter(
        col("Ward").isNotNull()
    ).approxQuantile("Ward", [0.5], 0.01)[0]
print(f"Median Ward : {median_community2}")

    # Fill missing values with median
df = df.withColumn(
        "Ward",
        when(col("Ward").isNull(), median_community2).otherwise(col("Ward"))
    )
print(f"Filled {missing_community} missing Ward  values with median")

print("No missing Community Area values to fill")
median_community = df.filter(
        col("Community Area").isNotNull()
    ).approxQuantile("Community Area", [0.5], 0.01)[0]

print(f"Median Community Area: {median_community}")

    # Fill missing values with median
df = df.withColumn(
        "Community Area",
        when(col("Community Area").isNull(), median_community).otherwise(col("Community Area"))
    )

df=df.cache()
# Final data overview
print("\n" + "_" * 80)
print("FINAL DATA OVERVIEW")
print("_" * 80)
df.printSchema()
df.show(5)
print(f"Final row count: {df.count()}")

# Check final null counts
print("\n" + "=" * 80)
print("NULL VALUE COUNTS (After Cleaning)")
print("=" * 80)
final_null_counts = df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
])
final_null_counts.show()

# Verify no nulls in critical columns
print("\n" + "=" * 80)
print("VERIFYING NO NULLS IN CRITICAL NUMERIC COLUMNS")
print("=" * 80)
critical_columns = ['District', 'Ward', 'Community Area', 'X Coordinate',
                    'Y Coordinate', 'Latitude', 'Longitude']
for col_name in critical_columns:
    null_count = df.filter(col(col_name).isNull()).count()
    print(f"{col_name}: {null_count} nulls")

# Check class balance
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION - PRIMARY TYPE")
print("=" * 80)
df.groupBy("Primary Type").count().orderBy("count", ascending=False).show(20)

# Show some statistics on numeric columns
print("\n" + "=" * 80)
print("NUMERIC COLUMN STATISTICS")
print("=" * 80)
df.select('District', 'Ward', 'Community Area', 'Latitude', 'Longitude').describe().show()

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE!")
print("=" * 80)
print(f"Started with: {initial_count} rows")
print(f"Final count: {df.count()} rows")
print(f"Rows removed: {initial_count - df.count()} ({((initial_count - df.count()) / initial_count * 100):.2f}%)")
# distribution of crime across district
# ============================================================================
# EXTRACT TEMPORAL FEATURES FROM DATE
# ============================================================================
print("\n" + "=" * 80)
print("EXTRACTING TEMPORAL FEATURES (Year, Month, Hour, DayOfWeek)...")
print("=" * 80)

from pyspark.sql.functions import to_timestamp, year, month, dayofweek, hour

# Convert Date string to timestamp
df = df.withColumn(
    "Date",
    to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a")
)

# Extract temporal features
df = df.withColumn("Year", year("Date")) \
       .withColumn("Month", month("Date")) \
       .withColumn("Hour", hour("Date")) \
       .withColumn("DayOfWeek", dayofweek("Date"))

print("✓ Temporal features extracted successfully")
print("New columns added: Year, Month, Hour, DayOfWeek")
df.select("Date", "Year", "Month", "Hour", "DayOfWeek").show(5)

# Final data overview
print("\n" + "_" * 80)
print("FINAL DATA OVERVIEW")
print("_" * 80)
df.printSchema()
df.show(5)
print(f"Final row count: {df.count()}")

# Check final null counts
print("\n" + "=" * 80)
print("NULL VALUE COUNTS (After Cleaning)")
print("=" * 80)
final_null_counts = df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
])
final_null_counts.show()

# Verify no nulls in critical columns
print("\n" + "=" * 80)
print("VERIFYING NO NULLS IN CRITICAL NUMERIC COLUMNS")
print("=" * 80)
critical_columns = ['District', 'Ward', 'Community Area', 'X Coordinate',
                    'Y Coordinate', 'Latitude', 'Longitude']
for col_name in critical_columns:
    null_count = df.filter(col(col_name).isNull()).count()
    print(f"{col_name}: {null_count} nulls")

# Check class balance
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION - PRIMARY TYPE")
print("=" * 80)
df.groupBy("Primary Type").count().orderBy("count", ascending=False).show(20)

# Show some statistics on numeric columns
print("\n" + "=" * 80)
print("NUMERIC COLUMN STATISTICS")
print("=" * 80)
df.select('District', 'Ward', 'Community Area', 'Latitude', 'Longitude').describe().show()

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETE!")
print("=" * 80)
print(f"Started with: {initial_count} rows")
print(f"Final count: {df.count()} rows")
print(f"Rows removed: {initial_count - df.count()} ({((initial_count - df.count()) / initial_count * 100):.2f}%)")


# CRIME DISTRIBUTION ANALYSIS

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME ACROSS DISTRICT")
print("=" * 80)
df.groupBy("District").count().orderBy("count", ascending=False).show()

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY PRIMARY TYPE")
print("=" * 80)
df.groupBy("Primary Type").count().orderBy("count", ascending=False).show()

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY DISTRICT AND PRIMARY TYPE")
print("=" * 80)
df.groupBy("District", "Primary Type").count().orderBy("count", ascending=False).show(30)

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY YEAR")
print("=" * 80)
df.groupBy("Year").count().orderBy("Year").show()

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY DISTRICT AND YEAR")
print("=" * 80)
df.groupBy("District", "Year") \
  .count() \
  .orderBy("District", "Year") \
  .show(50)

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY DISTRICT, YEAR, AND PRIMARY TYPE (Top 30)")
print("=" * 80)
df.groupBy("District", "Year", "Primary Type") \
  .count() \
  .orderBy("District", "Year", "count", ascending=False) \
  .show(30)


# TEMPORAL ANALYSIS - HOURLY PATTERNS time = M, d ,y ; h inn24 h

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY DISTRICT AND HOUR (Top 50)")
print("=" * 80)
df.groupBy("District", "Hour") \
  .count() \
  .orderBy("District", "Hour") \
  .show(50)

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY DISTRICT AND YEAR (Sorted by Year)")
print("=" * 80)
df.groupBy("District", "Year") \
  .count() \
  .orderBy("Year", ascending=False) \
  .show(50)

print("\n" + "=" * 80)
print("DISTRIBUTION OF CRIME BY PRIMARY TYPE AND HOUR (Top 30)")
print("=" * 80)
print("Analysis: Shows crime patterns by time of day (e.g., theft during day, assault at night)")
df.groupBy("Primary Type", "Hour") \
  .count() \
  .orderBy("Primary Type", "count", ascending=False) \
  .show(30)

print("\n" + "=" * 80)
print("HOURLY CRIME DISTRIBUTION (Overall)")
print("=" * 80)
df.groupBy("Hour") \
  .count() \
  .orderBy("Hour") \
  .show(24)

print("\n" + "=" * 80)
print("CREATING CRIME DISTRIBUTION MAP...")
print("=" * 80)

# Use ALL data for accurate visualization
print("Using ALL crime data for complete and accurate map visualization")
all_data_pd = df.select("Latitude", "Longitude", "Primary Type", "District").toPandas()

print(f"Using {len(all_data_pd)} crimes for map visualization (complete dataset)")

# Create base map centered on Chicago
chicago_map = folium.Map(
    location=[41.8781, -87.6298],  # Chicago coordinates
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Add a heat map layer with all data
heat_data = [[row['Latitude'], row['Longitude']] for index, row in all_data_pd.iterrows()]
HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(chicago_map)

# Save the map
chicago_map.save('chicago_crime_heatmap.html')
print("✓ Heat map saved as 'chicago_crime_heatmap.html' (using ALL crime data)")
print(f"  → Map contains {len(all_data_pd):,} crime locations")

# ============================================================================
# CREATE CRIME TYPE DISTRIBUTION MAP WITH COLOR-CODED LEGEND
# ============================================================================
print("\n" + "=" * 80)
print("CREATING CRIME TYPE DISTRIBUTION MAP WITH COLOR LEGEND...")
print("=" * 80)

# Get top crime types for visualization
top_crime_types = df.groupBy("Primary Type").count().orderBy("count", ascending=False).limit(10).toPandas()
top_crime_list = top_crime_types['Primary Type'].tolist()

print(f"Mapping top {len(top_crime_list)} crime types with distinct colors")

# Color palette for different crime types
crime_colors = {
    'THEFT': 'red',
    'BATTERY': 'darkred',
    'CRIMINAL DAMAGE': 'orange',
    'NARCOTICS': 'purple',
    'OTHER OFFENSE': 'gray',
    'ASSAULT': 'darkblue',
    'BURGLARY': 'brown',
    'MOTOR VEHICLE THEFT': 'pink',
    'ROBBERY': 'black',
    'CRIMINAL TRESPASS': 'green',
    'DECEPTIVE PRACTICE': 'lightblue',
    'PROSTITUTION': 'darkgreen',
    'WEAPONS VIOLATION': 'cadetblue',
    'OFFENSE INVOLVING CHILDREN': 'beige',
    'PUBLIC PEACE VIOLATION': 'lightgray'
}

# Sample data for each crime type to avoid overwhelming the map (use 100 samples per type)
sampled_crime_data = []
for crime_type in top_crime_list:
    crime_sample = df.filter(col("Primary Type") == crime_type).sample(fraction=0.005, seed=42).toPandas()
    sampled_crime_data.append(crime_sample)

crime_type_pd = pd.concat(sampled_crime_data, ignore_index=True)
print(f"Using {len(crime_type_pd):,} sampled crimes (balanced across types) for better map performance")

# Create the crime type map
crime_type_map = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11,
    tiles='CartoDB positron'
)

# Add markers for each crime type with different colors
from folium.plugins import MarkerCluster

# Create feature groups for each crime type (for legend)
crime_groups = {}
for crime_type in top_crime_list:
    color = crime_colors.get(crime_type, 'blue')
    crime_groups[crime_type] = folium.FeatureGroup(name=crime_type)

    # Filter data for this crime type
    crime_data = crime_type_pd[crime_type_pd['Primary Type'] == crime_type]

    # Add circle markers for this crime type
    for idx, row in crime_data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            popup=f"<b>{crime_type}</b><br>District: {int(row['District'])}",
            tooltip=crime_type,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1
        ).add_to(crime_groups[crime_type])

    # Add the feature group to the map
    crime_groups[crime_type].add_to(crime_type_map)

# Add layer control (legend) to toggle crime types
folium.LayerControl(collapsed=False).add_to(crime_type_map)

# Add a custom legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 220px; height: auto; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; border-radius: 5px; padding: 10px">
<h4 style="margin-top:0; margin-bottom:10px;">Crime Types</h4>
'''

for crime_type in top_crime_list:
    color = crime_colors.get(crime_type, 'blue')
    legend_html += f'<p style="margin:5px;"><span style="color:{color};">●</span> {crime_type}</p>'

legend_html += '</div>'
crime_type_map.get_root().html.add_child(folium.Element(legend_html))

# Save the crime type map
crime_type_map.save('chicago_crime_by_type_map.html')
print("✓ Crime type distribution map saved as 'chicago_crime_by_type_map.html'")
print("  → Map includes color-coded legend for each crime type")
print("  → You can toggle crime types on/off using the layer control")


# CREATE DISTRICT-BASED AGGREGATED MAP
print("\n" + "=" * 80)
print("CREATING DISTRICT-LEVEL CRIME DISTRIBUTION MAP...")
print("=" * 80)

# Get average coordinates per district
district_coords = df.groupBy("District").agg(
    avg("Latitude").alias("Avg_Lat"),
    avg("Longitude").alias("Avg_Long")
).toPandas()

# Get crime counts per district
district_crimes = df.groupBy("District").count().toPandas()
district_crimes.columns = ["District", "Crime_Count"]

# Merge the data
district_map_data = district_coords.merge(district_crimes, on="District")

print(f"Processing {len(district_map_data)} districts")

# Create district map
district_map = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11,
    tiles='CartoDB positron'
)

# Add markers for each district
for idx, row in district_map_data.iterrows():
    folium.CircleMarker(
        location=[row['Avg_Lat'], row['Avg_Long']],
        radius=row['Crime_Count'] / 5000,  # Scale circle size by crime count
        popup=f"<b>District {int(row['District'])}</b><br>Crimes: {row['Crime_Count']:,}",
        tooltip=f"District {int(row['District'])}",
        color='darkred',
        fill=True,
        fillColor='red',
        fillOpacity=0.6
    ).add_to(district_map)

district_map.save('chicago_crime_by_district.html')
print("✓ District map saved as 'chicago_crime_by_district.html'")


# ============================================================================
# CREATE CRIME TYPE DISTRIBUTION BAR CHART
# ============================================================================
print("\n" + "=" * 80)
print("CREATING CRIME TYPE VISUALIZATION...")
print("=" * 80)

# Get top 10 crime types
crime_types = df.groupBy("Primary Type").count().orderBy("count", ascending=False).limit(10).toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(data=crime_types, x='count', y='Primary Type', palette='viridis')
plt.title('Top 10 Crime Types in Chicago (2001-2004)', fontsize=16, fontweight='bold')
plt.xlabel('Number of Crimes', fontsize=12)
plt.ylabel('Crime Type', fontsize=12)
plt.tight_layout()
plt.savefig('top_10_crime_types.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Crime type chart saved as 'top_10_crime_types.png'")

# ============================================================================
# CREATE DISTRICT CRIME COUNT BAR CHART
# ============================================================================
print("\n" + "=" * 80)
print("CREATING DISTRICT CRIME DISTRIBUTION CHART...")
print("=" * 80)

# Get district crime counts sorted
district_crimes_sorted = district_crimes.sort_values('Crime_Count', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(data=district_crimes_sorted, x='District', y='Crime_Count', palette='rocket')
plt.title('Crime Distribution Across Chicago Districts (2001-2004)', fontsize=16, fontweight='bold')
plt.xlabel('District Number', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('district_crime_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ District distribution chart saved as 'district_crime_distribution.png'")

# ============================================================================
# CREATE YEARLY CRIME TREND CHART
# ============================================================================
print("\n" + "=" * 80)
print("CREATING YEARLY CRIME TREND CHART...")
print("=" * 80)

# Get crime counts by year
yearly_crimes = df.groupBy("Year").count().orderBy("Year").toPandas()

plt.figure(figsize=(10, 6))
plt.plot(yearly_crimes['Year'], yearly_crimes['count'], marker='o', linewidth=2, markersize=8, color='darkred')
plt.title('Crime Trend Over Years (2001-2004)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('yearly_crime_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Yearly trend chart saved as 'yearly_crime_trend.png'")

# ============================================================================
# CREATE HEATMAP: CRIME TYPE vs DISTRICT
# ============================================================================
print("\n" + "=" * 80)
print("CREATING CRIME TYPE vs DISTRICT HEATMAP...")
print("=" * 80)

# Get top 15 crime types for better visualization
top_crimes = df.groupBy("Primary Type").count().orderBy("count", ascending=False).limit(15).select("Primary Type").toPandas()
top_crime_list = top_crimes['Primary Type'].tolist()

# Filter data for top crime types
crime_district_data = df.filter(col("Primary Type").isin(top_crime_list)) \
    .groupBy("District", "Primary Type") \
    .count() \
    .toPandas()

# Pivot to create heatmap matrix
heatmap_data = crime_district_data.pivot(index='Primary Type', columns='District', values='count')
heatmap_data = heatmap_data.fillna(0)

# Create heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Crime Count'})
plt.title('Crime Type Distribution Across Districts (Top 15 Crime Types)', fontsize=16, fontweight='bold')
plt.xlabel('District Number', fontsize=12)
plt.ylabel('Crime Type', fontsize=12)
plt.tight_layout()
plt.savefig('crime_type_district_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Crime type vs district heatmap saved as 'crime_type_district_heatmap.png'")

# ============================================================================
# CREATE HEATMAP: YEAR vs DISTRICT
# ============================================================================
print("\n" + "=" * 80)
print("CREATING YEAR vs DISTRICT HEATMAP...")
print("=" * 80)

# Get year-district crime counts
year_district_data = df.groupBy("Year", "District") \
    .count() \
    .toPandas()

# Pivot to create heatmap matrix
year_heatmap_data = year_district_data.pivot(index='Year', columns='District', values='count')
year_heatmap_data = year_heatmap_data.fillna(0)

plt.figure(figsize=(16, 6))
sns.heatmap(year_heatmap_data, cmap='Blues', annot=True, fmt='g', cbar_kws={'label': 'Crime Count'})
plt.title('Crime Distribution by Year and District', fontsize=16, fontweight='bold')
plt.xlabel('District Number', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.tight_layout()
plt.savefig('year_district_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Year vs district heatmap saved as 'year_district_heatmap.png'")


# CREATE HOURLY CRIME PATTERN CHART

print("\n" + "=" * 80)
print("CREATING HOURLY CRIME PATTERN CHART...")
print("=" * 80)

# Get hourly crime counts
hourly_crimes = df.groupBy("Hour").count().orderBy("Hour").toPandas()

plt.figure(figsize=(14, 6))
plt.plot(hourly_crimes['Hour'], hourly_crimes['count'], marker='o', linewidth=2, markersize=8, color='darkblue')
plt.fill_between(hourly_crimes['Hour'], hourly_crimes['count'], alpha=0.3)
plt.title('Crime Distribution by Hour of Day (2001-2004)', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hourly_crime_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Hourly crime pattern saved as 'hourly_crime_pattern.png'")

# ============================================================================
# CREATE HEATMAP: CRIME TYPE vs HOUR (DAY/NIGHT PATTERNS)
# ============================================================================
print("\n" + "=" * 80)
print("CREATING CRIME TYPE vs HOUR HEATMAP (Day/Night Patterns)...")
print("=" * 80)

# Get top 12 crime types for better visualization
top_crimes_hourly = df.groupBy("Primary Type").count().orderBy("count", ascending=False).limit(12).select("Primary Type").toPandas()
top_crime_list_hourly = top_crimes_hourly['Primary Type'].tolist()

# Filter data for top crime types and get hourly distribution
crime_hour_data = df.filter(col("Primary Type").isin(top_crime_list_hourly)) \
    .groupBy("Primary Type", "Hour") \
    .count() \
    .toPandas()

# Pivot to create heatmap matrix
hour_heatmap_data = crime_hour_data.pivot(index='Primary Type', columns='Hour', values='count')
hour_heatmap_data = hour_heatmap_data.fillna(0)

# Create heatmap
plt.figure(figsize=(18, 8))
sns.heatmap(hour_heatmap_data, cmap='RdYlGn_r', annot=False, fmt='g', cbar_kws={'label': 'Crime Count'})
plt.title('Crime Type Distribution by Hour of Day (Shows Day vs Night Patterns)', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day (0=Midnight, 12=Noon)', fontsize=12)
plt.ylabel('Crime Type', fontsize=12)
plt.tight_layout()
plt.savefig('crime_type_hour_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Crime type vs hour heatmap saved as 'crime_type_hour_heatmap.png'")
print("  → This shows patterns like: THEFT during day, ASSAULT at night")

# ============================================================================
# CREATE DAY OF WEEK CRIME PATTERN
# ============================================================================
print("\n" + "=" * 80)
print("CREATING DAY OF WEEK CRIME PATTERN...")
print("=" * 80)

# Get day of week crime counts (1=Sunday, 7=Saturday in Spark)
dow_crimes = df.groupBy("DayOfWeek").count().orderBy("DayOfWeek").toPandas()

# Map day numbers to names
day_names = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
dow_crimes['DayName'] = dow_crimes['DayOfWeek'].map(day_names)

plt.figure(figsize=(12, 6))
sns.barplot(data=dow_crimes, x='DayName', y='count', palette='mako')
plt.title('Crime Distribution by Day of Week (2001-2004)', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('day_of_week_crime_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Day of week pattern saved as 'day_of_week_crime_pattern.png'")

# ============================================================================
# CREATE MONTHLY CRIME PATTERN
# ============================================================================
print("\n" + "=" * 80)
print("CREATING MONTHLY CRIME PATTERN...")
print("=" * 80)

# Get monthly crime counts
monthly_crimes = df.groupBy("Month").count().orderBy("Month").toPandas()

# Map month numbers to names
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
monthly_crimes['MonthName'] = monthly_crimes['Month'].map(month_names)

plt.figure(figsize=(12, 6))
plt.plot(monthly_crimes['Month'], monthly_crimes['count'], marker='o', linewidth=2, markersize=10, color='crimson')
plt.fill_between(monthly_crimes['Month'], monthly_crimes['count'], alpha=0.3, color='crimson')
plt.title('Crime Distribution by Month (2001-2004)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(range(1, 13), [month_names[i] for i in range(1, 13)])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_crime_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Monthly pattern saved as 'monthly_crime_pattern.png'")

# CREATE CORRELATION MATRIX
# ============================================================================
print("\n" + "=" * 80)
print("CREATING CORRELATION MATRIX...")
print("=" * 80)

# Select numeric columns for correlation analysis
numeric_cols = ['District', 'Ward', 'Community Area', 'Beat', 'Year', 'Month',
                'Hour', 'DayOfWeek', 'Latitude', 'Longitude', 'X Coordinate', 'Y Coordinate']

# Convert boolean columns to numeric (0/1)
df_corr = df.withColumn("Arrest_Num", when(col("Arrest") == True, 1).otherwise(0)) \
            .withColumn("Domestic_Num", when(col("Domestic") == True, 1).otherwise(0))

# Add the boolean columns to numeric list
correlation_cols = numeric_cols + ['Arrest_Num', 'Domestic_Num']

# Select only the columns we want to correlate
df_for_corr = df_corr.select(correlation_cols).toPandas()

# Calculate correlation matrix
correlation_matrix = df_for_corr.corr()

# Create correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Chicago Crime Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation matrix saved as 'correlation_matrix.png'")
print("\n✅ DATA IS NOW READY FOR MODEL TRAINING!")

# training ML MOdel random forest + class weight


# 1. Create your categories
from pyspark.sql.functions import when

df = df.withColumn("Crime_Category",
    when(col("Primary Type").isin(["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT",
                                   "CRIMINAL DAMAGE", "CRIMINAL TRESPASS"]), "PROPERTY_CRIME")
    .when(col("Primary Type").isin(["BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE"]), "VIOLENT_CRIME")
    .when(col("Primary Type").isin(["NARCOTICS", "OTHER NARCOTIC VIOLATION"]), "DRUG_CRIME")
    .when(col("Primary Type").isin(["PROSTITUTION", "GAMBLING", "PUBLIC PEACE VIOLATION",
                                    "LIQUOR LAW VIOLATION"]), "PUBLIC_ORDER")
    .when(col("Primary Type").isin(["SEX OFFENSE", "CRIM SEXUAL ASSAULT"]), "SEX_CRIME")
    .when(col("Primary Type") == "WEAPONS VIOLATION", "WEAPONS_CRIME")
    .otherwise("OTHER")
)

# 2. Check the distribution
print("\n" + "=" * 80)
print("CRIME CATEGORY DISTRIBUTION")
print("=" * 80)
df.groupBy("Crime_Category").count().orderBy("count", ascending=False).show()

