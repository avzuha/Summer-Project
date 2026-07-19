"""
US Accidents Dataset — Data Cleaning & Preprocessing

"""

import pandas as pd
import numpy as np



DATA_PATH = "https://raw.githubusercontent.com/avzuha/Summer-Project/refs/heads/main/data/US_Accidents_100k.csv"


print("Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names and types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())


#  Check missing values

print("\nMissing values per column (top 20):")
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_summary[missing_summary["missing_count"] > 0].head(20))


# - If a column is >50% missing, it's usually safer to drop it.
# - If a column is <5% missing, drop just those rows or fill with median/mode.
# - In between, use judgement (we fill numeric with median, categorical with mode).

cols_to_drop = missing_summary[missing_summary["missing_pct"] > 50].index.tolist()
if cols_to_drop:
    print(f"\nDropping columns with >50% missing values: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# Fill remaining missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

print(f"\nMissing values remaining: {df.isnull().sum().sum()}")


# Remove duplicate rows

before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nRemoved {before - after} duplicate rows")


# Fix data types 


for col in ["Start_Time", "End_Time", "Weather_Timestamp"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"Converted '{col}' to datetime")

# Create useful derived columns for EDA/modeling 
if "Start_Time" in df.columns:
    df["Year"] = df["Start_Time"].dt.year
    df["Month"] = df["Start_Time"].dt.month
    df["Day"] = df["Start_Time"].dt.day
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.day_name()
    print("Added Year, Month, Day, Hour, DayOfWeek columns from Start_Time")


# Drop columns that aren't useful for analysis/modeling
# ID is just a unique identifier — useless for analysis.
# Description is free text not usable.

drop_candidates = ["ID", "Description", "End_Lat", "End_Lng", "Source"]
existing_drops = [c for c in drop_candidates if c in df.columns]
if existing_drops:
    df = df.drop(columns=existing_drops)
    print(f"\nDropped non-useful columns: {existing_drops}")


# Handle outliers 


numeric_cols_to_check = [
    c for c in ["Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
                "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Distance(mi)"]
    if c in df.columns
]

for col in numeric_cols_to_check:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    before_out = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    print(f"Capped {before_out} outlier values in '{col}'")


print(f"\nFinal cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFinal columns:")
print(df.columns.tolist())

OUTPUT_PATH = "cleaned_accidents.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")
print("Hand this file off to Person 3 (EDA) and Person 4 (modeling).")
