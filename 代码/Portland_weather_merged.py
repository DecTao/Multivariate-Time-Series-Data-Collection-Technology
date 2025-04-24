import pandas as pd
from pathlib import Path

# Configuration
data_dir = Path("./archive")
target_city = "Portland"  # Change to other cities like "New York"
output_file = f"{target_city}_weather_merged.csv"

# Load city metadata
city_meta = pd.read_csv(data_dir / "city_attributes.csv")

# Define data files and English column names
data_config = {
    "temperature": ("temperature.csv", "temp_k"),
    "humidity": ("humidity.csv", "humidity_pct"),
    "pressure": ("pressure.csv", "pressure_hpa"),
    "wind_speed": ("wind_speed.csv", "wind_speed_mps")
}

# Initialize master dataframe
master_df = pd.DataFrame()

# Merge data files
for metric, (filename, col_name) in data_config.items():
    df = pd.read_csv(data_dir / filename, parse_dates=["datetime"])
    
    if target_city not in df.columns:
        raise ValueError(f"{target_city} not found in {filename}")
    
    city_data = df[["datetime", target_city]].copy()
    city_data.rename(columns={target_city: col_name}, inplace=True)
    
    if master_df.empty:
        master_df = city_data
    else:
        master_df = master_df.merge(city_data, on="datetime", how="outer")

# Data cleaning functions
def clean_numeric(series):
    """Clean and convert to numeric values"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("[^0-9.]", "", regex=True)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

# Apply data cleaning
numeric_cols = ["temp_k", "humidity_pct", "pressure_hpa", "wind_speed_mps"]
for col in numeric_cols:
    master_df[col] = clean_numeric(master_df[col])

# Convert Kelvin to Celsius
master_df["temp_c"] = master_df["temp_k"] - 273.15
master_df.drop(columns=["temp_k"], inplace=True)

# Time series processing
master_df = master_df.sort_values("datetime").set_index("datetime")
master_df = master_df.resample("H").asfreq()

# Handle missing values
print("\nMissing values before processing:")
print(master_df.isnull().sum())
master_df = master_df.interpolate(method="time").ffill().bfill()

# Save results
master_df.to_csv(output_file, encoding="utf-8-sig")

# Preview data
print(f"\nSample data for {target_city}:")
print(master_df.head(10))
print(f"\nData saved to: {output_file}")

# Generate statistics
stats_report = master_df.describe().T
stats_report["range"] = master_df.max() - master_df.min()
print("\nStatistical summary:")
print(stats_report[["mean", "std", "min", "50%", "max", "range"]])
