import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- Bounds & Accepted Values ---
TEMP_BOUNDS = (-10, 40)
HUMIDITY_BOUNDS = (10, 100)
RAIN_BOUNDS = (0, 100)
WIND_BOUNDS = (0, 150)
PRESSURE_BOUNDS = (950, 1050)

ACCEPTED_CONDITIONS = ['Clear', 'Rain', 'Fog', 'Storm', 'Snow']
ACCEPTED_SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']

# Path setup for reports (Kept for your project structure)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_FILE_PATH = os.path.join(BASE_DIR, "reports", "Weather_data_quality_report.txt")

def generate_report(initial_count, final_count, fixes):
    """Write the weather report locally."""
    report = f"""
# --- Data Quality Report: Weather Dataset ---
- Initial Row Count: {initial_count}
- Final Clean Row Count: {final_count}
- Rows Removed (Duplicates/Unfixable): {initial_count - final_count}

## Cleaning Summary:
"""
    for action, count in fixes.items():
        report += f"- {action}: {count} occurrences fixed/handled.\n"

    try:
        with open(REPORT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Weather Cleaning] Report saved to {REPORT_FILE_PATH}")
    except Exception as e:
        print(f"[Weather Cleaning] Warning: Could not save report. {e}")


def clean_weather_data(raw_df):
    """
    Clean weather dataframe coming directly from main pipeline.
    Returns only cleaned_df.
    """

    df = raw_df.copy()
    initial_rows = len(df)
    fixes = {}

    # 1. Remove row duplicates
    df.drop_duplicates(inplace=True)
    fixes['Row Duplicates Removed'] = initial_rows - len(df)

    # 2. Fix timestamps
    # Attempt to convert to datetime, coercing unfixable garbage/strings to NaT
    df['date_time_cleaned'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)
    
    # Track unfixable errors (difference between nulls before and after)
    fixes['Unfixable Timestamp Errors'] = df['date_time_cleaned'].isna().sum() - df['date_time'].isna().sum()

    df.drop(columns=['date_time'], inplace=True)
    df.rename(columns={'date_time_cleaned': 'date_time'}, inplace=True)

    # 3. Standardizing Categories & Simple Imputation
    
    # City & Condition Mode Imputation
    initial_city_nulls = df['city'].isna().sum()
    df['city'].fillna(df['city'].mode()[0], inplace=True)
    
    initial_condition_nulls = df['weather_condition'].isna().sum()
    df['weather_condition'].fillna(df['weather_condition'].mode()[0], inplace=True)
    
    fixes['Imputed Missing City/Condition'] = (initial_city_nulls - df['city'].isna().sum()) + (initial_condition_nulls - df['weather_condition'].isna().sum())

    # --- FIX: Impute missing SEASON before using it for smart imputation ---
    initial_season_nulls = df['season'].isna().sum()
    df['season'].fillna(df['season'].mode()[0], inplace=True)
    fixes['Imputed Missing Season'] = initial_season_nulls - df['season'].isna().sum()

    # 4. Numeric Conversion & Visibility Fix
    # Fix Visibility: Remove non-numeric strings and convert to float
    initial_visibility_non_numeric = df[pd.to_numeric(df['visibility_m'], errors='coerce').isna()].shape[0]
    df['visibility_m'] = pd.to_numeric(df['visibility_m'], errors='coerce')
    fixes['Fixed Non-Numeric Visibility'] = initial_visibility_non_numeric - df['visibility_m'].isna().sum()

    # Convert other numeric columns
    for col in ['temperature_c', 'humidity', 'air_pressure_hpa', 'rain_mm', 'wind_speed_kmh', 'weather_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Smart Imputation (Seasonal Means)
    # Calculate seasonal averages for imputation
    seasonal_means = df.groupby('season')[['temperature_c', 'humidity', 'air_pressure_hpa']].mean()

    for col in ['temperature_c', 'humidity', 'air_pressure_hpa']:
        initial_nulls = df[col].isna().sum()
        df[col] = df.apply(
            lambda row: seasonal_means.loc[row['season'], col] if pd.isna(row[col]) else row[col],
            axis=1
        )
        fixes[f'Imputed {col} by Season'] = initial_nulls - df[col].isna().sum()

    # Impute remaining simple NaNs (rain/wind/visibility/IDs)
    for col in ['rain_mm', 'wind_speed_kmh', 'weather_id']:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Visibility median fill
    df['visibility_m'].fillna(df['visibility_m'].median(), inplace=True)

    # 6. Outlier Clipping
    fixes['Corrected Temp Outliers'] = ((df['temperature_c'] < TEMP_BOUNDS[0]) | (df['temperature_c'] > TEMP_BOUNDS[1])).sum()
    df['temperature_c'] = df['temperature_c'].clip(*TEMP_BOUNDS)

    fixes['Corrected Humidity Outliers'] = ((df['humidity'] < HUMIDITY_BOUNDS[0]) | (df['humidity'] > HUMIDITY_BOUNDS[1])).sum()
    df['humidity'] = df['humidity'].clip(*HUMIDITY_BOUNDS)

    fixes['Corrected Rain Outliers'] = (df['rain_mm'] > RAIN_BOUNDS[1]).sum()
    df['rain_mm'] = df['rain_mm'].clip(*RAIN_BOUNDS)

    fixes['Corrected Wind Outliers'] = (df['wind_speed_kmh'] > WIND_BOUNDS[1]).sum()
    df['wind_speed_kmh'] = df['wind_speed_kmh'].clip(*WIND_BOUNDS)

    # 7. Final Cleanup
    # Drop invalid timestamps
    df.dropna(subset=['date_time'], inplace=True)

    # Deduplicate ID
    df['weather_id'] = df['weather_id'].astype(int)
    df.drop_duplicates(subset=['weather_id', 'date_time', 'city'], inplace=True)

    # Convert categories
    df['season'] = pd.Categorical(df['season'], categories=ACCEPTED_SEASONS)
    df['weather_condition'] = pd.Categorical(df['weather_condition'], categories=ACCEPTED_CONDITIONS)

    final_rows = len(df)

    # Save local report
    generate_report(initial_rows, final_rows, fixes)

    return df