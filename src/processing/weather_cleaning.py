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

# This points to: project_root/reports/Filename.txt
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_FILE_PATH = os.path.join(BASE_DIR, "reports", "Weather_data_quality_report.txt")

def generate_report(initial_count, final_count, fixes):
    """Write the weather report locally (not S3)."""
    report = f"""
# --- Data Quality Report: Weather Dataset ---
- Initial Row Count: {initial_count}
- Final Clean Row Count: {final_count}
- Rows Removed (Duplicates/Unfixable): {initial_count - final_count}

## Cleaning Summary:
"""
    for action, count in fixes.items():
        report += f"- {action}: {count} occurrences fixed/handled.\n"

    with open(REPORT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Weather Cleaning] Report saved to {REPORT_FILE_PATH}")


def clean_weather_data(raw_df):
    """
    Clean weather dataframe coming directly from main.
    Returns only cleaned_df.
    (Report is handled internally and saved locally.)
    """

    df = raw_df.copy()
    initial_rows = len(df)
    fixes = {}

    # --- Remove row duplicates ---
    df.drop_duplicates(inplace=True)
    fixes['Row Duplicates Removed'] = initial_rows - len(df)

    # --- Fix timestamps ---
    df['date_time_cleaned'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)
    fixes['Unfixable Timestamp Errors'] = df['date_time_cleaned'].isna().sum()

    df.drop(columns=['date_time'], inplace=True)
    df.rename(columns={'date_time_cleaned': 'date_time'}, inplace=True)

    # --- Convert non-numeric visibility + numeric fields ---
    df['visibility_m'] = pd.to_numeric(df['visibility_m'], errors='coerce')

    for col in ['temperature_c', 'humidity', 'air_pressure_hpa', 'rain_mm',
                'wind_speed_kmh', 'weather_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Simple category imputations ---
    df['city'].fillna(df['city'].mode()[0], inplace=True)
    df['weather_condition'].fillna(df['weather_condition'].mode()[0], inplace=True)

    # --- Seasonal smart imputation ---
    seasonal_means = df.groupby('season')[['temperature_c', 'humidity', 'air_pressure_hpa']].mean()

    for col in ['temperature_c', 'humidity', 'air_pressure_hpa']:
        before = df[col].isna().sum()
        df[col] = df.apply(
            lambda r: seasonal_means.loc[r['season'], col]
            if pd.isna(r[col]) and pd.notna(r['season']) else r[col],
            axis=1
        )
        fixes[f'Imputed {col} by Season'] = before - df[col].isna().sum()

    # Remaining NaNs (median)
    for col in ['rain_mm', 'wind_speed_kmh', 'visibility_m', 'weather_id']:
        df[col].fillna(df[col].median(), inplace=True)

    # --- Outlier clipping ---
    fixes['Corrected Temp Outliers'] = ((df['temperature_c'] < TEMP_BOUNDS[0]) |
                                        (df['temperature_c'] > TEMP_BOUNDS[1])).sum()
    df['temperature_c'] = df['temperature_c'].clip(*TEMP_BOUNDS)

    fixes['Corrected Humidity Outliers'] = ((df['humidity'] < HUMIDITY_BOUNDS[0]) |
                                            (df['humidity'] > HUMIDITY_BOUNDS[1])).sum()
    df['humidity'] = df['humidity'].clip(*HUMIDITY_BOUNDS)

    fixes['Corrected Rain Outliers'] = (df['rain_mm'] > RAIN_BOUNDS[1]).sum()
    df['rain_mm'] = df['rain_mm'].clip(*RAIN_BOUNDS)

    fixes['Corrected Wind Outliers'] = (df['wind_speed_kmh'] > WIND_BOUNDS[1]).sum()
    df['wind_speed_kmh'] = df['wind_speed_kmh'].clip(*WIND_BOUNDS)

    # --- Drop invalid timestamps ---
    df.dropna(subset=['date_time'], inplace=True)

    # --- ID cleaning & dedupe ---
    df['weather_id'] = df['weather_id'].astype(int)
    df.drop_duplicates(subset=['weather_id', 'date_time', 'city'], inplace=True)

    # --- Convert categories ---
    df['season'] = pd.Categorical(df['season'], categories=ACCEPTED_SEASONS)
    df['weather_condition'] = pd.Categorical(df['weather_condition'], categories=ACCEPTED_CONDITIONS)

    final_rows = len(df)

    # Save local report
    generate_report(initial_rows, final_rows, fixes)

    return df
