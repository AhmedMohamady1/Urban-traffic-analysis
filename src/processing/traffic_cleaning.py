import pandas as pd
import numpy as np
from datetime import datetime
import os

# Bounds & Accepted values (unchanged)
VEHICLE_COUNT_BOUNDS = (10, 4000)
SPEED_BOUNDS = (1, 120)
ACCIDENT_BOUNDS = (0, 10)
VISIBILITY_BOUNDS = (50, 15000)

ACCEPTED_ROAD_CONDITION = ['Dry', 'Wet', 'Snowy', 'Damaged']

# This points to: project_root/reports/Filename.txt
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_FILE_PATH = os.path.join(BASE_DIR, "reports", "Traffic_data_quality_report.txt")

def generate_report(initial_count, final_count, fixes):
    """Write the report to disk (inside the cleaning file only)."""
    report = f"""
# --- Data Quality Report: Traffic Dataset ---
- Initial Row Count: {initial_count}
- Final Clean Row Count: {final_count}
- Rows Removed (Duplicates/Unfixable): {initial_count - final_count}

## Cleaning Summary:
"""
    for action, count in fixes.items():
        report += f"- {action}: {count} occurrences fixed/handled.\n"

    # Write report to local file
    with open(REPORT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Traffic Cleaning] Report saved to {REPORT_FILE_PATH}")


def clean_traffic_data(raw_df):
    """
    Clean traffic dataframe coming directly from main.
    Returns only cleaned_df.
    (Report is handled internally and saved to disk.)
    """

    df = raw_df.copy()
    initial_rows = len(df)
    fixes = {}

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    fixes['Row Duplicates Removed'] = initial_rows - len(df)

    # Fix timestamps
    df['date_time_cleaned'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)
    fixes['Unfixable Timestamp Errors'] = df['date_time_cleaned'].isna().sum()
    df.drop(columns=['date_time'], inplace=True)
    df.rename(columns={'date_time_cleaned': 'date_time'}, inplace=True)

    # Convert numerics
    for col in ['vehicle_count', 'avg_speed_kmh', 'accident_count', 'visibility_m', 'traffic_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix congestion level
    df['congestion_level'] = df['congestion_level'].astype(str).str.lower().str.strip()
    mapping = {
        'low': 'Low', 'medium': 'Medium', 'high': 'High',
        'unknown': 'Unknown', 'nan': np.nan, 'non_standard': 'Unknown',
        '4': 'Unknown', '99': 'Unknown'
    }
    df['congestion_level'] = df['congestion_level'].replace(mapping)
    fixes['Standardized Congestion Categories'] = df[df['congestion_level'] == 'Unknown'].shape[0]

    # Fix road condition
    df['road_condition'].fillna(df['road_condition'].mode()[0], inplace=True)

    # Fix area
    df['area'].fillna(df['area'].mode()[0], inplace=True)

    # Mean by area
    area_means = df.groupby('area')[['avg_speed_kmh', 'vehicle_count']].mean()

    for col in ['avg_speed_kmh', 'vehicle_count']:
        init_nulls = df[col].isna().sum()
        df[col] = df.apply(
            lambda row: area_means.loc[row['area'], col] if pd.isna(row[col]) else row[col],
            axis=1
        )
        fixes[f'Imputed Missing {col} by Area'] = init_nulls - df[col].isna().sum()

    # Remaining NaNs (median)
    for col in ['traffic_id', 'accident_count', 'visibility_m']:
        df[col].fillna(df[col].median(), inplace=True)

    # Negative speeds
    fixes['Corrected Negative Speeds'] = (df['avg_speed_kmh'] < 0).sum()
    df['avg_speed_kmh'] = df['avg_speed_kmh'].clip(lower=1)

    # Outlier clipping
    df['vehicle_count'] = df['vehicle_count'].clip(*VEHICLE_COUNT_BOUNDS)
    df['avg_speed_kmh'] = df['avg_speed_kmh'].clip(*SPEED_BOUNDS)
    df['accident_count'] = df['accident_count'].clip(*ACCIDENT_BOUNDS)

    # Drop invalid timestamps
    df.dropna(subset=['date_time'], inplace=True)

    # Remove duplicated measurements
    df['traffic_id'] = df['traffic_id'].astype(int)
    df.drop_duplicates(subset=['traffic_id', 'date_time', 'area'], inplace=True)

    # Convert categories
    df['congestion_level'] = pd.Categorical(df['congestion_level'], categories=['Low', 'Medium', 'High', 'Unknown'])
    df['road_condition'] = pd.Categorical(df['road_condition'], categories=ACCEPTED_ROAD_CONDITION)

    final_rows = len(df)

    # Save report locally inside this file
    generate_report(initial_rows, final_rows, fixes)

    return df
