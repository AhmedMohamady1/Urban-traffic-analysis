import pandas as pd
import numpy as np
from datetime import datetime
import os

# Bounds & Accepted values
VEHICLE_COUNT_BOUNDS = (10, 4000)
SPEED_BOUNDS = (1, 120)
ACCIDENT_BOUNDS = (0, 10)
VISIBILITY_BOUNDS = (50, 15000)

ACCEPTED_ROAD_CONDITION = ['Dry', 'Wet', 'Snowy', 'Damaged']

# Path setup for reports (Kept for your project structure)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_FILE_PATH = os.path.join(BASE_DIR, "reports", "Traffic_data_quality_report.txt")

def generate_report(initial_count, final_count, fixes):
    """Write the report to disk."""
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
    try:
        with open(REPORT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Traffic Cleaning] Report saved to {REPORT_FILE_PATH}")
    except Exception as e:
        print(f"[Traffic Cleaning] Warning: Could not save report. {e}")


def clean_traffic_data(raw_df):
    """
    Clean traffic dataframe coming directly from main pipeline.
    Returns only cleaned_df.
    """

    df = raw_df.copy()
    initial_rows = len(df)
    fixes = {}

    # 1. Remove duplicates
    df.drop_duplicates(inplace=True)
    fixes['Row Duplicates Removed'] = initial_rows - len(df)

    # 2. Fix timestamps
    df['date_time_cleaned'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)
    fixes['Unfixable Timestamp Errors'] = df['date_time_cleaned'].isna().sum()
    df.drop(columns=['date_time'], inplace=True)
    df.rename(columns={'date_time_cleaned': 'date_time'}, inplace=True)

    # 3. Convert numerics
    for col in ['vehicle_count', 'avg_speed_kmh', 'accident_count', 'visibility_m', 'traffic_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Standardize Congestion Level
    df['congestion_level'] = df['congestion_level'].astype(str).str.lower().str.strip()
    mapping = {
        'low': 'Low', 'medium': 'Medium', 'high': 'High',
        'unknown': 'Unknown', 'nan': 'Unknown', 'non_standard': 'Unknown', # Updated to match "Good" file
        '4': 'Unknown', '99': 'Unknown'
    }
    df['congestion_level'] = df['congestion_level'].replace(mapping)
    fixes['Standardized Congestion Categories'] = df[df['congestion_level'] == 'Unknown'].shape[0]

    # 5. Impute Missing Categorical Data (City, Road, Area)
    
    # -- Added Missing Logic: City --
    initial_city_nulls = df['city'].isna().sum()
    df['city'].fillna(df['city'].mode()[0], inplace=True)
    fixes['Imputed Missing City'] = initial_city_nulls - df['city'].isna().sum()

    # Road Condition
    df['road_condition'].fillna(df['road_condition'].mode()[0], inplace=True)

    # Area
    df['area'].fillna(df['area'].mode()[0], inplace=True)

    # 6. Smart Imputation (Mean by Area)
    area_means = df.groupby('area')[['avg_speed_kmh', 'vehicle_count']].mean()

    for col in ['avg_speed_kmh', 'vehicle_count']:
        init_nulls = df[col].isna().sum()
        df[col] = df.apply(
            lambda row: area_means.loc[row['area'], col] if pd.isna(row[col]) and pd.notna(row['area']) else row[col],
            axis=1
        )
        fixes[f'Imputed Missing {col} by Area'] = init_nulls - df[col].isna().sum()

    # 7. Impute Remaining NaNs (Median)
    for col in ['traffic_id', 'accident_count', 'visibility_m']:
        df[col].fillna(df[col].median(), inplace=True)

    # 8. Negative Speeds Handling
    # Using the logic from the good file: explicitly count negatives before fixing
    fixes['Corrected Negative Speeds'] = (df['avg_speed_kmh'] < 0).sum()
    # Set to 1.0 (traffic jam) instead of clipping to lower bound if generic
    df['avg_speed_kmh'] = np.where(df['avg_speed_kmh'] < 0, 1.0, df['avg_speed_kmh'])

    # 9. Outlier Clipping
    df['vehicle_count'] = df['vehicle_count'].clip(*VEHICLE_COUNT_BOUNDS)
    df['avg_speed_kmh'] = df['avg_speed_kmh'].clip(*SPEED_BOUNDS)
    df['accident_count'] = df['accident_count'].clip(*ACCIDENT_BOUNDS)

    # 10. Final Cleanup
    # Drop invalid timestamps
    df.dropna(subset=['date_time'], inplace=True)

    # Remove duplicated measurements (Composite key)
    df['traffic_id'] = df['traffic_id'].astype(int)
    df.drop_duplicates(subset=['traffic_id', 'date_time', 'area'], inplace=True)

    # Convert categories for parquet efficiency
    df['congestion_level'] = pd.Categorical(df['congestion_level'], categories=['Low', 'Medium', 'High', 'Unknown'])
    df['road_condition'] = pd.Categorical(df['road_condition'], categories=ACCEPTED_ROAD_CONDITION)

    final_rows = len(df)

    # Save report locally inside this file
    generate_report(initial_rows, final_rows, fixes)

    return df