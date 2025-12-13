import pandas as pd

def merge_datasets(weather_df, traffic_df):
    """
    Merge cleaned Weather & Traffic datasets passed directly as DataFrames.
    Returns the merged DataFrame only.
    """
    
    print(f"Merging data... Weather rows: {len(weather_df)}, Traffic rows: {len(traffic_df)}")

    # 1. Ensure key columns are string type
    weather_df['city'] = weather_df['city'].astype(str)
    traffic_df['city'] = traffic_df['city'].astype(str)
    
    # 2. Create hourly join key (Common Key)
    # Using 'h' instead of 'H' to avoid FutureWarning in recent pandas versions
    weather_df['date_hour'] = weather_df['date_time'].dt.floor('h')
    traffic_df['date_hour'] = traffic_df['date_time'].dt.floor('h')
    
    # 3. Aggregate traffic data to the hour
    # Traffic data is high frequency/variable, so we aggregate to avoid duplicate keys during join
    traffic_agg = traffic_df.groupby(['date_hour', 'city']).agg(
        # Mean for continuous variables
        avg_speed_kmh=('avg_speed_kmh', 'mean'),
        visibility_m_traffic=('visibility_m', 'mean'),
        
        # Sum for counts
        vehicle_count=('vehicle_count', 'sum'),
        accident_count=('accident_count', 'sum'),
        
        # Mode/First for categorical
        area=('area', 'first'),
        congestion_level=('congestion_level', lambda x: x.mode()[0] if not x.mode().empty else 'Medium'),
        road_condition=('road_condition', lambda x: x.mode()[0] if not x.mode().empty else 'Dry')
    ).reset_index()
    
    print(f"Traffic aggregated to {len(traffic_agg)} hourly slots.")

    # 4. Perform the Merge (Inner Join)
    merged_df = pd.merge(
        left=weather_df,
        right=traffic_agg,
        on=['date_hour', 'city'],
        how='inner',
        suffixes=('_weather', '_traffic')
    )
    
    # 5. Final Cleanup
    # Drop temporary keys and IDs that are no longer needed
    merged_df.drop(columns=['date_hour', 'traffic_id', 'weather_id', 'visibility_m_traffic'], inplace=True, errors='ignore')
    
    # Rename weather visibility to be the primary visibility column
    merged_df.rename(columns={'visibility_m_weather': 'visibility_m'}, inplace=True)
    
    print(f"Merge complete. Final dataset size: {len(merged_df)} rows.")
    
    return merged_df