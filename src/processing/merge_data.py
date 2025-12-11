import pandas as pd

def merge_datasets(weather_df, traffic_df):
    """
    Merge cleaned Weather & Traffic datasets passed directly as DataFrames.
    Returns the merged DataFrame only.
    """
    # Ensure key columns are string type
    weather_df['city'] = weather_df['city'].astype(str)
    traffic_df['city'] = traffic_df['city'].astype(str)
    
    # Create hourly join key
    weather_df['date_hour'] = weather_df['date_time'].dt.floor('H')
    traffic_df['date_hour'] = traffic_df['date_time'].dt.floor('H')
    
    # Aggregate traffic data to the hour
    traffic_agg = traffic_df.groupby(['date_hour', 'city']).agg(
        avg_speed_kmh=('avg_speed_kmh', 'mean'),
        visibility_m_traffic=('visibility_m', 'mean'),
        vehicle_count=('vehicle_count', 'sum'),
        accident_count=('accident_count', 'sum'),
        area=('area', 'first'),
        congestion_level=('congestion_level', lambda x: x.mode()[0] if not x.mode().empty else 'Medium'),
        road_condition=('road_condition', lambda x: x.mode()[0] if not x.mode().empty else 'Dry')
    ).reset_index()
    
    # Merge datasets
    merged_df = pd.merge(weather_df, traffic_agg, on=['date_hour', 'city'], how='inner', suffixes=('_weather','_traffic'))
    
    # Cleanup columns
    merged_df.drop(columns=['date_hour','traffic_id','weather_id','visibility_m_traffic'], inplace=True, errors='ignore')
    merged_df.rename(columns={'visibility_m_weather':'visibility_m'}, inplace=True)
    
    return merged_df
