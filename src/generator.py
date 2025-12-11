import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


class WeatherTrafficDataGenerator:

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.CITY = "London"
        self.START_DATE = datetime(2024, 1, 1)

        # Error & noise probabilities
        self.P_NULL = 0.02
        self.P_OUTLIER = 0.01
        self.P_DUPLICATE = 0.03
        self.P_FORMAT = 0.02
        self.P_GARBAGE = 0.015

    # -----------------------------
    # Helper Functions
    # -----------------------------

    def introduce_mess(self, value, p_null, p_outlier=0, outlier_val=None,
                       p_format=0, format_func=None):

        missing_value = None if isinstance(value, (str, datetime)) else np.nan

        if random.random() < p_null:
            return missing_value
        if p_outlier > 0 and random.random() < p_outlier:
            return outlier_val
        if p_format > 0 and random.random() < p_format and format_func:
            return format_func(value)

        return value

    def random_datetime(self):
        return self.START_DATE + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

    # -----------------------------
    # Weather Data Generator
    # -----------------------------
    def generate_weather_dataset(self, rows=5000):

        weather_data = []
        weather_ids = list(range(5001, 5001 + rows))

        for i in range(rows):
            dt = self.random_datetime()

            if dt.month in [12, 1, 2]: season = 'Winter'
            elif dt.month in [3, 4, 5]: season = 'Spring'
            elif dt.month in [6, 7, 8]: season = 'Summer'
            else: season = 'Autumn'

            temp = random.uniform(5, 25)
            humidity = random.randint(50, 90)
            rain = max(0, random.gauss(5, 10))
            wind = max(0, random.gauss(30, 15))
            visibility = random.randint(1000, 10000)
            pressure = random.uniform(980, 1030)
            condition = random.choice(['Clear', 'Rain', 'Fog', 'Storm', 'Snow'])

            dt_mess = dt.strftime('%Y-%m-%d %H:%M:%S')

            # Messiness
            temp_mess = self.introduce_mess(temp, self.P_NULL,
                                            p_outlier=self.P_OUTLIER,
                                            outlier_val=random.choice([-15, 45]))

            humidity_mess = self.introduce_mess(humidity, self.P_NULL,
                                                p_outlier=self.P_OUTLIER,
                                                outlier_val=120)

            rain_mess = self.introduce_mess(rain, self.P_NULL,
                                            p_outlier=self.P_OUTLIER,
                                            outlier_val=80.0)

            wind_mess = self.introduce_mess(wind, self.P_NULL,
                                            p_outlier=self.P_OUTLIER,
                                            outlier_val=120.0)

            visibility_mess = self.introduce_mess(
                visibility, self.P_NULL,
                p_outlier=self.P_OUTLIER,
                outlier_val=20000,
                p_format=self.P_FORMAT,
                format_func=lambda v: "LOW"
            )

            weather_id_mess = self.introduce_mess(
                weather_ids[i], self.P_NULL,
                p_outlier=self.P_OUTLIER,
                outlier_val=weather_ids[random.randint(0, rows - 1)]
            )

            weather_data.append([
                weather_id_mess, dt_mess, self.CITY, season,
                temp_mess, humidity_mess, rain_mess, wind_mess,
                visibility_mess, condition, pressure
            ])

        df = pd.DataFrame(weather_data, columns=[
            'weather_id', 'date_time', 'city', 'season', 'temperature_c',
            'humidity', 'rain_mm', 'wind_speed_kmh', 'visibility_m',
            'weather_condition', 'air_pressure_hpa'
        ])

        # Duplicates
        duplicate_rows = df.sample(frac=self.P_DUPLICATE, random_state=42)
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    # -----------------------------
    # Traffic Data Generator
    # -----------------------------
    def generate_traffic_dataset(self, rows=5000, weather_df=None):

        if weather_df is None:
            raise ValueError("weather_df is required to synchronize timestamps.")

        traffic_data = []
        traffic_ids = list(range(9001, 9001 + rows))
        areas = ['Camden', 'Chelsea', 'Islington', 'Southwark',
                 'Kensington', 'Westminster', 'Greenwich']

        for i in range(rows):

            area = random.choice(areas)
            vehicle_count = random.randint(100, 3000)
            avg_speed = random.uniform(20, 80)
            accident_count = random.randint(0, 5)
            congestion = random.choice(['Low', 'Medium', 'High'])
            road_condition = random.choice(['Dry', 'Wet', 'Snowy', 'Damaged'])
            visibility = random.randint(1000, 10000)

            # use same datetime as weather
            dt = datetime.strptime(weather_df.loc[i, 'date_time'], "%Y-%m-%d %H:%M:%S")
            dt_mess = dt.strftime('%Y-%m-%d %H:%M:%S')

            vehicle_count_mess = self.introduce_mess(vehicle_count, self.P_NULL,
                                                     p_outlier=self.P_OUTLIER,
                                                     outlier_val=5000)

            avg_speed_mess = self.introduce_mess(avg_speed, self.P_NULL,
                                                 p_outlier=self.P_OUTLIER,
                                                 outlier_val=random.choice([-5.0, 100.0]))

            accident_count_mess = self.introduce_mess(accident_count, self.P_NULL,
                                                      p_outlier=self.P_OUTLIER,
                                                      outlier_val=15)

            congestion_mess = self.introduce_mess(
                congestion, self.P_NULL,
                p_format=self.P_FORMAT,
                format_func=lambda c: random.choice(['NON_STANDARD', '4'])
            )

            road_mess = self.introduce_mess(road_condition, self.P_NULL)
            visibility_mess = self.introduce_mess(visibility, self.P_NULL,
                                                  p_outlier=self.P_OUTLIER,
                                                  outlier_val=20000)

            traffic_id_mess = self.introduce_mess(
                traffic_ids[i], self.P_NULL,
                p_outlier=self.P_OUTLIER,
                outlier_val=traffic_ids[random.randint(0, rows - 1)]
            )

            traffic_data.append([
                traffic_id_mess, dt_mess, self.CITY, area, vehicle_count_mess,
                avg_speed_mess, accident_count_mess, congestion_mess,
                road_mess, visibility_mess
            ])

        df = pd.DataFrame(traffic_data, columns=[
            'traffic_id', 'date_time', 'city', 'area',
            'vehicle_count', 'avg_speed_kmh', 'accident_count',
            'congestion_level', 'road_condition', 'visibility_m'
        ])

        duplicate_rows = df.sample(frac=self.P_DUPLICATE, random_state=42)
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    # -----------------------------
    # Combined Runner
    # -----------------------------
    def generate_all(self, rows=5000, save=False):
        weather_df = self.generate_weather_dataset(rows)
        traffic_df = self.generate_traffic_dataset(rows, weather_df)

        if save:
            weather_df.to_csv("raw_weather_data.csv", index=False)
            traffic_df.to_csv("raw_traffic_data.csv", index=False)

        return weather_df, traffic_df
