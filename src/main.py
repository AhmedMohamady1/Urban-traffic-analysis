import os
import boto3
import io
import subprocess
from botocore.client import Config
from datetime import datetime

from src.generator import WeatherTrafficDataGenerator 
from src.processing.weather_cleaning import clean_weather_data
from src.processing.traffic_cleaning import clean_traffic_data
from src.processing.merge_data import merge_datasets

class DataLakeManager:
    def __init__(self, endpoint_url, access_key, secret_key):
        self.endpoint = endpoint_url
        
        # S3 Client (Boto3)
        self.s3 = boto3.client('s3',
                               endpoint_url=endpoint_url,
                               aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key,
                               config=Config(signature_version='s3v4'),
                               region_name='us-east-1')
        
        # Setup buckets and log
        self._ensure_buckets_exist(['bronze', 'silver', 'logs'])
        self.log_message("Data Lake Manager Initialized (Boto3 Only).")

    def _ensure_buckets_exist(self, buckets):
        # List existing buckets
        existing = [b['Name'] for b in self.s3.list_buckets().get('Buckets', [])]
        for b in buckets:
            if b not in existing:
                print(f"⚠️ Creating missing bucket: {b}")
                self.s3.create_bucket(Bucket=b)

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)
        
        # Simple logging to S3
        log_filename = f"datalake_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
        try:
            obj = self.s3.get_object(Bucket='logs', Key=log_filename)
            current_content = obj['Body'].read().decode('utf-8')
        except:
            current_content = ""
            
        new_content = current_content + full_msg + "\n"
        self.s3.put_object(Bucket='logs', Key=log_filename, Body=new_content.encode('utf-8'))

    def upload_data(self, df, layer, filename):
        self.log_message(f"⬆️ Uploading data to {layer}/{filename}...")
        buffer = io.BytesIO()
        
        # Check filename to determine format
        if filename.endswith('.parquet'):
            # Requires: pip install pyarrow
            df.to_parquet(buffer, index=False)
        else:
            # Default to CSV
            df.to_csv(buffer, index=False)
            
        buffer.seek(0)
        self.s3.put_object(Bucket=layer, Key=filename, Body=buffer)
        self.log_message("✅ Upload successful.")

    def backup_to_hdfs(self, layer, filename, subfolder=None):
            """
            Copy to HDFS. Optional 'subfolder' argument allows organizing
            by data type (Weather vs Traffic) as required by Phase 3.
            """
            s3_source = f"s3a://{layer}/{filename}"
            
            # If a subfolder is defined (e.g., 'weather'), put it there. 
            # Otherwise, keep generic structure /datalake/silver/
            if subfolder:
                hdfs_dest_dir = f"/datalake/{subfolder}/"
            else:
                hdfs_dest_dir = f"/datalake/{layer}/"
            
            self.log_message(f"🔄 Backing up {filename} to HDFS folder: {hdfs_dest_dir}...")

            try:
                # 1. Ensure the destination directory exists
                subprocess.run(
                    ["docker", "exec", "hadoop_namenode", "hdfs", "dfs", "-mkdir", "-p", hdfs_dest_dir],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )

                # 2. Copy the file
                subprocess.run(
                    ["docker", "exec", "hadoop_namenode", "hdfs", "dfs", "-cp", "-f", s3_source, hdfs_dest_dir],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                self.log_message("✅ HDFS Backup successful.")
            except subprocess.CalledProcessError as e:
                self.log_message(f"❌ HDFS Backup Failed: {e}")

    def run_pipeline(self, num_rows=5000):
        self.log_message(" STARTING FULL PIPELINE (Generate -> Bronze -> Silver)...")

        # --- STEP 1: Generate Raw Data ---
        self.log_message("1️⃣ Generating Weather & Traffic Data...")
        gen = WeatherTrafficDataGenerator()
        weather_df, traffic_df = gen.generate_all(rows=num_rows)

        # --- STEP 2: Upload Raw Data to Bronze ---
        self.upload_data(weather_df, 'bronze', 'raw_weather_data.csv')
        # self.backup_to_hdfs('bronze', 'raw_weather_data.csv') 

        self.upload_data(traffic_df, 'bronze', 'raw_traffic_data.csv')

        # --- STEP 3: Clean Weather Data and Upload to Silver ---
        self.log_message("2️⃣ Cleaning Weather Data...")
        clean_weather_df = clean_weather_data(weather_df)
        self.upload_data(clean_weather_df, 'silver', 'cleaned_weather_data.parquet')
        
        # BACKUP TO HDFS
        self.backup_to_hdfs('silver', 'cleaned_weather_data.parquet', subfolder='weather')
        
        # --- STEP 4: Clean Traffic Data and Upload to Silver ---
        self.log_message("3️⃣ Cleaning Traffic Data...")
        clean_traffic_df = clean_traffic_data(traffic_df)
        self.upload_data(clean_traffic_df, 'silver', 'cleaned_traffic_data.parquet')

        # BACKUP TO HDFS
        self.backup_to_hdfs('silver', 'cleaned_traffic_data.parquet', subfolder='traffic')
        
        # --- STEP 5: Merge Cleaned Weather & Traffic ---
        self.log_message("4️⃣ Merging Cleaned Weather & Traffic Data...")
        merged_df = merge_datasets(clean_weather_df, clean_traffic_df) 
        if merged_df is not None:
            self.upload_data(merged_df, 'silver', 'merged_dataset.parquet')
            
            self.log_message("✅ Merge complete and uploaded to Silver.")

        self.log_message("✅ Silver Layer Ready (Weather + Traffic + Merged).")


if __name__ == "__main__":
    MINIO_HOST = os.getenv("MINIO_ENDPOINT", "localhost")
    
    lake = DataLakeManager(f"http://{MINIO_HOST}:9000", "admin", "password123")
    lake.run_pipeline(num_rows=5000)