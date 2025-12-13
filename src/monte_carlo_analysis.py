import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import boto3
import io
import os
from botocore.client import Config

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. SETUP S3 CONNECTION
# ==========================================
MINIO_HOST = os.getenv("MINIO_ENDPOINT", "localhost")
ACCESS_KEY = "admin"
SECRET_KEY = "password123"

s3 = boto3.client('s3',
                  endpoint_url=f"http://{MINIO_HOST}:9000",
                  aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY,
                  config=Config(signature_version='s3v4'),
                  region_name='us-east-1')

print("⬇️  Reading data from Silver bucket...")

# ==========================================
# 2. READ FROM MINIO (Silver)
# ==========================================
try:
    obj = s3.get_object(Bucket='silver', Key='merged_dataset.parquet')
    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    print(f"✅ Data loaded: {len(df)} rows")
except Exception as e:
    print(f"❌ Error reading from S3: {e}")
    exit(1)

SCENARIOS = {
    'Heavy Rain': df['rain_mm'] > 20,
    'Temperature Extremes': (df['temperature_c'] < 0) | (df['temperature_c'] > 30),
    'High Humidity': df['humidity'] > 85,
    'Low Visibility': df['visibility_m'] < 2000,
    'Strong Winds': df['wind_speed_kmh'] > 50
}

def calculate_probabilities(data_subset):
    if len(data_subset) == 0:
        return {'traffic_jam_prob': 0.0, 'accident_prob': 0.0}
    
    traffic_jam_prob = (data_subset['congestion_level'] == 'High').mean()
    accident_prob = (data_subset['accident_count'] > 0).mean()
    
    return {
        'traffic_jam_prob': float(traffic_jam_prob),
        'accident_prob': float(accident_prob)
    }
    
def run_monte_carlo_simulation(data, condition, n_simulations=10000, sample_size=100):
    scenario_data = data[condition]
    
    if len(scenario_data) == 0:
        print(f"No data for this scenario, returning zeros...")
        return np.zeros(n_simulations, dtype=float), np.zeros(n_simulations, dtype=float)
    
    jam_probs = []
    accident_probs = []
    
    for i in range(n_simulations):
        sample = scenario_data.sample(n=sample_size, replace=True, random_state=i)
        
        jam_prob = (sample['congestion_level'] == 'High').mean()
        accident_prob = (sample['accident_count'] > 0).mean()
        
        jam_probs.append(float(jam_prob))
        accident_probs.append(float(accident_prob))
        
    return np.array(jam_probs, dtype=float), np.array(accident_probs, dtype=float)

print("🔄 Running Monte Carlo Simulations...")

# Dictionary to store all simulation results
simulation_results = {}
for scenario, condition in SCENARIOS.items():
    actual_probs = calculate_probabilities(df[condition])
    
    jam_monte_carlo_probs, accident_monte_carlo_probs = run_monte_carlo_simulation(
        data=df,
        condition=condition,
        n_simulations=10000,
        sample_size=100
    )
    
    # --- CHANGED: Split Tuples into explicit Lower/Upper keys ---
    simulation_results[scenario] = {
        'actual_traffic_jam_prob': float(actual_probs['traffic_jam_prob']),
        'monte_carlo_traffic_jam_mean': float(jam_monte_carlo_probs.mean()),
        'monte_carlo_traffic_jam_std': float(jam_monte_carlo_probs.std()),
        
        # Flattened CI for Traffic Jam
        'monte_carlo_traffic_jam_ci_95_lower': float(np.percentile(jam_monte_carlo_probs, 2.5)),
        'monte_carlo_traffic_jam_ci_95_upper': float(np.percentile(jam_monte_carlo_probs, 97.5)),

        'actual_accident_risk_prob': float(actual_probs['accident_prob']),
        'monte_carlo_accident_risk_mean': float(accident_monte_carlo_probs.mean()),
        'monte_carlo_accident_risk_std': float(accident_monte_carlo_probs.std()),

        # Flattened CI for Accident Risk
        'monte_carlo_accident_risk_ci_95_lower': float(np.percentile(accident_monte_carlo_probs, 2.5)),
        'monte_carlo_accident_risk_ci_95_upper': float(np.percentile(accident_monte_carlo_probs, 97.5)),

        # Store full array for plotting only (removed before CSV save)
        'plot_data': jam_monte_carlo_probs 
    }

# Prepare DataFrame for CSV
results_df = pd.DataFrame(simulation_results).T

# Drop the raw plotting data before saving to CSV
results_df_clean = results_df.drop(columns=['plot_data'])

# --- NOW SAFE TO CONVERT: No tuples exist in the dataframe ---
results_df_clean = results_df_clean.astype(float)
results_df_clean.index.name = 'Scenario'

# ==========================================
# 3. WRITE CSV TO MINIO (Gold/monte_carlo)
# ==========================================
print("⬆️  Uploading results CSV to Gold bucket (monte_carlo folder)...")
csv_buffer = io.BytesIO()
results_df_clean.to_csv(csv_buffer)
csv_buffer.seek(0)
s3.put_object(Bucket='gold', Key='monte_carlo/simulation_results.csv', Body=csv_buffer)

# ==========================================
# 4. PLOTTING
# ==========================================
print("📊 Generating Plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

scenario_thresholds = {
    'Heavy Rain': 'Rain > 20mm/hr',
    'Temperature Extremes': 'Temp < 0°C or > 30°C',
    'High Humidity': 'Humidity > 85%',
    'Low Visibility': 'Visibility < 2000m',
    'Strong Winds': 'Wind > 50 km/h'
}

for idx, (scenario, results) in enumerate(simulation_results.items()):
    ax = axes[idx]
    
    # Retrieve the raw data we saved specifically for plotting
    plot_data = results['plot_data']
    
    ax.hist(plot_data, bins=50, alpha=0.7, 
            density=True, color='skyblue', edgecolor='black')
    
    ax.axvline(x=results['actual_traffic_jam_prob'], color='red', 
               linestyle='--', linewidth=2, label=f'Actual: {results["actual_traffic_jam_prob"]:.4f}')
    
    ax.axvline(x=results['monte_carlo_traffic_jam_mean'], color='green', 
               linestyle='--', linewidth=2, label=f'MC Mean: {results["monte_carlo_traffic_jam_mean"]:.4f}')
    
    sns.kdeplot(plot_data, ax=ax, color='darkblue', linewidth=2)
    
    ax.set_title(f'{scenario}\n{scenario_thresholds[scenario]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Congestion Probability', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- CHANGED: Access separate keys instead of tuple indexing ---
    stats_text = f"MC Mean: {results['monte_carlo_traffic_jam_mean']:.4f}\n"
    stats_text += f"MC Std: {results['monte_carlo_traffic_jam_std']:.4f}\n"
    stats_text += f"95% CI: [{results['monte_carlo_traffic_jam_ci_95_lower']:.4f}, {results['monte_carlo_traffic_jam_ci_95_upper']:.4f}]"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].axis('off')
plt.suptitle('Monte Carlo Simulation: Congestion Probability Distribution\n'
             'Under Different Weather Scenarios', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# ==========================================
# 5. WRITE PLOT TO MINIO (Gold/monte_carlo)
# ==========================================
print("⬆️  Uploading plot to Gold bucket (monte_carlo folder)...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='monte_carlo/congestion_probability_distribution.png', Body=img_buffer)

print("✅ Analysis Complete. Check the 'gold/monte_carlo' folder.")
plt.close() # Close memory