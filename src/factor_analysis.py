# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import warnings
import boto3
import io
import os
from botocore.client import Config

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

# Create a copy for preprocessing
df_processed = df.copy()

print("ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

# Encode congestion_level
congestion_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Unknown': 1  # Treat Unknown as Medium
}
df_processed['congestion_level_encoded'] = df_processed['congestion_level'].map(congestion_mapping)

# Also create a binary accident indicator
df_processed['accident_indicator'] = (df_processed['accident_count'] > 0).astype(int)

# Display encoding results
print("Congestion level mapping:")
for level, code in congestion_mapping.items():
    count = (df_processed['congestion_level'] == level).sum()
    print(f"  {level} → {code} (Count: {count})")
    
# Select relevant weather and traffic variables for factor analysis
weather_vars = [
    'temperature_c',
    'humidity',
    'rain_mm',
    'wind_speed_kmh',
    'visibility_m',
    'air_pressure_hpa'
]

traffic_vars = [
    'avg_speed_kmh',
    'vehicle_count',
    'accident_count',
    'congestion_level_encoded'
]

# Create a subset dataframe for factor analysis
fa_vars = weather_vars + traffic_vars
df_fa = df_processed[fa_vars].copy()

# Handle NaNs before FA (drop rows with any NaNs in selected columns)
df_fa.dropna(inplace=True)
# Align original processed df with dropped rows if necessary
df_processed = df_processed.loc[df_fa.index] 

print("\n" + "=" * 80)
print("ASSESSING SUITABILITY OF DATA FOR FACTOR ANALYSIS")
print("=" * 80)

# 1. Check for normality (visual inspection)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for i, col in enumerate(fa_vars):
    ax = axes[i]
    df_fa[col].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution of {col}', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Distribution of Variables for Factor Analysis', y=1.02, fontsize=16)

# Save Plot 1 to S3
print("⬆️  Uploading Distribution Plot to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/variable_distributions.png', Body=img_buffer)
plt.close()

# 2. Check correlation matrix
print("\n1. Correlation Matrix Analysis:")

# Calculate correlation matrix
corr_matrix = df_fa.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Selected Variables', fontsize=16)
plt.tight_layout()

# Save Plot 2 to S3
print("⬆️  Uploading Correlation Matrix to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/correlation_matrix.png', Body=img_buffer)
plt.close()

# Identify strong correlations (|r| > 0.3)
print("\nStrong Correlations (|r| > 0.3):")
strong_corrs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.3:
            strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                               corr_matrix.iloc[i, j]))
            
for var1, var2, corr in strong_corrs[:10]:  # Show top 10
    print(f"  {var1} ↔ {var2}: {corr:.3f}")

# 3. Bartlett's Test of Sphericity
print("\n2. Bartlett's Test of Sphericity:")
chi_square, p_value = calculate_bartlett_sphericity(df_fa)
print(f"  Chi-square: {chi_square:.2f}")
print(f"  p-value: {p_value:.6f}")

# 4. Kaiser-Meyer-Olkin (KMO) Test
print("\n3. Kaiser-Meyer-Olkin (KMO) Test:")
kmo_all, kmo_model = calculate_kmo(df_fa)
print(f"  Overall KMO: {kmo_model:.3f}")

# Individual KMO values
print("\n  Individual KMO values:")
kmo_df = pd.DataFrame({'Variable': fa_vars, 'KMO': kmo_all})
print(kmo_df.round(3).to_string(index=False))

print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF FACTORS")
print("=" * 80)

# Method 1: Kaiser Criterion (Eigenvalues > 1)
fa = FactorAnalyzer(rotation=None, n_factors=len(fa_vars))
fa.fit(df_fa)

# Get eigenvalues
ev, v = fa.get_eigenvalues()
print("1. Eigenvalues:")
for i, val in enumerate(ev, 1):
    print(f"  Factor {i}: {val:.3f}")

# Method 2: Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ev) + 1), ev, marker='o', linewidth=2, markersize=8)
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue = 1)')
plt.title('Scree Plot for Factor Analysis', fontsize=16)
plt.xlabel('Factor Number', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save Plot 3 to S3
print("⬆️  Uploading Scree Plot to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/scree_plot.png', Body=img_buffer)
plt.close()


print("\n" + "=" * 80)
print("PERFORMING FACTOR ANALYSIS")
print("=" * 80)

# We'll use 3 factors
n_factors = 3

# Perform Factor Analysis with Varimax rotation
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(df_fa)

# Get factor loadings
loadings = fa.loadings_

# Calculate Factor Scores
factor_scores = fa.transform(df_fa)

# Create factor loadings dataframe
loadings_df = pd.DataFrame(
    loadings,
    index=fa_vars,
    columns=[f'Factor {i+1}' for i in range(n_factors)]
)

# --- NEW EDIT: Drop 'congestion_level_encoded' from loadings ---
if "congestion_level_encoded" in loadings_df.index:
    loadings_df.drop("congestion_level_encoded", inplace=True)
    print("Dropped 'congestion_level_encoded' from loadings dataframe (as requested).")

print(f"Factor Analysis with {n_factors} factors:")
print("\nFactor Loadings Matrix:")
print(loadings_df.round(3))

# Save Loadings CSV to S3
print("⬆️  Uploading Factor Loadings CSV to Gold...")
csv_buffer = io.BytesIO()
loadings_df.to_csv(csv_buffer)
csv_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/factor_loadings.csv', Body=csv_buffer)

# Visualize factor loadings
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=1,
            cbar_kws={'label': 'Factor Loading'})
plt.title(f'Factor Loadings (Varimax Rotation, {n_factors} Factors)', fontsize=16)
plt.tight_layout()

# Save Plot 4 to S3
print("⬆️  Uploading Factor Loadings Heatmap to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/factor_loadings_heatmap.png', Body=img_buffer)
plt.close()

# ==========================================
# 5. FACTOR INTERPRETATION & NAMING (NEW SECTION)
# ==========================================
# Helper to capture text for the .txt file upload
report_buffer = io.StringIO()

def log_print(text):
    """Helper to print to console AND write to buffer"""
    print(text)
    report_buffer.write(text + "\n")

log_print("\n" + "=" * 80)
log_print("FACTOR INTERPRETATION AND NAMING")
log_print("=" * 80)

# Define threshold for significant loadings
threshold = 0.11

# Analyze each factor
factor_names = []
factor_interpretations = []

for i in range(n_factors):
    factor_col = f'Factor {i+1}'
    loadings_series = loadings_df[factor_col].abs()
    
    # Get variables with significant loadings
    significant_vars = loadings_series[loadings_series > threshold].sort_values(ascending=False)
    
    log_print(f"\n{factor_col}:")
    log_print("-" * 40)
    
    # Get top variables with their actual loadings (signed)
    top_vars = []
    for var in significant_vars.index:
        loading = loadings_df.loc[var, factor_col]
        top_vars.append((var, loading))
        log_print(f"  {var}: {loading:.3f}")
    
    # Interpret based on variables
    name = f"Factor {i+1}"
    interpretation = "Unknown"

    if i == 0:
        # Check what variables load highly
        weather_vars_in_factor = [var for var, _ in top_vars if var in weather_vars]
        traffic_vars_in_factor = [var for var, _ in top_vars if var in traffic_vars]
        
        if len(weather_vars_in_factor) > len(traffic_vars_in_factor):
            name = "Weather Severity Factor"
            interpretation = "Represents adverse weather conditions that affect driving"
        else:
            name = "Traffic Stress Factor"
            interpretation = "Represents high traffic volume and congestion levels"
    
    elif i == 1:
        # Second factor often captures different pattern
        if 'accident_count' in [var for var, _ in top_vars]:
            name = "Accident Risk Factor"
            interpretation = "Represents conditions associated with increased accident likelihood"
        else:
            name = "Visibility & Speed Factor"
            interpretation = "Related to visibility conditions and average speed"
    
    elif i == 2:
        name = "Atmospheric Pressure Factor"
        interpretation = "Mainly captures atmospheric pressure variations"
    
    factor_names.append(name)
    factor_interpretations.append(interpretation)
    
    log_print(f"\n  Proposed Name: {name}")
    log_print(f"  Interpretation: {interpretation}")

# Create final factor interpretation table
interpretation_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(n_factors)],
    'Proposed Name': factor_names,
    'Key Variables (Loadings > 0.4)': [''] * n_factors,
    'Interpretation': factor_interpretations,
    'Variance Explained': fa.get_factor_variance()[0]
})

# Fill in key variables
for i in range(n_factors):
    factor_col = f'Factor {i+1}'
    significant_vars = loadings_df[factor_col].abs()
    significant_vars = significant_vars[significant_vars > threshold].sort_values(ascending=False)
    
    var_list = []
    for var in significant_vars.index:
        loading = loadings_df.loc[var, factor_col]
        var_list.append(f"{var} ({loading:.2f})")
    
    interpretation_df.loc[i, 'Key Variables (Loadings > 0.4)'] = ', '.join(var_list)

log_print("\n" + "=" * 80)
log_print("FINAL FACTOR INTERPRETATION TABLE")
log_print("=" * 80)
log_print(interpretation_df.to_string(index=False))

# --- SAVE TEXT REPORT TO GOLD ---
print("⬆️  Uploading Interpretation Report to Gold bucket...")
text_output = report_buffer.getvalue()
s3.put_object(Bucket='gold', Key='factor_analysis/factor_interpretation.txt', Body=text_output.encode('utf-8'))

# --- SAVE SUMMARY CSV TO GOLD ---
print("⬆️  Uploading Interpretation Summary CSV to Gold bucket...")
csv_buffer = io.BytesIO()
interpretation_df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/factor_interpretation_summary.csv', Body=csv_buffer)


print("\n" + "=" * 80)
print("FACTOR DISTRIBUTIONS")
print("=" * 80)

# Visualize distribution of factor scores
fig, axes = plt.subplots(1, n_factors, figsize=(15, 4))
for i, (ax, name) in enumerate(zip(axes, factor_names)):
    ax.hist(factor_scores[:, i], bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribution of {name} Scores', fontsize=12)
    ax.set_xlabel('Factor Score')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save Plot 5 to S3
print("⬆️  Uploading Factor Scores Distribution to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/factor_scores_distribution.png', Body=img_buffer)
plt.close()

print("\n" + "=" * 80)
print("ANALYZING FACTOR RELATIONSHIPS WITH CONGESTION LEVEL")
print("=" * 80)

# Calculate correlation between factor scores and congestion
congestion_correlations = []
for i, name in enumerate(factor_names):
    corr = np.corrcoef(factor_scores[:, i], df_fa['congestion_level_encoded'])[0, 1]
    congestion_correlations.append((name, corr))

print("Correlation between Factor Scores and Congestion Level:")
for name, corr in congestion_correlations:
    print(f"  {name}: {corr:.3f}")

# Visualize factor scores by congestion level
fig, axes = plt.subplots(1, n_factors, figsize=(15, 5))
congestion_levels = ['Low', 'Medium', 'High', 'Unknown']

for i, (ax, name) in enumerate(zip(axes, factor_names)):
    data_for_plot = []
    for level_idx, level in enumerate(congestion_levels):
        mask = df_processed['congestion_level'] == level
        if mask.any():
            scores = factor_scores[mask, i]
            data_for_plot.append(scores)
    
    ax.boxplot(data_for_plot, labels=congestion_levels)
    ax.set_title(f'{name} by Congestion Level', fontsize=12)
    ax.set_ylabel('Factor Score')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save Plot 6 to S3
print("⬆️  Uploading Factor Boxplots to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/factor_congestion_boxplots.png', Body=img_buffer)
plt.close()

# Calculate average factor scores by congestion level
print("\nAverage Factor Scores by Congestion Level:")
for level in congestion_levels:
    mask = df_processed['congestion_level'] == level
    if mask.sum() > 0:
        print(f"\n{level}:")
        for i, name in enumerate(factor_names):
            avg_score = factor_scores[mask, i].mean()
            print(f"  {name}: {avg_score:.3f}")

# Final Bar Chart for Weather Impact
df_weather_impact = df_processed[weather_vars + ['congestion_level']].copy()
df_weather_impact['congestion_encoded'] = df_weather_impact['congestion_level'].map(
    {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 1}
).fillna(1)

correlations = {}
for weather_var in weather_vars:
    corr = np.corrcoef(df_weather_impact[weather_var], df_weather_impact['congestion_encoded'])[0, 1]
    correlations[weather_var] = abs(corr)

fig, ax = plt.subplots(figsize=(10, 6))
sorted_corrs = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
variables = list(sorted_corrs.keys())
values = list(sorted_corrs.values())

bars = ax.barh(variables, values, color='steelblue', alpha=0.8)
ax.set_xlabel('Absolute Correlation with Congestion Level', fontsize=12)
ax.set_title('Impact of Weather Variables on Traffic Congestion', fontsize=14, pad=20)
ax.set_xlim(0, max(values) * 1.2) # Adjust limit dynamically

for bar, value in zip(bars, values):
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{value:.3f}', va='center', fontsize=10)

plt.tight_layout()

# Save Plot 7 to S3
print("⬆️  Uploading Weather Impact Bar Chart to Gold...")
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', bbox_inches='tight')
img_buffer.seek(0)
s3.put_object(Bucket='gold', Key='factor_analysis/weather_impact_chart.png', Body=img_buffer)
plt.close()

print("="*60)
print("✅ FACTOR ANALYSIS COMPLETE. Results saved to Gold bucket (factor_analysis/).")
print("="*60)