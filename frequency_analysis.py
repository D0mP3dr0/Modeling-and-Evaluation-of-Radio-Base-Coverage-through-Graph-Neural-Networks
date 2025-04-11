import os
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# File paths
csv_path = 'data/csv_licenciamento_bruto.csv.csv'

# Check if file exists
if not os.path.exists(csv_path):
    print(f"Error: RBS data file {csv_path} does not exist.")
    exit(1)

print(f"Loading RBS data from {csv_path}...")

# Function to read CSV and extract relevant data with focus on frequencies
def read_rbs_data(filepath):
    data = []
    operators = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Extract basic information
                operator = row.get('NomeEntidade', 'Unknown')
                tech = row.get('Tecnologia', 'Unknown')
                
                # Extract frequency information
                try:
                    freq_tx = float(row.get('FreqTxMHz', 0))
                except (ValueError, TypeError):
                    freq_tx = 0
                    
                try:
                    freq_rx = float(row.get('FreqRxMHz', 0))
                except (ValueError, TypeError):
                    freq_rx = 0
                
                # Skip records with no frequency data
                if freq_tx > 0 or freq_rx > 0:
                    # Create a record
                    record = {
                        'operator': operator,
                        'tech': tech,
                        'freq_tx': freq_tx,
                        'freq_rx': freq_rx
                    }
                    
                    data.append(record)
                    
                    # Count by operator
                    if operator in operators:
                        operators[operator] += 1
                    else:
                        operators[operator] = 1
            except Exception as e:
                # Skip problematic rows
                continue
    
    return data, operators

# Read the data
rbs_data, operators = read_rbs_data(csv_path)
print(f"Loaded {len(rbs_data)} RBS records with frequency data")

# Filter for major telecom operators only
major_operators = ['TELEFONICA BRASIL S.A.', 'CLARO S.A.', 'TIM S A']
telecom_data = [r for r in rbs_data if r['operator'] in major_operators]
print(f"Selected {len(telecom_data)} stations from major telecom operators")

# Frequency bands categorization
def categorize_frequency(freq):
    if freq == 0:
        return 'Unknown'
    elif freq < 700:
        return '< 700 MHz'
    elif 700 <= freq < 900:
        return '700-900 MHz'
    elif 900 <= freq < 1800:
        return '900-1800 MHz'
    elif 1800 <= freq < 2100:
        return '1800-2100 MHz'
    elif 2100 <= freq < 2600:
        return '2100-2600 MHz'
    elif 2600 <= freq < 3500:
        return '2600-3500 MHz'
    else:
        return '> 3500 MHz'

# Group data by operator
operator_freq_data = defaultdict(lambda: defaultdict(list))
for record in telecom_data:
    # Add transmit frequency
    if record['freq_tx'] > 0:
        band = categorize_frequency(record['freq_tx'])
        operator_freq_data[record['operator']][band].append(record['freq_tx'])
    
    # Add receive frequency
    if record['freq_rx'] > 0:
        band = categorize_frequency(record['freq_rx'])
        operator_freq_data[record['operator']][band].append(record['freq_rx'])

# Calculate frequency statistics by operator and band
print("\nFrequency Usage by Operator:")
print("-" * 80)
for operator in major_operators:
    print(f"\n{operator}:")
    
    # Count frequencies in each band
    band_counts = {band: len(freqs) for band, freqs in operator_freq_data[operator].items()}
    total = sum(band_counts.values())
    
    # Print band distribution
    if total > 0:
        for band, count in sorted(band_counts.items()):
            if band != 'Unknown':
                percentage = (count / total) * 100
                print(f"  {band:<15}: {count:5} frequencies ({percentage:5.1f}%)")

# Technology vs. Frequency Analysis
tech_freq_data = defaultdict(list)
for record in telecom_data:
    if record['tech'] != 'Unknown' and record['tech'] is not None and record['tech'] != '':
        if record['freq_tx'] > 0:
            tech_freq_data[record['tech']].append(record['freq_tx'])

# Calculate average frequency by technology
tech_avg_freq = {}
for tech, freqs in tech_freq_data.items():
    tech_avg_freq[tech] = sum(freqs) / len(freqs) if freqs else 0

print("\nAverage Frequency by Technology:")
print("-" * 80)
for tech, avg_freq in sorted(tech_avg_freq.items(), key=lambda x: x[1]):
    if avg_freq > 0:
        print(f"{tech:<10}: {avg_freq:.2f} MHz")

# Create visualizations
print("\nCreating visualizations...")

# 1. Stacked bar chart of frequency bands by operator
plt.figure(figsize=(14, 8))

# Prepare data for stacked bar chart
bands = ['< 700 MHz', '700-900 MHz', '900-1800 MHz', '1800-2100 MHz', '2100-2600 MHz', '2600-3500 MHz', '> 3500 MHz']
operators_list = major_operators
data = []

for band in bands:
    band_data = []
    for operator in operators_list:
        band_count = len(operator_freq_data[operator].get(band, []))
        band_data.append(band_count)
    data.append(band_data)

# Create the stacked bar chart
bottom = np.zeros(len(operators_list))
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#d9d9d9']

for i, (band_data, band, color) in enumerate(zip(data, bands, colors)):
    plt.bar(operators_list, band_data, bottom=bottom, label=band, color=color)
    bottom += band_data

plt.title('Frequency Band Usage by Operator')
plt.xlabel('Operator')
plt.ylabel('Number of Frequencies')
plt.xticks([op.split()[0] for op in operators_list])  # Use first word of operator name for clarity
plt.legend(title='Frequency Bands')
plt.tight_layout()
plt.savefig('frequency_bands_by_operator.png')
print("Saved frequency bands by operator chart to frequency_bands_by_operator.png")

# 2. Violin plots of frequency distribution by operator
plt.figure(figsize=(12, 8))

# Prepare data for violin plots
violin_data = []
labels = []

for operator in operators_list:
    # Combine all frequencies for this operator
    all_freqs = []
    for freqs in operator_freq_data[operator].values():
        all_freqs.extend(freqs)
    
    if all_freqs:  # Only add if there's data
        violin_data.append(all_freqs)
        labels.append(operator.split()[0])  # Use first word of operator name for clarity

# Create the violin plot
if violin_data:
    plt.violinplot(violin_data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title('Frequency Distribution by Operator')
    plt.ylabel('Frequency (MHz)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('frequency_distribution_by_operator.png')
    print("Saved frequency distribution by operator to frequency_distribution_by_operator.png")

# 3. Bar chart of average frequency by technology
plt.figure(figsize=(12, 6))

# Sort technologies by average frequency
sorted_techs = sorted(tech_avg_freq.items(), key=lambda x: x[1])
techs = [t[0] for t in sorted_techs if t[1] > 0]
avg_freqs = [t[1] for t in sorted_techs if t[1] > 0]

if techs:
    # Create bar chart
    plt.barh(techs, avg_freqs, color='skyblue')
    plt.xlabel('Average Frequency (MHz)')
    plt.ylabel('Technology')
    plt.title('Average Frequency by Technology')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('avg_frequency_by_technology.png')
    print("Saved average frequency by technology to avg_frequency_by_technology.png")

# 4. Heatmap of technology usage across frequency bands
tech_band_matrix = defaultdict(lambda: defaultdict(int))

for record in telecom_data:
    if record['tech'] != 'Unknown' and record['tech'] is not None and record['tech'] != '':
        if record['freq_tx'] > 0:
            band = categorize_frequency(record['freq_tx'])
            tech_band_matrix[record['tech']][band] += 1

# Convert to lists for plotting
techs_for_heatmap = list(tech_band_matrix.keys())
if techs_for_heatmap:
    plt.figure(figsize=(12, 8))
    
    # Create matrix for heatmap
    matrix = []
    for tech in techs_for_heatmap:
        row = [tech_band_matrix[tech].get(band, 0) for band in bands]
        matrix.append(row)
    
    # Create heatmap
    plt.imshow(matrix, cmap='YlGnBu')
    plt.colorbar(label='Number of Frequencies')
    plt.xticks(range(len(bands)), bands, rotation=45)
    plt.yticks(range(len(techs_for_heatmap)), techs_for_heatmap)
    plt.title('Technology Usage Across Frequency Bands')
    plt.tight_layout()
    plt.savefig('technology_frequency_heatmap.png')
    print("Saved technology-frequency heatmap to technology_frequency_heatmap.png")

print("\nAnalysis completed successfully!") 