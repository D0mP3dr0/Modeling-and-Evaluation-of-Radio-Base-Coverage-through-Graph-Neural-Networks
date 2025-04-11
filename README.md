# Modeling and Evaluation of Radio Base Station Coverage Through GNN

This project performs advanced analysis of Radio Base Stations (RBS), from data exploration,
coverage processing, geospatial visualizations, to graph modeling and neural networks
for analyzing connectivity between stations.

## Overview

The project addresses the problem of RBS coverage analysis and modeling using Anatel data,
with special focus on the Sorocaba-SP area. It implements:

1. **Data Processing**: Cleaning and transformation of raw Anatel data.
2. **Coverage Modeling**: Calculation of EIRP, coverage radius and creation of sector geometries.
3. **Geospatial Visualizations**: Interactive and static maps of coverage, overlap and heatmaps.
4. **Graph Analysis**: Modeling RBS as nodes in a graph, with edges representing connectivity.
5. **GNN Preparation**: Transformation into PyTorch Geometric compatible format for advanced analysis via GNN.

## Project Structure

```
projeto_erb/
├── data/                  # Raw and processed data
│   └── README.md          # Instructions about required data
├── results/               # Generated results (maps, charts, metrics)
├── src/                   # Modularized source code
│   ├── __init__.py        # Package initialization file
│   ├── analysis.py        # Basic exploratory analysis functions
│   ├── coverage_models.py # Coverage calculation models
│   ├── data_processing.py # Data processing functions
│   ├── graph_analysis.py  # Functions for graph analysis and GNN
│   └── visualization.py   # Functions for advanced visualizations
├── main.py                # Main script to execute the complete flow
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Features

### 1. Data Processing
- Cleaning of raw Anatel data
- Filtering for region of interest
- Normalization of operators (Claro, Vivo, TIM, Oi)
- Intelligent filling of missing values

### 2. Coverage Modeling
- Calculation of EIRP (Effective Isotropic Radiated Power)
- Calculation of coverage radii based on frequency, power and environment
- Generation of coverage sectors (polygons) for each RBS
- Classification of area type (urban, suburban, rural)

### 3. Advanced Visualizations
- RBS positioning map by operator
- Coverage maps by operator
- Coverage overlap map
- EIRP power heat map
- Interactive folium map for dynamic navigation

### 4. Graph Analysis
- Construction of connectivity graphs between RBS
- Calculation of network metrics (centrality, clustering, etc.)
- Graph visualization by operator and centrality
- Graph generation based on Voronoi diagram

### 5. GNN Preparation
- Conversion to PyTorch Geometric format
- Definition of node features (power, gain, etc.)
- Definition of edge features (distance, overlap)
- Structuring for future GNN application

## How to Use

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`
- Anatel licensing CSV file (not included due to size)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/D0mP3dr0/Modeling-and-Evaluation-of-Radio-Base-Coverage-through-Graph-Neural-Networks.git
   cd projeto_erb
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   # venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add data:**
   - Place the Anatel licensing CSV file in the `data/` folder.
   - Rename to `csv_licenciamento_bruto.csv` or adjust the path in `main.py`.

### Execution

To run the complete project:
```bash
python main.py
```

The script processes all steps sequentially and saves the results in the `results/` folder.

### Generated Results

- **Statistics:** Metrics about RBS and graphs
- **Static Maps:** High-resolution PNG visualizations in the `results/` folder
- **Interactive Maps:** HTML file with interactive Folium map
- **Graphs:** Visualizations of connectivity and Voronoi graphs

## Main Dependencies

- **Data Analysis:** pandas, numpy
- **Geospatial:** geopandas, shapely, folium
- **Visualization:** matplotlib, seaborn, contextily
- **Graphs:** networkx, scipy
- **GNN:** torch, torch-geometric (optional)

## Implementation Notes

- The code is structured to be modular and extensible.
- Detailed documentation functions explain parameters and returns.
- The system can operate even with partial data.
- GNN implementation requires PyTorch and PyTorch Geometric, but other functionalities remain operational without them.

## Authors and Contributions

- **Development:** D0mP3dr0
- **Contributions:** PRs are welcome!
