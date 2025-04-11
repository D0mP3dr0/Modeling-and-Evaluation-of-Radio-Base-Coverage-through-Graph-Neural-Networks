#!/usr/bin/env python
# coding: utf-8

# # Análise de Estações Rádio Base (ERB) - Google Colab
# 
# Análise completa de dados de Estações Rádio Base, incluindo visualizações geográficas, análise estatística e construção de grafos de rede no Google Colab.

# ## 1. Configuração do Ambiente
# 
# Primeiro, vamos instalar todas as dependências necessárias e clonar o repositório do projeto.

# In[ ]:


# Instalar dependências necessárias
!pip install numpy pandas scipy geopandas shapely pyproj matplotlib seaborn folium plotly scikit-learn networkx graphviz


# In[ ]:


# Clonar o repositório
!git clone https://github.com/D0mP3dr0/Modeling-and-Evaluation-of-Radio-Base-Coverage-through-Graph-Neural-Networks.git rbs_analysis
%cd rbs_analysis


# In[ ]:


# Fazer upload do arquivo de dados
from google.colab import files
import os

# Criar o diretório de dados se não existir
!mkdir -p data

# Opção 1: Upload do arquivo pelo usuário
print("Por favor, faça o upload do arquivo 'csv_licenciamento_bruto.csv.csv' quando solicitado.")
uploaded = files.upload()

for filename in uploaded.keys():
    # Mover o arquivo para o diretório de dados
    !mv "$filename" "data/$filename"
    print(f"Arquivo {filename} movido para o diretório 'data/'")

# Verificar se o arquivo existe
if os.path.exists('data/csv_licenciamento_bruto.csv.csv'):
    print("Arquivo de dados encontrado com sucesso!")
else:
    print("ATENÇÃO: Arquivo de dados não encontrado. Algumas células abaixo podem falhar.")
    print("Por favor, certifique-se de fazer o upload do arquivo 'csv_licenciamento_bruto.csv.csv'")


# ### Preparação do Sistema
# 
# Vamos configurar o ambiente Python para utilizar o módulo RBS Analysis.

# In[ ]:


# Adicionar o diretório do projeto ao PYTHONPATH
import sys
import os
from pathlib import Path

# Obter o diretório atual
current_dir = Path.cwd()
if current_dir not in sys.path:
    sys.path.append(str(current_dir))

# Configurar a variável de ambiente para usar GPU se disponível
os.environ['USE_GPU'] = 'true'

# Importar módulos do projeto
try:
    from src.config import setup_logging
    
    # Configurar logging
    logger = setup_logging('colab_notebook.log', console_level=20)  # INFO level
    
    # Mostrar resumo da configuração
    print("\n=== RBS Analysis Configuration ===")
    print(f"Root directory: {current_dir}")
    print(f"Data directory: {current_dir / 'data'}")
    print(f"GPU acceleration: {os.environ.get('USE_GPU', 'false')}")
    print("=================================\n")
    
    print("Sistema configurado com sucesso!")
except Exception as e:
    print(f"Erro ao configurar o sistema: {e}")
    print("Verifique se o repositório foi clonado corretamente.")


# ## 2. Carregamento e Processamento de Dados

# In[ ]:


# Carregar e processar os dados
try:
    # Carregar dados usando a função do módulo
    from src.data_processing import load_and_process_data
    input_path = 'data/csv_licenciamento_bruto.csv.csv'
    gdf_rbs = load_and_process_data(input_path)
    
    print(f"Dados carregados com sucesso! Total de registros: {len(gdf_rbs)}")
    
    # Mostrar as primeiras linhas
    display(gdf_rbs.head())
    
    # Mostrar informações sobre as colunas
    print("\nInformações sobre os dados:")
    gdf_rbs.info()
    
    # Mostrar estatísticas descritivas
    print("\nEstatísticas descritivas:")
    display(gdf_rbs.describe())
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")
    print("Tentando carregar os dados manualmente...")
    
    # Carregar manualmente como fallback
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    
    df = pd.read_csv('data/csv_licenciamento_bruto.csv.csv')
    print(f"Dados carregados manualmente. Total de registros: {len(df)}")
    
    # Tentar converter para GeoDataFrame
    try:
        # Identificar colunas de latitude e longitude
        lat_col = next(col for col in df.columns if 'lat' in col.lower())
        lon_col = next(col for col in df.columns if 'lon' in col.lower())
        
        # Remover linhas com coordenadas faltantes
        df_clean = df.dropna(subset=[lat_col, lon_col])
        
        # Converter para GeoDataFrame
        geometry = [Point(xy) for xy in zip(df_clean[lon_col], df_clean[lat_col])]
        gdf_rbs = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
        
        print(f"Conversão para GeoDataFrame bem-sucedida. Total de registros: {len(gdf_rbs)}")
        display(gdf_rbs.head())
    except Exception as e2:
        print(f"Erro na conversão para GeoDataFrame: {e2}")
        # Usar o DataFrame normal
        gdf_rbs = df
        print("Usando DataFrame normal sem informações geográficas.")
        display(gdf_rbs.head())


# ## 3. Visualizações Básicas
# 
# Vamos criar algumas visualizações básicas para entender a distribuição dos dados.

# In[ ]:


# Visualização geográfica das estações
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

try:
    # Verificar se os dados são um GeoDataFrame
    if isinstance(gdf_rbs, gpd.GeoDataFrame):
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plotar pontos
        gdf_rbs.plot(ax=ax, markersize=5, alpha=0.7, column='Operator' if 'Operator' in gdf_rbs.columns else None, legend=True)
        
        # Configurar título e layout
        ax.set_title('Distribuição Geográfica das Estações Rádio Base', fontsize=16)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        
        # Mostrar o gráfico
        plt.show()
    else:
        print("Os dados não são um GeoDataFrame. Não é possível criar a visualização geográfica.")
except Exception as e:
    print(f"Erro ao criar a visualização geográfica: {e}")


# In[ ]:


# Visualização da distribuição por operadora
try:
    # Verificar se a coluna de operadora existe
    operator_col = next((col for col in gdf_rbs.columns if 'operator' in col.lower()), None)
    
    if operator_col:
        # Contar por operadora
        operator_counts = gdf_rbs[operator_col].value_counts()
        
        # Criar gráfico
        plt.figure(figsize=(12, 6))
        ax = operator_counts.plot(kind='bar', color='skyblue')
        
        # Configurar título e rótulos
        plt.title('Distribuição de ERBs por Operadora', fontsize=16)
        plt.xlabel('Operadora')
        plt.ylabel('Quantidade de ERBs')
        plt.xticks(rotation=45)
        
        # Adicionar valores nas barras
        for i, v in enumerate(operator_counts):
            ax.text(i, v + 0.1, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Coluna de operadora não encontrada nos dados.")
except Exception as e:
    print(f"Erro ao criar a visualização por operadora: {e}")


# ## 4. Análise de Grafos
# 
# Vamos analisar as ERBs usando teoria de grafos, onde cada estação é um nó no grafo e as conexões são estabelecidas com base na proximidade geográfica.

# In[ ]:


# Criação e análise de grafo
try:
    from src.graph_analysis import create_rbs_graph, calculate_graph_metrics, visualize_graph
    
    # Verificar se temos um GeoDataFrame
    if isinstance(gdf_rbs, gpd.GeoDataFrame):
        # Criar grafo com raio de conexão de 3km
        print("Criando grafo de ERBs...")
        G = create_rbs_graph(gdf_rbs, connection_radius=3.0)
        
        # Calcular métricas do grafo
        print("\nCalculando métricas do grafo...")
        metrics = calculate_graph_metrics(G)
        
        # Mostrar métricas
        print("\nMétricas do grafo:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        # Visualizar o grafo
        print("\nVisualizando o grafo...")
        # Criar um arquivo temporário para a visualização
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            visualize_graph(G, tmp.name, title="Grafo de Conectividade entre ERBs", by_operator=True)
            # Exibir a imagem
            from IPython.display import Image
            display(Image(filename=tmp.name))
    else:
        print("Os dados não são um GeoDataFrame. Não é possível criar o grafo.")
except Exception as e:
    print(f"Erro na análise de grafos: {e}")
    print("\nTentando criar um grafo simplificado...")
    
    # Fallback: criar um grafo simplificado
    try:
        import networkx as nx
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        
        # Identificar colunas de latitude e longitude
        lat_col = next((col for col in gdf_rbs.columns if 'lat' in col.lower()), None)
        lon_col = next((col for col in gdf_rbs.columns if 'lon' in col.lower()), None)
        
        if lat_col and lon_col:
            # Limitar a 1000 pontos para evitar sobrecarga
            sample_size = min(1000, len(gdf_rbs))
            sample_df = gdf_rbs.sample(sample_size) if len(gdf_rbs) > sample_size else gdf_rbs
            
            # Extrair coordenadas
            coords = sample_df[[lon_col, lat_col]].values
            
            # Calcular matriz de distância
            dist_matrix = squareform(pdist(coords))
            
            # Criar grafo
            G = nx.Graph()
            
            # Adicionar nós
            for i in range(len(sample_df)):
                G.add_node(i, pos=(coords[i][0], coords[i][1]))
            
            # Adicionar arestas para pontos próximos (3km aproximado em graus)
            threshold = 0.03  # ~3km
            for i in range(len(dist_matrix)):
                for j in range(i+1, len(dist_matrix)):
                    if dist_matrix[i][j] < threshold:
                        G.add_edge(i, j, weight=1.0/max(0.001, dist_matrix[i][j]))
            
            # Mostrar estatísticas do grafo
            print(f"Grafo simplificado criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
            print(f"Densidade do grafo: {nx.density(G):.6f}")
            
            # Visualizar o grafo
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)  # Layout para visualização
            nx.draw(G, pos, node_size=20, node_color='skyblue', alpha=0.8, width=0.5)
            plt.title("Grafo Simplificado de ERBs")
            plt.show()
        else:
            print("Colunas de latitude e longitude não encontradas nos dados.")
    except Exception as e2:
        print(f"Erro ao criar grafo simplificado: {e2}")


# ## 5. Estimativa de Cobertura
# 
# Vamos estimar a cobertura das estações rádio base com base em seus parâmetros de transmissão.

# In[ ]:


# Estimativa de cobertura
try:
    from src.coverage_models import estimate_coverage
    
    # Verificar se temos um GeoDataFrame
    if isinstance(gdf_rbs, gpd.GeoDataFrame):
        # Estimar cobertura
        print("Estimando cobertura das ERBs...")
        gdf_with_coverage = estimate_coverage(gdf_rbs)
        
        # Mostrar informações de cobertura
        if 'Coverage_Radius_km' in gdf_with_coverage.columns:
            print(f"\nEstatísticas de raio de cobertura (km):")
            print(gdf_with_coverage['Coverage_Radius_km'].describe())
            
            # Visualizar cobertura
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plotar pontos com tamanho baseado na cobertura
            gdf_with_coverage.plot(
                ax=ax,
                column='Coverage_Radius_km',
                cmap='viridis',
                markersize=gdf_with_coverage['Coverage_Radius_km'] * 5, # Tamanho proporcional à cobertura
                legend=True,
                alpha=0.7
            )
            
            ax.set_title('Estimativa de Cobertura das ERBs', fontsize=16)
            plt.tight_layout()
            plt.show()
    else:
        print("Os dados não são um GeoDataFrame. Não é possível estimar a cobertura.")
except Exception as e:
    print(f"Erro na estimativa de cobertura: {e}")


# ## 6. Mapa Interativo
# 
# Vamos criar um mapa interativo para visualizar as ERBs e suas coberturas.

# In[ ]:


# Criar mapa interativo com Folium
try:
    import folium
    from folium.plugins import MarkerCluster
    
    # Verificar se temos um GeoDataFrame
    if isinstance(gdf_rbs, gpd.GeoDataFrame):
        # Calcular centro do mapa
        center_lat = gdf_rbs.geometry.y.mean()
        center_lon = gdf_rbs.geometry.x.mean()
        
        # Criar mapa base
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Adicionar cluster de marcadores
        marker_cluster = MarkerCluster().add_to(m)
        
        # Verificar se temos estimativa de cobertura
        has_coverage = 'Coverage_Radius_km' in gdf_rbs.columns
        
        # Verificar se temos dados de operadora
        operator_col = next((col for col in gdf_rbs.columns if 'operator' in col.lower()), None)
        
        # Adicionar marcadores para cada ERB
        for idx, row in gdf_rbs.iterrows():
            # Extrair coordenadas
            lat, lon = row.geometry.y, row.geometry.x
            
            # Construir popup
            popup_text = f"ID: {idx}<br>"
            
            if operator_col:
                popup_text += f"Operadora: {row[operator_col]}<br>"
                
            if has_coverage:
                popup_text += f"Raio de Cobertura: {row['Coverage_Radius_km']:.2f} km<br>"
                
            # Adicionar mais informações se disponíveis
            for col in ['Tecnologia', 'FreqTxMHz', 'PotenciaTransmissorWatts', 'Municipio']:
                if col in row and pd.notna(row[col]):
                    popup_text += f"{col}: {row[col]}<br>"
            
            # Criar popup
            popup = folium.Popup(popup_text, max_width=300)
            
            # Adicionar marcador
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(icon="signal", prefix="fa")
            ).add_to(marker_cluster)
            
            # Adicionar círculo de cobertura se disponível
            if has_coverage and pd.notna(row['Coverage_Radius_km']):
                folium.Circle(
                    location=[lat, lon],
                    radius=row['Coverage_Radius_km'] * 1000,  # Converter para metros
                    color='blue',
                    fill=True,
                    fill_opacity=0.1
                ).add_to(m)
        
        # Exibir o mapa
        m
    else:
        print("Os dados não são um GeoDataFrame. Não é possível criar o mapa interativo.")
except Exception as e:
    print(f"Erro ao criar mapa interativo: {e}")


# ## 7. Conclusão
# 
# Neste notebook, exploramos os dados de Estações Rádio Base (ERB) usando diversas técnicas de análise e visualização. Realizamos:
# 
# 1. Carregamento e processamento dos dados
# 2. Visualizações básicas para entender a distribuição geográfica
# 3. Análise de grafos para identificar padrões de conectividade
# 4. Estimativa de cobertura das estações
# 5. Criação de um mapa interativo para exploração dos dados
# 
# Para continuar a análise, você pode:
# - Explorar outros módulos do projeto, como análise temporal ou predição
# - Modificar parâmetros nas funções de análise para obter diferentes resultados
# - Adicionar mais visualizações ou análises específicas para seu caso de uso
# 
# ### Próximos Passos
# 
# - Aprofundar a análise por operadora e tecnologia
# - Realizar análise de qualidade de cobertura
# - Explorar modelos de predição para novos sites de ERB
# 
# Para mais informações, visite o [repositório do projeto](https://github.com/D0mP3dr0/Modeling-and-Evaluation-of-Radio-Base-Coverage-through-Graph-Neural-Networks). 