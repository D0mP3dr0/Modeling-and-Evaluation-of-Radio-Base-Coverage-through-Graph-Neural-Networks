import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from mpl_toolkits.basemap import Basemap # Comentado devido à complexidade de instalação
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def setup_visualization_options():
    """Configura opções globais para as bibliotecas de visualização."""
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    pd.set_option('display.max_columns', None)
    plt.rcParams['figure.figsize'] = (12, 8)

def analise_exploratoria_erbs(df: pd.DataFrame, results_path: str):
    """Realiza análise exploratória e gera visualizações para o dataset de ERBs."""

    if df is None or df.empty:
        print("DataFrame vazio ou inválido. Análise não pode ser realizada.")
        return

    print("=" * 80)
    print("ANÁLISE EXPLORATÓRIA - ESTAÇÕES RÁDIO BASE")
    print("=" * 80)

    # --- 1. VISÃO GERAL --- 
    print("\n1. VISÃO GERAL DOS DADOS")
    print("-" * 50)
    print(f"Total de estações: {df.shape[0]}")
    print(f"Variáveis disponíveis: {df.shape[1]}")
    print("\nPrimeiras linhas do dataset:")
    print(df.head())
    print("\nTipos de dados:")
    print(df.dtypes)

    # --- 2. VALORES AUSENTES --- 
    print("\n\n2. ANÁLISE DE VALORES AUSENTES")
    print("-" * 50)
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Valores ausentes': missing_values,
        'Percentual (%)': missing_percent.round(2)
    })
    missing_data_sorted = missing_data[missing_data['Valores ausentes'] > 0].sort_values('Percentual (%)', ascending=False)
    if not missing_data_sorted.empty:
      print(missing_data_sorted)
    else:
      print("Não há valores ausentes no dataset.")

    # --- 3. ESTATÍSTICAS DESCRITIVAS --- 
    print("\n\n3. ESTATÍSTICAS DESCRITIVAS")
    print("-" * 50)
    
    # Tenta converter colunas potencialmente numéricas que podem estar como object
    potential_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in potential_numeric_cols:
        try:
            # Tenta converter para numérico após tratar vírgulas como separadores decimais
            converted_col = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')
            # Se a maioria dos valores não nulos pôde ser convertida, atualiza a coluna
            if converted_col.notna().sum() / df[col].notna().sum() > 0.8: # Threshold de 80%
                df[col] = converted_col
                print(f"Coluna '{col}' convertida para tipo numérico.")
        except AttributeError:
            # Ignora colunas que não são strings
            pass 
        except Exception as e:
            print(f"Não foi possível processar coluna '{col}' para conversão numérica: {e}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        print("\nEstatísticas de variáveis numéricas:")
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats['median'] = df[numeric_cols].median()
        numeric_stats['variance'] = df[numeric_cols].var()
        print(numeric_stats)
    else:
        print("Nenhuma coluna numérica encontrada para estatísticas.")

    # --- 4. ANÁLISE GEOGRÁFICA --- 
    print("\n\n4. ANÁLISE GEOGRÁFICA")
    print("-" * 50)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        print(f"Amplitude Lat: {df['Latitude'].min()} a {df['Latitude'].max()}")
        print(f"Amplitude Lon: {df['Longitude'].min()} a {df['Longitude'].max()}")
        if 'Municipio.NomeMunicipio' in df.columns:
            print("\nTop 10 Municípios por nº de ERBs:")
            print(df['Municipio.NomeMunicipio'].value_counts().head(10))
        
        # Gerar Mapa Folium (baseado no código visto)
        try:
            geo_df = df.dropna(subset=['Latitude', 'Longitude'])
            geo_df = geo_df[(geo_df['Latitude'] != 0) & (geo_df['Longitude'] != 0)]
            
            if not geo_df.empty:
                map_center = [geo_df['Latitude'].mean(), geo_df['Longitude'].mean()]
                m = folium.Map(location=map_center, zoom_start=10)
                marker_cluster = MarkerCluster().add_to(m)

                for idx, row in geo_df.iterrows():
                    popup_text = f"""
                    <b>Operadora:</b> {row.get('NomeEntidade', 'N/A')}<br>
                    <b>Tecnologia:</b> {row.get('Tecnologia', 'N/A')}<br>
                    <b>Freq. Tx:</b> {row.get('FreqTxMHz', 'N/A')} MHz<br>
                    <b>Altura:</b> {row.get('AlturaAntena', 'N/A')} m<br>
                    """
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{row.get('NomeEntidade', 'ERB')}: {row.get('Tecnologia', '')}"
                    ).add_to(marker_cluster)
                
                map_file_path = f"{results_path}/mapa_erbs_distribuicao.html"
                m.save(map_file_path)
                print(f"\nMapa interativo de distribuição salvo em: {map_file_path}")
            else:
                 print("\nNão há dados de geolocalização válidos para gerar o mapa.")
                 
        except Exception as e:
            print(f"\nErro ao gerar mapa Folium: {e}")
            
    else:
        print("Colunas 'Latitude' e/ou 'Longitude' não encontradas. Análise geográfica limitada.")
        
    # --- 5. ANÁLISE TECNOLÓGICA --- 
    print("\n\n5. ANÁLISE TECNOLÓGICA")
    print("-" * 50)
    if 'Tecnologia' in df.columns:
        print("\nDistribuição de tecnologias:")
        print(df['Tecnologia'].value_counts())
    if 'tipoTecnologia' in df.columns:
        print("\nDistribuição de tipos de tecnologia:")
        print(df['tipoTecnologia'].value_counts())
        
    # --- 6. ANÁLISE DE FREQUÊNCIAS --- 
    print("\n\n6. ANÁLISE DE FREQUÊNCIAS")
    print("-" * 50)
    if 'FreqTxMHz' in df.columns:
        print(f"Freq. Tx: {df['FreqTxMHz'].min()} MHz a {df['FreqTxMHz'].max()} MHz")
    if 'FreqRxMHz' in df.columns:
        print(f"Freq. Rx: {df['FreqRxMHz'].min()} MHz a {df['FreqRxMHz'].max()} MHz")
        
    # --- 7. ANÁLISE POR OPERADORA --- 
    print("\n\n7. ANÁLISE POR OPERADORA")
    print("-" * 50)
    if 'NomeEntidade' in df.columns:
        print("\nTop 10 Operadoras por nº de ERBs:")
        print(df['NomeEntidade'].value_counts().head(10))

    # --- 8. VISUALIZAÇÕES ADICIONAIS (Gráficos Estáticos) ---
    print("\n\n8. GERANDO GRÁFICOS ADICIONAIS")
    print("-" * 50)
    
    try:
        plt.figure(figsize=(18, 6))
        
        # Histograma Altura Antena
        if 'AlturaAntena' in df.columns and pd.api.types.is_numeric_dtype(df['AlturaAntena']):
            plt.subplot(1, 2, 1)
            sns.histplot(df['AlturaAntena'].dropna(), bins=30, kde=True)
            plt.title('Distribuição de Altura das Antenas')
            plt.xlabel('Altura (m)')
            plt.ylabel('Frequência')
        else:
             print("Coluna 'AlturaAntena' não encontrada ou não numérica para histograma.")
             
        # Gráfico de Barras Tecnologias
        if 'Tecnologia' in df.columns:
            plt.subplot(1, 2, 2)
            tech_counts = df['Tecnologia'].value_counts()
            sns.barplot(x=tech_counts.index, y=tech_counts.values)
            plt.title('Distribuição de Tecnologias')
            plt.xlabel('Tecnologia')
            plt.ylabel('Quantidade')
            plt.xticks(rotation=45, ha='right')
        else:
             print("Coluna 'Tecnologia' não encontrada para gráfico de barras.")

        plt.tight_layout()
        plot_file_path = f"{results_path}/plots_distribuicao_altura_tecnologia.png"
        plt.savefig(plot_file_path)
        print(f"Gráficos de distribuição salvos em: {plot_file_path}")
        plt.close() # Fechar a figura para liberar memória

    except Exception as e:
        print(f"Erro ao gerar gráficos estáticos: {e}")

    # Adicione aqui o código para as outras visualizações que estavam no notebook
    # (ex: Ganho x Potência), adaptando para salvar em `results_path`
    print("\nAnálise exploratória concluída.")

# Exemplo de como chamar (será feito no main.py):
# setup_visualization_options()
# df_analise = pd.read_csv('data/erb_sorocaba_limpo.csv') # Supondo que o arquivo limpo está em data/
# analise_exploratoria_erbs(df_analise, 'results')
