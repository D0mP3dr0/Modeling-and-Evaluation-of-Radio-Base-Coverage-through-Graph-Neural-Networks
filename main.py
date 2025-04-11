import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

from src.data_processing import load_and_clean_data
from src.analysis import setup_visualization_options, analise_exploratoria_erbs
from src.coverage_models import (
    calcular_eirp, calcular_raio_cobertura_aprimorado, calcular_area_cobertura,
    criar_setor_preciso, calcular_tipo_area
)
from src.visualization import (
    configurar_estilo_visualizacao, criar_mapa_posicionamento, 
    criar_mapa_cobertura_por_operadora, criar_mapa_sobreposicao,
    criar_mapa_calor_potencia, criar_mapa_folium
)
from src.graph_analysis import (
    criar_grafo_erb, calcular_metricas_grafo, visualizar_grafo,
    criar_grafo_voronoi_erb, converter_para_pyg
)

# --- Configuração --- 
# Defina os caminhos aqui. Use caminhos relativos para melhor portabilidade.
INPUT_CSV_PATH = "data/csv_licenciamento_bruto.csv"  # Substitua pelo caminho real do arquivo
OUTPUT_CSV_PATH = "data/erb_sorocaba_limpo.csv"
RESULTS_DIR = "results"

# Cria os diretórios se não existirem
for directory in [RESULTS_DIR, "data"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Etapa 1: Carregar e Limpar os Dados --- 
def etapa_1_carregamento():
    print("\n" + "="*80)
    print("ETAPA 1: CARREGAMENTO E LIMPEZA DE DADOS")
    print("="*80 + "\n")
    
    print(f"Carregando dados brutos de: {INPUT_CSV_PATH}")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"AVISO: Arquivo {INPUT_CSV_PATH} não encontrado.")
        print("Você precisa adicionar o arquivo CSV original na pasta 'data'.")
        print("Pulando esta etapa.")
        
        # Para fins de teste, tente usar arquivo já processado
        if os.path.exists(OUTPUT_CSV_PATH):
            print(f"Encontrado arquivo limpo existente: {OUTPUT_CSV_PATH}")
            df = pd.read_csv(OUTPUT_CSV_PATH)
            print(f"Carregados {len(df)} registros.")
            return df
        else:
            print("Nenhum arquivo de dados encontrado. Encerrando.")
            return None
    
    df_limpo = load_and_clean_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    return df_limpo

# --- Etapa 2: Análise Exploratória --- 
def etapa_2_analise_exploratoria(df):
    if df is None or df.empty:
        print("Sem dados para análise exploratória. Pulando etapa.")
        return df
    
    print("\n" + "="*80)
    print("ETAPA 2: ANÁLISE EXPLORATÓRIA DE DADOS")
    print("="*80 + "\n")
    
    # Configura as opções de visualização antes de chamar a análise
    setup_visualization_options() 
    
    # Chama a função de análise, passando o DataFrame limpo e o diretório de resultados
    analise_exploratoria_erbs(df, RESULTS_DIR)
    
    return df

# --- Etapa 3: Processamento de Cobertura --- 
def etapa_3_processamento_cobertura(df):
    if df is None or df.empty:
        print("Sem dados para processamento de cobertura. Pulando etapa.")
        return None, None
    
    print("\n" + "="*80)
    print("ETAPA 3: PROCESSAMENTO DE COBERTURA DE SINAL")
    print("="*80 + "\n")
    
    # Configurar estilo visual
    configurar_estilo_visualizacao()
    
    # Copiar dataframe para não modificar o original
    df_cobertura = df.copy()
    
    # Converter tipos de dados
    print("Convertendo tipos de dados...")
    colunas_numericas = ['PotenciaTransmissorWatts', 'FreqTxMHz', 'GanhoAntena', 'Azimute']
    for coluna in colunas_numericas:
        if coluna in df_cobertura.columns:
            df_cobertura[coluna] = pd.to_numeric(df_cobertura[coluna], errors='coerce')
    
    # Filtrar para região de Sorocaba e principais operadoras
    print("Filtrando dados para região de interesse...")
    
    # Verificar se temos coordenadas
    if 'Latitude' not in df_cobertura.columns or 'Longitude' not in df_cobertura.columns:
        print("ERRO: Dados não contêm coordenadas (Latitude/Longitude). Impossível continuar.")
        return None, None
    
    # Definir região para filtrar (bbox de Sorocaba)
    sorocaba_bbox = [-23.60, -23.30, -47.65, -47.25]  # [lat_min, lat_max, lon_min, lon_max]
    
    # Filtrar por região
    df_cobertura = df_cobertura[
        (df_cobertura['Latitude'] >= sorocaba_bbox[0]) &
        (df_cobertura['Latitude'] <= sorocaba_bbox[1]) &
        (df_cobertura['Longitude'] >= sorocaba_bbox[2]) &
        (df_cobertura['Longitude'] <= sorocaba_bbox[3])
    ]
    
    # Normalizar nomes de operadoras (se a coluna existir)
    if 'NomeEntidade' in df_cobertura.columns:
        print("Padronizando nomes de operadoras...")
        
        # Mapeamento para nomes padronizados
        mapeamento_operadoras = {
            'CLARO': 'CLARO',
            'CLARO S.A.': 'CLARO',
            'OI': 'OI',
            'OI MÓVEL S.A.': 'OI',
            'VIVO': 'VIVO',
            'TELEFÔNICA BRASIL S.A.': 'VIVO',
            'TIM': 'TIM',
            'TIM S.A.': 'TIM'
        }
        
        # Função para mapear operadora
        def mapear_operadora(nome):
            for padrao, padronizado in mapeamento_operadoras.items():
                if padrao in nome.upper():
                    return padronizado
            return "OUTRA"
        
        # Criar coluna Operadora
        df_cobertura['Operadora'] = df_cobertura['NomeEntidade'].apply(mapear_operadora)
        
        # Filtrar principais operadoras
        df_cobertura = df_cobertura[df_cobertura['Operadora'].isin(['CLARO', 'OI', 'VIVO', 'TIM'])]
    
    print(f"Após filtragem: {len(df_cobertura)} ERBs na região de interesse")
    
    # Preencher valores ausentes
    print("Preenchendo valores ausentes...")
    
    # Potência
    if 'PotenciaTransmissorWatts' in df_cobertura.columns:
        media_potencia = df_cobertura['PotenciaTransmissorWatts'].median()
        if pd.isna(media_potencia) or media_potencia <= 0:
            media_potencia = 20.0
        df_cobertura.loc[pd.isna(df_cobertura['PotenciaTransmissorWatts']) | 
                         (df_cobertura['PotenciaTransmissorWatts'] <= 0), 
                         'PotenciaTransmissorWatts'] = media_potencia
    else:
        df_cobertura['PotenciaTransmissorWatts'] = 20.0  # Valor padrão
    
    # Ganho
    if 'GanhoAntena' in df_cobertura.columns:
        media_ganho = df_cobertura['GanhoAntena'].median()
        if pd.isna(media_ganho) or media_ganho <= 0:
            media_ganho = 16.0
        df_cobertura.loc[pd.isna(df_cobertura['GanhoAntena']) | 
                         (df_cobertura['GanhoAntena'] <= 0), 
                         'GanhoAntena'] = media_ganho
    else:
        df_cobertura['GanhoAntena'] = 16.0  # Valor padrão
    
    # Frequência
    if 'FreqTxMHz' in df_cobertura.columns:
        media_freq = df_cobertura['FreqTxMHz'].median()
        if pd.isna(media_freq) or media_freq <= 0:
            media_freq = 850.0
        df_cobertura.loc[pd.isna(df_cobertura['FreqTxMHz']) | 
                         (df_cobertura['FreqTxMHz'] <= 0), 
                         'FreqTxMHz'] = media_freq
    else:
        df_cobertura['FreqTxMHz'] = 850.0  # Valor padrão
    
    # Azimute
    if 'Azimute' not in df_cobertura.columns or df_cobertura['Azimute'].isna().sum() > 0:
        print("Gerando azimutes aleatórios para valores ausentes...")
        if 'Azimute' not in df_cobertura.columns:
            df_cobertura['Azimute'] = 0
        
        # Preencher valores ausentes de azimute com 0, 120, 240 (padrão para 3 setores)
        azimutes_padrao = [0, 120, 240]
        for i, row in df_cobertura[df_cobertura['Azimute'].isna()].iterrows():
            df_cobertura.loc[i, 'Azimute'] = azimutes_padrao[i % len(azimutes_padrao)]
    
    # Transformar os dados para GeoDataFrame
    print("Convertendo para GeoDataFrame...")
    geometria = [Point(xy) for xy in zip(df_cobertura['Longitude'], df_cobertura['Latitude'])]
    gdf_erb = gpd.GeoDataFrame(df_cobertura, geometry=geometria, crs="EPSG:4326")
    
    # Calcular tipo de área com base na densidade de ERBs
    print("Calculando tipo de área (urbana, suburbana, rural)...")
    gdf_erb = calcular_tipo_area(gdf_erb, raio=0.01)
    
    # Calcular EIRP (Potência Efetivamente Irradiada)
    print("Calculando EIRP (Potência Efetivamente Irradiada)...")
    gdf_erb['EIRP_dBm'] = gdf_erb.apply(
        lambda row: calcular_eirp(row['PotenciaTransmissorWatts'], row['GanhoAntena']), 
        axis=1
    )
    
    # Calcular raio de cobertura
    print("Calculando raio de cobertura...")
    gdf_erb['Raio_Cobertura_km'] = gdf_erb.apply(
        lambda row: calcular_raio_cobertura_aprimorado(
            row['EIRP_dBm'], row['FreqTxMHz'], row['tipo_area']
        ), 
        axis=1
    )
    
    # Calcular área de cobertura
    print("Calculando área de cobertura...")
    gdf_erb['Area_Cobertura_km2'] = gdf_erb.apply(
        lambda row: calcular_area_cobertura(row['Raio_Cobertura_km']), 
        axis=1
    )
    
    # Criar geometrias dos setores
    print("Criando geometrias de setores de cobertura...")
    gdf_erb['setor_geometria'] = gdf_erb.apply(
        lambda row: criar_setor_preciso(
            row['Latitude'], row['Longitude'], 
            row['Raio_Cobertura_km'], row['Azimute']
        ), 
        axis=1
    )
    
    # Criar GeoDataFrame apenas com setores
    gdf_setores = gpd.GeoDataFrame(
        gdf_erb[['Operadora', 'EIRP_dBm', 'Raio_Cobertura_km', 'Area_Cobertura_km2', 'tipo_area']],
        geometry=gdf_erb['setor_geometria'],
        crs="EPSG:4326"
    ).dropna(subset=['geometry'])
    
    print(f"Processamento concluído: {len(gdf_erb)} ERBs e {len(gdf_setores)} setores de cobertura.")
    
    return gdf_erb, gdf_setores

# --- Etapa 4: Visualizações Avançadas --- 
def etapa_4_visualizacoes(gdf_erb, gdf_setores):
    if gdf_erb is None or gdf_erb.empty:
        print("Sem dados para visualizações avançadas. Pulando etapa.")
        return
    
    print("\n" + "="*80)
    print("ETAPA 4: VISUALIZAÇÕES AVANÇADAS")
    print("="*80 + "\n")
    
    # Criar mapa de posicionamento das ERBs
    print("\n4.1. Mapa de posicionamento das ERBs")
    mapa_posicionamento = os.path.join(RESULTS_DIR, "mapa_posicionamento_erbs.png")
    criar_mapa_posicionamento(gdf_erb, mapa_posicionamento)
    
    # Criar mapa de cobertura por operadora
    if gdf_setores is not None and not gdf_setores.empty:
        print("\n4.2. Mapa de cobertura por operadora")
        mapa_cobertura = os.path.join(RESULTS_DIR, "mapa_cobertura_por_operadora.png")
        criar_mapa_cobertura_por_operadora(gdf_erb, gdf_setores, mapa_cobertura)
        
        print("\n4.3. Mapa de sobreposição de cobertura")
        mapa_sobreposicao = os.path.join(RESULTS_DIR, "mapa_sobreposicao_cobertura.png")
        criar_mapa_sobreposicao(gdf_erb, gdf_setores, mapa_sobreposicao)
    
    # Criar mapa de calor de potência
    print("\n4.4. Mapa de calor de potência EIRP")
    mapa_calor = os.path.join(RESULTS_DIR, "mapa_calor_potencia.png")
    criar_mapa_calor_potencia(gdf_erb, mapa_calor)
    
    # Criar mapa interativo Folium
    print("\n4.5. Mapa interativo Folium")
    mapa_folium = os.path.join(RESULTS_DIR, "mapa_interativo_erbs.html")
    criar_mapa_folium(gdf_erb, mapa_folium)
    
    print("\nTodas as visualizações foram geradas com sucesso.")

# --- Etapa 5: Análise de Grafos --- 
def etapa_5_analise_grafos(gdf_erb):
    if gdf_erb is None or gdf_erb.empty:
        print("Sem dados para análise de grafos. Pulando etapa.")
        return
    
    print("\n" + "="*80)
    print("ETAPA 5: ANÁLISE DE GRAFOS E REDES")
    print("="*80 + "\n")
    
    # Criar grafo de conectividade
    print("\n5.1. Criando grafo de conectividade...")
    grafo = criar_grafo_erb(gdf_erb, raio_conexao=2.0, ponderado=True)
    
    # Calcular métricas do grafo
    print("\n5.2. Calculando métricas do grafo...")
    metricas = calcular_metricas_grafo(grafo)
    
    # Mostrar métricas
    print("\nMétricas do grafo de ERBs:")
    for metrica, valor in metricas.items():
        print(f"  - {metrica}: {valor:.4f}" if isinstance(valor, float) else f"  - {metrica}: {valor}")
    
    # Salvar métricas em arquivo texto
    metricas_path = os.path.join(RESULTS_DIR, "metricas_grafo.txt")
    with open(metricas_path, 'w') as f:
        f.write("MÉTRICAS DO GRAFO DE ERBS\n")
        f.write("=========================\n\n")
        for metrica, valor in metricas.items():
            f.write(f"{metrica}: {valor}\n")
    print(f"Métricas salvas em: {metricas_path}")
    
    # Visualizar grafo
    print("\n5.3. Criando visualização do grafo...")
    grafo_viz_path = os.path.join(RESULTS_DIR, "grafo_conectividade_erbs.png")
    visualizar_grafo(grafo, grafo_viz_path, 
                    titulo="Grafo de Conectividade entre ERBs",
                    por_operadora=True)
    
    # Visualizar grafo colorido por centralidade
    grafo_central_path = os.path.join(RESULTS_DIR, "grafo_centralidade_erbs.png")
    visualizar_grafo(grafo, grafo_central_path, 
                    titulo="Centralidade de Intermediação no Grafo de ERBs",
                    por_operadora=False)
    
    # Criar grafo Voronoi
    print("\n5.4. Criando grafo baseado em diagrama de Voronoi...")
    voronoi_path = os.path.join(RESULTS_DIR, "grafo_voronoi_erbs.png")
    grafo_voronoi = criar_grafo_voronoi_erb(gdf_erb, voronoi_path)
    
    # Calcular métricas para grafo Voronoi
    print("\n5.5. Calculando métricas do grafo Voronoi...")
    metricas_voronoi = calcular_metricas_grafo(grafo_voronoi)
    
    # Salvar métricas em arquivo texto
    metricas_voronoi_path = os.path.join(RESULTS_DIR, "metricas_grafo_voronoi.txt")
    with open(metricas_voronoi_path, 'w') as f:
        f.write("MÉTRICAS DO GRAFO VORONOI DE ERBS\n")
        f.write("================================\n\n")
        for metrica, valor in metricas_voronoi.items():
            f.write(f"{metrica}: {valor}\n")
    print(f"Métricas Voronoi salvas em: {metricas_voronoi_path}")
    
    # Verificar se é possível criar modelo PyG
    try:
        converter_para_pyg(grafo)
        print("\nGrafo convertido para formato PyTorch Geometric com sucesso.")
    except Exception as e:
        print(f"\nNão foi possível converter o grafo para formato PyG: {e}")
    
    print("\nAnálise de grafos concluída com sucesso.")

# --- Função Principal --- 
def main():
    print("="*80)
    print("ANÁLISE DE ESTAÇÕES RÁDIO BASE (ERBs)")
    print("="*80)
    print(f"Diretório de resultados: {os.path.abspath(RESULTS_DIR)}")
    
    # Etapa 1: Carregar e limpar dados
    df = etapa_1_carregamento()
    
    # Etapa 2: Análise exploratória básica
    df = etapa_2_analise_exploratoria(df)
    
    # Etapa 3: Processamento de cobertura
    gdf_erb, gdf_setores = etapa_3_processamento_cobertura(df)
    
    # Etapa 4: Visualizações avançadas
    etapa_4_visualizacoes(gdf_erb, gdf_setores)
    
    # Etapa 5: Análise de grafos
    etapa_5_analise_grafos(gdf_erb)
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA")
    print("="*80)
    print(f"Todos os resultados foram salvos em: {os.path.abspath(RESULTS_DIR)}")
    print("Verifique o diretório para visualizar mapas, gráficos e relatórios gerados.")

if __name__ == "__main__":
    main()
