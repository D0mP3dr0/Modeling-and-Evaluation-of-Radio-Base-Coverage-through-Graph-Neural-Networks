import os
from src.data_processing import load_and_clean_data
from src.analysis import setup_visualization_options, analise_exploratoria_erbs

# --- Configuração --- 
# Defina os caminhos aqui. Use caminhos relativos para melhor portabilidade.
# TODO: Substitua pelo caminho real do seu arquivo de dados brutos.
INPUT_CSV_PATH = "data/csv_licenciamento_bruto.csv"  
OUTPUT_CSV_PATH = "data/erb_sorocaba_limpo.csv"
RESULTS_DIR = "results"

# Cria o diretório de resultados se não existir
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Etapa 1: Carregar e Limpar os Dados --- 
print("Iniciando Etapa 1: Carregamento e Limpeza de Dados")
df_limpo = load_and_clean_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH)

# --- Etapa 2: Análise Exploratória --- 
if df_limpo is not None:
    print("\nIniciando Etapa 2: Análise Exploratória de Dados")
    # Configura as opções de visualização antes de chamar a análise
    setup_visualization_options() 
    # Chama a função de análise, passando o DataFrame limpo e o diretório de resultados
    analise_exploratoria_erbs(df_limpo, RESULTS_DIR)
else:
    print("\nProcessamento interrompido devido a erro no carregamento/limpeza dos dados.")

# --- Etapa 3: (Opcional) Análise de Grafos/GNN --- 
# TODO: Adicione aqui a chamada para suas funções de análise de grafos/GNN,
#       que você moverá do notebook original para um novo arquivo em src/
# Exemplo:
# from src.graph_analysis import build_and_analyze_graph
# if df_limpo is not None:
#     print("\nIniciando Etapa 3: Análise de Grafo")
#     build_and_analyze_graph(df_limpo, RESULTS_DIR)

print("\n--- Processo Concluído ---")
