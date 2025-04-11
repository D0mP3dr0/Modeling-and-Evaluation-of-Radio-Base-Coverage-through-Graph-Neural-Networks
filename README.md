# Modelagem e Avaliação de Cobertura de Estações Rádio Base Através de GNN

Este projeto realiza análises avançadas de Estações Rádio Base (ERBs), desde a exploração de dados, 
processamento de cobertura, visualizações geoespaciais, até modelagem de grafos e redes neurais
para análise da conectividade entre as estações.

## Visão Geral

O projeto aborda o problema de análise e modelagem de cobertura de ERBs utilizando dados da Anatel,
com foco especial na área de Sorocaba-SP. Ele implementa:

1. **Processamento de Dados**: Limpeza e transformação dos dados brutos da Anatel.
2. **Modelagem de Cobertura**: Cálculo de EIRP, raio de cobertura e criação de geometrias de setores.
3. **Visualizações Geoespaciais**: Mapas interativos e estáticos de cobertura, sobreposição e heatmaps.
4. **Análise de Grafos**: Modelagem das ERBs como nós de um grafo, com arestas representando conectividade.
5. **Preparação para GNN**: Transformação em formato compatível com PyTorch Geometric para análises avançadas via GNN.

## Estrutura do Projeto

```
projeto_erb/
├── data/                  # Dados brutos e processados
│   └── README.md          # Instruções sobre os dados necessários
├── results/               # Resultados gerados (mapas, gráficos, métricas)
├── src/                   # Código fonte modularizado
│   ├── __init__.py        # Arquivo de inicialização do pacote
│   ├── analysis.py        # Funções de análise exploratória básica
│   ├── coverage_models.py # Modelos de cálculo de cobertura
│   ├── data_processing.py # Funções de processamento de dados
│   ├── graph_analysis.py  # Funções para análise de grafos e GNN
│   └── visualization.py   # Funções para visualizações avançadas
├── main.py                # Script principal para executar o fluxo completo
├── requirements.txt       # Dependências Python
└── README.md              # Este arquivo
```

## Funcionalidades

### 1. Processamento de Dados
- Limpeza de dados brutos da Anatel
- Filtragem para região de interesse
- Normalização de operadoras (Claro, Vivo, TIM, Oi)
- Preenchimento inteligente de valores ausentes

### 2. Modelagem de Cobertura
- Cálculo de EIRP (Potência Efetivamente Irradiada)
- Cálculo de raios de cobertura baseados em frequência, potência e ambiente
- Geração de setores de cobertura (polígonos) para cada ERB
- Classificação do tipo de área (urbana, suburbana, rural)

### 3. Visualizações Avançadas
- Mapa de posicionamento das ERBs por operadora
- Mapas de cobertura por operadora
- Mapa de sobreposição de coberturas
- Mapa de calor de potência EIRP
- Mapa interativo folium para navegação dinâmica

### 4. Análise de Grafos
- Construção de grafos de conectividade entre ERBs
- Cálculo de métricas de rede (centralidade, clustering, etc.)
- Visualização de grafos por operadora e centralidade
- Geração de grafo baseado em diagrama de Voronoi

### 5. Preparação para GNN
- Conversão para formato PyTorch Geometric
- Definição de features para nós (potência, ganho, etc.)
- Definição de features para arestas (distância, sobreposição)
- Estruturação para aplicação futura de GNN

## Como Usar

### Pré-requisitos
- Python 3.8+
- Dependências listadas em `requirements.txt`
- Arquivo CSV de licenciamento da Anatel (não incluído devido ao tamanho)

### Configuração

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/D0mP3dr0/Modeling-and-Evaluation-of-Radio-Base-Coverage-through-Graph-Neural-Networks.git
   cd projeto_erb
   ```

2. **Crie um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   # venv\Scripts\activate    # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Adicione os dados:**
   - Coloque o arquivo CSV de licenciamento da Anatel na pasta `data/`.
   - Renomeie para `csv_licenciamento_bruto.csv` ou ajuste o caminho em `main.py`.

### Execução

Para executar o projeto completo:
```bash
python main.py
```

O script processa todas as etapas sequencialmente e salva os resultados na pasta `results/`.

### Resultados Gerados

- **Estatísticas:** Métricas sobre ERBs e grafos
- **Mapas Estáticos:** Visualizações em PNG de alta resolução na pasta `results/`
- **Mapas Interativos:** Arquivo HTML com mapa interativo Folium
- **Grafos:** Visualizações de grafos de conectividade e Voronoi

## Dependências Principais

- **Análise de Dados:** pandas, numpy
- **Geoespacial:** geopandas, shapely, folium
- **Visualização:** matplotlib, seaborn, contextily
- **Grafos:** networkx, scipy
- **GNN:** torch, torch-geometric (opcional)

## Notas de Implementação

- O código está estruturado para ser modular e extensível.
- Funções detalhadas de documentação explicam os parâmetros e retornos.
- O sistema pode operar mesmo com dados parciais.
- A implementação de GNN requer PyTorch e PyTorch Geometric, mas as outras funcionalidades continuam operacionais sem eles.

## Autores e Contribuições

- **Desenvolvimento:** D0mP3dr0
- **Contribuições:** PRs são bem-vindos!
