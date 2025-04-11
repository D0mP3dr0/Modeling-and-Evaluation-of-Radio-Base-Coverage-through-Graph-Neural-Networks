# Notebook para Google Colab - Análise de ERBs

Este diretório contém um script Python (`rbs_analysis_colab.py`) que pode ser facilmente convertido em um notebook Google Colab para análise de Estações Rádio Base (ERBs).

## Como utilizar no Google Colab

Existem duas formas de utilizar o notebook no Google Colab:

### Método 1: Upload do arquivo Python

1. Acesse o [Google Colab](https://colab.research.google.com/)
2. No menu, clique em `File` > `Upload notebook`
3. Selecione o arquivo `rbs_analysis_colab.py` do seu computador
4. O Google Colab converterá automaticamente o arquivo Python em um notebook Jupyter

### Método 2: Conversão direta no Google Colab

1. Acesse o [Google Colab](https://colab.research.google.com/)
2. Abra um novo notebook
3. No menu, clique em `File` > `Upload notebook`
4. Selecione a aba "GitHub"
5. Cole a URL do repositório: `https://github.com/D0mP3dr0/Modeling-and-Evaluation-of-Radio-Base-Coverage-through-Graph-Neural-Networks`
6. Selecione o arquivo `notebooks/rbs_analysis_colab.py`

## Sobre o Notebook

Este notebook realiza uma análise completa de dados de Estações Rádio Base, incluindo:

1. Configuração do ambiente e dependências
2. Carregamento e processamento de dados
3. Visualizações básicas (distribuição geográfica, operadoras)
4. Análise de grafos (conectividade entre ERBs)
5. Estimativa de cobertura das ERBs
6. Mapa interativo para visualização da cobertura

## Requisitos do Arquivo de Dados

Para executar o notebook completo, você precisará fazer upload do arquivo `csv_licenciamento_bruto.csv.csv` quando solicitado. Este arquivo contém os dados das ERBs a serem analisados.

## Funções de Fallback

O notebook inclui mecanismos de fallback para lidar com erros comuns, como:
- Carregamento manual dos dados caso o módulo de processamento falhe
- Criação de um grafo simplificado caso o módulo de análise de grafos falhe
- Adaptação para trabalhar com diferentes formatos de dados de entrada

## Problemas Conhecidos

- Algumas visualizações podem exigir adaptações dependendo da estrutura exata do arquivo de dados
- As funções que utilizam o módulo `folium` podem não exibir corretamente em todas as versões do Colab
- Em caso de erros relacionados a módulos ausentes, execute manualmente o comando de instalação: `!pip install numpy pandas scipy geopandas shapely pyproj matplotlib seaborn folium plotly scikit-learn networkx graphviz` 