# Radio Base Stations (RBS) Analysis Tool

Uma ferramenta completa para análise de dados de estações rádio base, incluindo distribuição espacial, tendências tecnológicas, padrões temporais e estimativas de cobertura.

## Sobre o Projeto

Esta ferramenta foi desenvolvida para auxiliar na análise de dados de Estações Rádio Base (ERBs), permitindo:

1. **Análise espacial** de distribuição de ERBs
2. **Análise de cobertura** baseada em modelos de propagação
3. **Análise de tecnologias e frequências** utilizadas
4. **Análise temporal** de implantação de ERBs
5. **Análise avançada de grafos** da rede de ERBs
6. **Documentação educacional** para entender o sistema

## Novidades (Versão 1.1.0)

- **Estrutura de Código Melhorada**: Reorganização do código para melhor manutenibilidade
- **Sistema de Logging**: Registro detalhado de operações para facilitar diagnósticos
- **Configuração Centralizada**: Parâmetros configuráveis em um único arquivo
- **Tratamento de Erros Robusto**: Mensagens de erro informativas e recuperação de falhas
- **Validação de Dados**: Verificação de integridade dos dados de entrada
- **Documentação Aprimorada**: Docstrings e comentários explicativos em todo o código
- **Módulo Educacional**: Documentação visual do fluxo de trabalho e análises

## Características

- **Análise Básica**: Resumo estatístico dos dados de ERBs.
- **Visualização**: Gera mapas e gráficos para visualizar a distribuição de ERBs.
- **Análise de Grafos**: Análise de rede das conexões e clusters de ERBs.
- **Modelagem de Cobertura**: Estima áreas de cobertura com base em frequência, potência e terreno.
- **Análise de Tecnologia e Frequência**: Analisa a relação entre frequências, tecnologias e potência.
- **Análise Temporal Avançada**: Estuda padrões de implantação ao longo do tempo com detecção de sazonalidade.
- **Análise de Correlação**: Identifica relações entre variáveis técnicas.
- **Análise Espacial**: Clustering espacial avançado e detecção de padrões.
- **Análise Integrada**: Análise combinada de aspectos temporais e tecnológicos.
- **Análise Preditiva**: Previsão de tendências futuras de implantação usando modelos de séries temporais.
- **Dashboard Interativo**: Dashboard de visualização interativa baseado na web.
- **Relatórios Automatizados**: Gera relatórios PDF e Excel abrangentes.
- **Documentação Educacional**: Fornece explicações visuais e interativas sobre as análises realizadas.

## Estrutura do Projeto

```
├── data/                    # Diretório para dados de entrada/saída
├── logs/                    # Registros de execução
├── results/                 # Resultados das análises
├── src/                     # Código-fonte do projeto
│   ├── __init__.py          # Inicialização do pacote
│   ├── config.py            # Configurações centralizadas
│   ├── data_processing.py   # Processamento de dados
│   ├── analysis.py          # Análise estatística básica
│   ├── visualization.py     # Visualizações básicas
│   ├── graph_analysis.py    # Análise de grafos
│   ├── coverage_models.py   # Modelagem de cobertura
│   ├── advanced_*.py        # Módulos de análise avançada
│   └── unit_tests.py        # Testes unitários
├── rbs_analysis.py          # Script principal
└── requirements.txt         # Dependências do projeto
```

## Instalação

### Requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)
- Dependências listadas em `requirements.txt`

### Passos para Instalação

Clone o repositório e instale as dependências necessárias:

```bash
git clone https://github.com/seuusuario/analise-erb.git
cd analise-erb
pip install -r requirements.txt
```

#### Dependências Opcionais

Algumas funcionalidades avançadas requerem pacotes adicionais, que estão incluídos no `requirements.txt` mas podem ser instalados separadamente:

- **Visualização 3D**: `pip install plotly dash`
- **Análise de GNN**: `pip install torch torch-geometric`
- **Geração de Fluxogramas**: `pip install graphviz`

## Uso

### Linha de Comando

Use o script `rbs_analysis.py` como ponto de entrada principal:

```bash
python rbs_analysis.py --input <arquivo_entrada> --output <diretorio_saida> [opções]
```

#### Exemplos

```bash
# Executa todas as análises
python rbs_analysis.py --input data/erb_data.csv --all

# Análise básica e visualização
python rbs_analysis.py --input data/erb_data.csv --basic --visualization

# Gera documentação educacional
python rbs_analysis.py --input data/erb_data.csv --educational-docs

# Executa análise avançada de grafos com campo de data específico
python rbs_analysis.py --input data/erb_data.csv --advanced-graph --time-field "data_instalacao"
```

#### Opções Disponíveis

Execute `python rbs_analysis.py --help` para ver todas as opções disponíveis.

### Como Importar como Biblioteca

O projeto também pode ser usado como uma biblioteca Python:

```python
from src import load_and_process_data, run_basic_analysis
from src.visualization import create_visualizations

# Carrega dados
gdf_rbs = load_and_process_data('data/erb_data.csv')

# Executa análise básica
run_basic_analysis(gdf_rbs, 'results/basic_analysis')
```

## Formato dos Dados de Entrada

A ferramenta aceita dados em formato CSV ou GeoJSON. As seguintes colunas são essenciais:

### Colunas Obrigatórias
- `Latitude`/`Longitude` ou campo `geometry` (para GeoJSON)

### Colunas Recomendadas
- `Operator`: Nome da operadora de rede móvel
- `Tecnologia`: Tecnologia (2G, 3G, 4G, 5G)
- `FreqTxMHz`: Frequência de transmissão em MHz
- `PotenciaTransmissorWatts`: Potência do transmissor em watts
- `installation_date`: Data de instalação (para análises temporais)

Se algumas colunas estiverem ausentes, a ferramenta tentará usar valores padrão ou gerar dados sintéticos quando apropriado.

## Configuração

O arquivo `src/config.py` centraliza todas as configurações do projeto. Você pode modificar:

- **Diretórios de Trabalho**: Locais para dados, resultados e logs
- **Parâmetros Padrão**: Valores default para dados ausentes
- **Configurações de Visualização**: Tamanhos, cores e estilos
- **Parâmetros de Análise**: Limiares e fatores para os modelos

## Logs e Depuração

Os logs são armazenados no diretório `logs/`. Para habilitar o modo de depuração:

```bash
python rbs_analysis.py --input data/erb_data.csv --debug
```

## Desenvolvimento

### Requisitos para Desenvolvimento

- Python 3.8+
- pytest (para testes unitários)
- flake8 e pylint (para verificação de estilo)

### Executando Testes

```bash
python -m pytest src/unit_tests.py
```

### Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## Agradecimentos

- Provedores de dados de telecomunicações brasileiros
- Comunidades de código aberto de geoespacial e ciência de dados
