# Radio Base Stations (RBS) Analysis Tool

Uma ferramenta completa para análise de dados de estações rádio base, incluindo distribuição espacial, tendências tecnológicas, padrões temporais e estimativas de cobertura.

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
- **Integração com IA**: Suporte opcional para integração com API Claude da Anthropic para análises avançadas.

## Instalação

Clone o repositório e instale as dependências necessárias:

```bash
git clone https://github.com/seuusuario/analise-erb.git
cd analise-erb
pip install -r requirements.txt
```

Algumas dependências adicionais podem ser necessárias para análises específicas:
- Para previsão com Prophet: `pip install prophet`
- Para diagramas de Voronoi: `pip install geovoronoi`

## Uso

### Interface de Linha de Comando

A ferramenta pode ser executada a partir da linha de comando com várias opções:

```bash
python src/main.py --input <arquivo_entrada> --output <diretorio_saida> [opções]
```

Opções:
- `--input`, `-i`: Caminho para o arquivo de dados de entrada (CSV ou GeoJSON) [obrigatório]
- `--output`, `-o`: Caminho para o diretório de saída (padrão: 'results')
- `--all`, `-a`: Executa todas as análises disponíveis
- `--basic`, `-b`: Executa análise básica
- `--visualization`, `-v`: Cria visualizações
- `--graph`, `-g`: Executa análise de grafos
- `--coverage`, `-c`: Estima cobertura
- `--tech-frequency`, `-tf`: Executa análise de tecnologia e frequência
- `--temporal`, `-t`: Executa análise temporal avançada
- `--correlation`, `-cr`: Executa análise de correlação
- `--spatial`, `-s`: Executa análise espacial
- `--integration`, `-int`: Executa análise integrada
- `--prediction`, `-p`: Executa análise preditiva
- `--dashboard`, `-d`: Executa dashboard interativo
- `--report`, `-r`: Gera relatório abrangente
- `--test`: Executa testes unitários
- `--debug`: Ativa o modo de depuração

### Exemplo

```bash
# Executa todas as análises
python src/main.py --input data/erb_data.csv --output results --all

# Gera um relatório
python src/main.py --input data/erb_data.csv --output results --report

# Executa o dashboard interativo
python src/main.py --input data/erb_data.csv --dashboard
```

### API Python

A ferramenta também pode ser usada como uma biblioteca Python:

```python
import geopandas as gpd
from src.data_processing import load_and_process_data
from src.tech_frequency_analysis import run_tech_frequency_analysis
from src.report_generator import run_report_generation

# Carrega dados
gdf_rbs = load_and_process_data('data/erb_data.csv')

# Executa análise de tecnologia e frequência
run_tech_frequency_analysis(gdf_rbs, 'results/tech_analysis')

# Gera relatório
run_report_generation(gdf_rbs, 'results/reports')
```

## Formato dos Dados de Entrada

A ferramenta aceita dados em formato CSV ou GeoJSON. As seguintes colunas são usadas por várias análises:

### Colunas Obrigatórias
- Latitude/Longitude ou campo geometry (para GeoJSON)
- Operator: Nome da operadora de rede móvel
- Tecnologia: Tecnologia (2G, 3G, 4G, 5G)
- FreqTxMHz: Frequência de transmissão em MHz
- PotenciaTransmissorWatts: Potência do transmissor em watts

### Colunas Opcionais (usadas se disponíveis)
- AlturaAntena: Altura da antena em metros
- installation_date ou data_licenciamento: Data de instalação ou licenciamento
- EIRP_dBm: Potência Isotrópica Radiada Efetiva
- FrequencyBand: Categoria da banda de frequência

Se algumas colunas estiverem ausentes, a ferramenta tentará estimar ou gerar dados fictícios quando possível.

## Saída

A ferramenta gera várias saídas dependendo das análises executadas:
- Visualizações estáticas (arquivos PNG)
- Visualizações interativas (arquivos HTML)
- Arquivos de dados (CSV, GeoJSON)
- Relatórios abrangentes (PDF, Excel)
- Dashboard interativo (baseado na web)

## Integração com API Claude da Anthropic

Esta ferramenta suporta integração opcional com a API Claude da Anthropic para:

1. **Análise de Texto Avançada**: Extração de insights de descrições textuais e comentários.
2. **Interpretação de Resultados**: Geração de resumos e interpretações dos gráficos e análises.
3. **Recomendações de Otimização**: Sugestões para melhorar a cobertura da rede.
4. **Detecção de Anomalias**: Identificação de padrões incomuns nos dados.

### Configuração da API Claude

Para usar a API Claude:

1. Obtenha uma chave de API em [https://console.anthropic.com/](https://console.anthropic.com/)
2. Configure a chave em um arquivo `.env` na raiz do projeto:
   ```
   CLAUDE_API_KEY=sua_chave_api_aqui
   ```
3. Instale as dependências adicionais:
   ```bash
   pip install anthropic python-dotenv
   ```

### Usar a API na Análise

A funcionalidade Claude está disponível no módulo `claude_integration.py`:

```python
from src.claude_integration import get_coverage_insights, analyze_temporal_trends

# Obter insights de cobertura
insights = get_coverage_insights(gdf_rbs)

# Analisar tendências temporais
trends_analysis = analyze_temporal_trends(gdf_rbs)
```

## Módulos

- **data_processing.py**: Carregamento e pré-processamento de dados
- **analysis.py**: Análise estatística básica
- **visualization.py**: Funções básicas de visualização
- **graph_analysis.py**: Análise de rede das estações ERB
- **coverage_models.py**: Estimativa de área de cobertura
- **tech_frequency_analysis.py**: Análise de tecnologia e frequência
- **advanced_temporal_analysis.py**: Análise de padrões temporais
- **correlation_analysis.py**: Análise de correlação de variáveis
- **spatial_analysis.py**: Análise de padrões espaciais
- **integration_analysis.py**: Integração de análises temporais e tecnológicas
- **prediction_module.py**: Previsão de tendências futuras
- **dashboard_interactive.py**: Dashboard web interativo
- **report_generator.py**: Geração de relatórios automatizados
- **unit_tests.py**: Testes unitários para validação
- **main.py**: Ponto de entrada principal com CLI
- **claude_integration.py**: Integração com a API Claude (opcional)

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## Agradecimentos

- Provedores de dados de telecomunicações brasileiros
- Comunidades de código aberto de geoespacial e ciência de dados
- Anthropic por fornecer acesso à API Claude para processamento de linguagem natural
