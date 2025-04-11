# Projeto de Análise de ERBs (Estações Rádio Base)

Este projeto realiza a análise exploratória de dados de Estações Rádio Base (ERBs), 
focando na região de Sorocaba, com base em dados de licenciamento.

## Estrutura do Projeto

*   `data/`: Contém os dados brutos (`csv_licenciamento_bruto.csv` - **adicione este arquivo aqui!**) e processados (`erb_sorocaba_limpo.csv`).
*   `src/`: Contém o código fonte Python modularizado.
    *   `data_processing.py`: Funções para carregar e limpar os dados.
    *   `analysis.py`: Funções para análise exploratória e visualização.
    *   `(outros).py`: (TODO: Adicione módulos para análise de grafos/GNN se aplicável).
*   `notebooks/`: (Opcional) Pode conter notebooks Jupyter para exploração ou apresentação de resultados.
*   `results/`: Armazena os resultados gerados pela análise (mapas HTML, gráficos PNG, etc.).
*   `main.py`: Script principal para executar o fluxo completo do projeto.
*   `requirements.txt`: Lista de dependências Python.
*   `README.md`: Este arquivo.

## Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-seu-repositorio>
    cd projeto_erb
    ```
2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # ou
    # venv\Scripts\activate  # Windows
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: A instalação de `torch` e `torch-geometric` pode variar dependendo do seu sistema operacional e se você tem GPU. Consulte a documentação oficial do PyTorch e PyG.* 
    *Nota 2: `basemap` foi comentado em `requirements.txt` devido à complexidade. Se precisar, descomente e siga as instruções de instalação específicas para seu sistema.* 

4.  **Adicione o arquivo de dados brutos:**
    *   Coloque o seu arquivo CSV de licenciamento original dentro da pasta `data/` e certifique-se que o nome corresponde ao definido em `INPUT_CSV_PATH` dentro de `main.py` (atualmente `data/csv_licenciamento_bruto.csv`).

5.  **Execute o script principal:**
    ```bash
    python main.py
    ```

6.  **Verifique os resultados:**
    *   Os dados limpos estarão em `data/erb_sorocaba_limpo.csv`.
    *   Os mapas e gráficos estarão na pasta `results/`.

## Próximos Passos (TODO)

*   Revisar o restante do código do notebook original (`mba.ipynb`/`mba.txt`).
*   Mover a lógica de análise de grafos/GNN (se houver) para módulos Python em `src/`.
*   Integrar a chamada dessas novas funções no `main.py`.
*   Adicionar mais visualizações ou análises conforme necessário.
*   Refinar o `README.md` com mais detalhes sobre a análise e os resultados.
*   Configurar o `.gitignore` para evitar versionar dados grandes ou arquivos temporários.
