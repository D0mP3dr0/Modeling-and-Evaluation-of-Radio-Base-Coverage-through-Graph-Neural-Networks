# Pasta de Resultados

Esta pasta armazena todos os resultados gerados pelo processamento do projeto. Os arquivos são gerados automaticamente ao executar `main.py`.

## Arquivos Gerados

### Mapas e Visualizações Estáticas

* `mapa_posicionamento_erbs.png` - Mapa mostrando a posição de todas as ERBs por operadora
* `mapa_cobertura_por_operadora.png` - Mapa com 4 painéis mostrando a cobertura de cada operadora
* `mapa_sobreposicao_cobertura.png` - Mapa de sobreposição de cobertura entre operadoras
* `mapa_calor_potencia.png` - Mapa de calor mostrando a intensidade de EIRP
* `plots_distribuicao_altura_tecnologia.png` - Gráficos estatísticos de distribuição de altura e tecnologia

### Visualizações de Grafos

* `grafo_conectividade_erbs.png` - Grafo de conectividade entre ERBs colorido por operadora
* `grafo_centralidade_erbs.png` - Grafo colorido por centralidade de intermediação
* `grafo_voronoi_erbs.png` - Grafo baseado em diagrama de Voronoi

### Mapas Interativos

* `mapa_interativo_erbs.html` - Mapa interativo Folium para navegação dinâmica

### Relatórios e Métricas

* `metricas_grafo.txt` - Métricas calculadas para o grafo de conectividade
* `metricas_grafo_voronoi.txt` - Métricas calculadas para o grafo Voronoi

## Interpretação dos Resultados

* **Mapas de Cobertura**: Mostram o alcance estimado dos sinais de cada operadora
* **Mapa de Sobreposição**: Indica áreas com cobertura de múltiplas operadoras (redundância)
* **Métricas de Grafo**: Fornecem insights sobre a conectividade da rede
  * `densidade`: Proporção de conexões existentes vs. possíveis
  * `clustering_medio`: Mede a tendência de formação de grupos conectados
  * `betweenness_max`: Identifica nós críticos para o fluxo na rede

## Utilização dos Resultados

Estes resultados podem ser utilizados para:

* Planejamento de novas ERBs
* Análise de áreas com cobertura insuficiente
* Identificação de ERBs críticas para resiliência da rede
* Modelagem de propagação de sinal
* Preparação para análises mais avançadas com GNN 