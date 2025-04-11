"""
Módulo para modelagem e análise de ERBs utilizando teoria dos grafos e GNNs.
Este módulo permite transformar dados de ERBs em grafos, calcular métricas de grafos e 
analisar a conectividade entre as estações.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import seaborn as sns
import os
from scipy.spatial import Voronoi

# Verificar se PyTorch e PyG estão disponíveis
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data as PyGData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch e/ou PyTorch Geometric não encontrados. Funcionalidades GNN não estarão disponíveis.")

def criar_grafo_erb(gdf_erb, raio_conexao=3.0, ponderado=True):
    """
    Cria um grafo NetworkX onde os nós são ERBs e as arestas representam a conectividade.
    
    Args:
        gdf_erb (GeoDataFrame): GeoDataFrame contendo os dados das ERBs
        raio_conexao (float): Raio máximo (km) para considerar duas ERBs conectadas
        ponderado (bool): Se True, adiciona pesos às arestas baseados na distância
        
    Returns:
        G (nx.Graph): Grafo NetworkX representando a rede de ERBs
    """
    print(f"Criando grafo de ERBs com raio de conexão de {raio_conexao} km...")
    
    # Criar grafo vazio
    G = nx.Graph()
    
    # Adicionar nós (ERBs)
    for idx, row in gdf_erb.iterrows():
        # Definir atributos do nó
        node_attrs = {
            'pos': (row['Longitude'], row['Latitude']),
            'operadora': row.get('Operadora', 'N/A'),
            'tecnologia': row.get('Tecnologia', 'N/A'),
            'freq_tx': row.get('FreqTxMHz', 0),
            'potencia': row.get('PotenciaTransmissorWatts', 0),
            'ganho': row.get('GanhoAntena', 0),
            'raio_cobertura': row.get('Raio_Cobertura_km', 0)
        }
        
        # Adicionar ao grafo
        G.add_node(idx, **node_attrs)
    
    # Adicionar arestas (conexões entre ERBs)
    nós = list(G.nodes())
    contador_arestas = 0
    
    for i in range(len(nós)):
        for j in range(i+1, len(nós)):
            node_i = nós[i]
            node_j = nós[j]
            
            # Obter posições
            pos_i = G.nodes[node_i]['pos']
            pos_j = G.nodes[node_j]['pos']
            
            # Calcular distância aproximada em km (distância Haversine)
            lon1, lat1 = pos_i
            lon2, lat2 = pos_j
            
            # Converter para radianos
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            
            # Fórmula de Haversine
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distancia = 6371 * c  # Raio da Terra em km
            
            # Se a distância for menor que o raio de conexão, adicionar aresta
            if distancia <= raio_conexao:
                # Definir peso da aresta (inverso da distância)
                if ponderado:
                    peso = 1.0 / max(0.1, distancia)  # Evita divisão por zero
                else:
                    peso = 1.0
                
                # Obter raios de cobertura dos nós
                raio_i = G.nodes[node_i]['raio_cobertura']
                raio_j = G.nodes[node_j]['raio_cobertura']
                
                # Verificar sobreposição de coberturas
                sobreposicao = False
                if raio_i > 0 and raio_j > 0:
                    sobreposicao = distancia < (raio_i + raio_j)
                
                # Adicionar aresta com atributos
                G.add_edge(node_i, node_j, weight=peso, distance=distancia, overlap=sobreposicao)
                contador_arestas += 1
    
    print(f"Grafo criado com {len(G.nodes())} nós e {contador_arestas} arestas.")
    return G

def calcular_metricas_grafo(G):
    """
    Calcula métricas de rede para o grafo de ERBs.
    
    Args:
        G (nx.Graph): Grafo NetworkX de ERBs
        
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    print("Calculando métricas do grafo...")
    
    metricas = {}
    
    # Número de nós e arestas
    metricas['num_nos'] = G.number_of_nodes()
    metricas['num_arestas'] = G.number_of_edges()
    
    # Densidade do grafo
    metricas['densidade'] = nx.density(G)
    
    # Componentes conectados
    componentes = list(nx.connected_components(G))
    metricas['num_componentes'] = len(componentes)
    metricas['tamanho_maior_componente'] = len(max(componentes, key=len))
    
    # Distância média e diâmetro
    if nx.is_connected(G):
        metricas['distancia_media'] = nx.average_shortest_path_length(G, weight='distance')
        metricas['diametro'] = nx.diameter(G, e=None, weight='distance')
    else:
        # Calcular apenas para o maior componente
        maior_componente = max(componentes, key=len)
        subgrafo = G.subgraph(maior_componente).copy()
        metricas['distancia_media'] = nx.average_shortest_path_length(subgrafo, weight='distance')
        metricas['diametro'] = nx.diameter(subgrafo, e=None, weight='distance')
    
    # Coeficiente de clustering
    metricas['clustering_medio'] = nx.average_clustering(G)
    
    # Centralidade
    betweenness = nx.betweenness_centrality(G, weight='distance')
    metricas['betweenness_max'] = max(betweenness.values())
    metricas['betweenness_meio'] = np.median(list(betweenness.values()))
    
    # Adicionar centralidades como atributos dos nós
    nx.set_node_attributes(G, betweenness, 'betweenness')
    
    # Distribuição de grau
    graus = [d for n, d in G.degree()]
    metricas['grau_min'] = min(graus)
    metricas['grau_max'] = max(graus)
    metricas['grau_medio'] = sum(graus) / len(graus)
    
    print("Métricas calculadas com sucesso.")
    return metricas

def visualizar_grafo(G, caminho_saida, titulo="Grafo de Conectividade entre ERBs", 
                     por_operadora=True, mostrar_pesos=False):
    """
    Visualiza o grafo de ERBs.
    
    Args:
        G (nx.Graph): Grafo NetworkX de ERBs
        caminho_saida (str): Caminho para salvar a visualização
        titulo (str): Título do gráfico
        por_operadora (bool): Se True, colorir nós por operadora
        mostrar_pesos (bool): Se True, mostrar pesos das arestas
    """
    print("Criando visualização do grafo...")
    
    plt.figure(figsize=(14, 10))
    
    # Obter posições dos nós
    pos = nx.get_node_attributes(G, 'pos')
    
    # Definir cores para operadoras
    cores_operadoras = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    if por_operadora:
        # Agrupar nós por operadora
        operadoras = nx.get_node_attributes(G, 'operadora')
        grupos_operadoras = {}
        
        for node, operadora in operadoras.items():
            if operadora not in grupos_operadoras:
                grupos_operadoras[operadora] = []
            grupos_operadoras[operadora].append(node)
        
        # Desenhar nós por operadora
        for operadora, nós in grupos_operadoras.items():
            cor = cores_operadoras.get(operadora, '#CCCCCC')
            nx.draw_networkx_nodes(G, pos, nodelist=nós, node_color=cor, 
                                  node_size=100, alpha=0.8, label=operadora)
    else:
        # Colorir por centralidade
        betweenness = nx.get_node_attributes(G, 'betweenness')
        
        if betweenness:
            node_colors = [betweenness.get(node, 0) for node in G.nodes()]
            nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         node_size=100, alpha=0.8, cmap=plt.cm.viridis)
            plt.colorbar(nodes, label='Centralidade de Intermediação')
        else:
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', 
                                  node_size=100, alpha=0.8)
    
    # Desenhar arestas
    if mostrar_pesos:
        # Obter pesos das arestas
        weights = nx.get_edge_attributes(G, 'weight').values()
        # Normalizar espessuras
        edge_widths = [w * 2 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    else:
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Adicionar rótulos apenas para nós importantes
    betweenness = nx.get_node_attributes(G, 'betweenness')
    nos_importantes = []
    
    if betweenness:
        # Selecionar os 5% dos nós mais importantes
        threshold = np.percentile(list(betweenness.values()), 95)
        nos_importantes = [n for n, c in betweenness.items() if c >= threshold]
        
        labels = {n: f"{n}" for n in nos_importantes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    # Adicionar legenda para operadoras
    if por_operadora:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cor, markersize=10, 
                            label=operadora) for operadora, cor in cores_operadoras.items()]
        plt.legend(handles=handles, title="Operadoras", loc='best')
    
    # Adicionar título e informações
    plt.title(titulo, fontsize=16)
    plt.text(0.01, 0.01, f"Nós: {G.number_of_nodes()}, Arestas: {G.number_of_edges()}", 
            transform=plt.gca().transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualização do grafo salva em {caminho_saida}")

def converter_para_pyg(G):
    """
    Converte um grafo NetworkX para formato PyTorch Geometric.
    
    Args:
        G (nx.Graph): Grafo NetworkX de ERBs
        
    Returns:
        PyGData or None: Objeto PyG Data se PyTorch estiver disponível, senão None
    """
    if not TORCH_AVAILABLE:
        print("PyTorch ou PyTorch Geometric não encontrado. Não é possível converter para formato PyG.")
        return None
    
    print("Convertendo grafo para formato PyTorch Geometric...")
    
    # Mapear nós para índices contíguos
    nos_para_idx = {n: i for i, n in enumerate(G.nodes())}
    
    # Criar lista de arestas
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        edge_index.append([nos_para_idx[u], nos_para_idx[v]])
        edge_index.append([nos_para_idx[v], nos_para_idx[u]])  # Adicionar aresta em ambas direções
        
        # Atributos da aresta
        peso = data.get('weight', 1.0)
        distancia = data.get('distance', 0.0)
        sobreposicao = 1.0 if data.get('overlap', False) else 0.0
        
        edge_attr.append([peso, distancia, sobreposicao])
        edge_attr.append([peso, distancia, sobreposicao])  # Duplicar para a aresta inversa
    
    # Converter para tensor PyTorch
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Criar features para os nós
    features = []
    y = []  # Classes/rótulos (exemplo: operadora como rótulo)
    
    operadoras_map = {'CLARO': 0, 'OI': 1, 'VIVO': 2, 'TIM': 3, 'N/A': 4}
    
    for node in G.nodes():
        attrs = G.nodes[node]
        
        # Feature vector: [potencia, ganho, raio_cobertura, freq_tx, centralidade]
        potencia = float(attrs.get('potencia', 0))
        ganho = float(attrs.get('ganho', 0))
        raio = float(attrs.get('raio_cobertura', 0))
        freq = float(attrs.get('freq_tx', 0))
        betweenness = float(attrs.get('betweenness', 0))
        
        features.append([potencia, ganho, raio, freq, betweenness])
        
        # Usar operadora como rótulo de exemplo
        operadora = attrs.get('operadora', 'N/A')
        y.append(operadoras_map.get(operadora, 4))
    
    # Converter para tensores PyTorch
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    # Criar objeto PyG Data
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(f"Grafo convertido para PyG: {data}")
    return data

def criar_grafo_voronoi_erb(gdf_erb, caminho_saida, bound_factor=0.1):
    """
    Cria um grafo baseado no diagrama de Voronoi das ERBs.
    
    Args:
        gdf_erb (GeoDataFrame): GeoDataFrame contendo os dados das ERBs
        caminho_saida (str): Caminho para salvar a visualização
        bound_factor (float): Fator para estender os limites do diagrama
        
    Returns:
        nx.Graph: Grafo NetworkX baseado nas células de Voronoi
    """
    print("Criando grafo baseado em diagrama de Voronoi...")
    
    # Extrair pontos
    pontos = np.array([(r['Longitude'], r['Latitude']) for _, r in gdf_erb.iterrows()])
    
    # Calcular limites com margem
    x_min, y_min = np.min(pontos, axis=0) - bound_factor
    x_max, y_max = np.max(pontos, axis=0) + bound_factor
    
    # Adicionar pontos nos cantos para fechar o diagrama
    far_points = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max]
    ])
    
    all_points = np.vstack([pontos, far_points])
    
    # Calcular diagrama de Voronoi
    vor = Voronoi(all_points)
    
    # Criar grafo
    G = nx.Graph()
    
    # Adicionar nós de ERBs
    for i, (_, row) in enumerate(gdf_erb.iterrows()):
        G.add_node(i, pos=(row['Longitude'], row['Latitude']), 
                  operadora=row.get('Operadora', 'N/A'),
                  raio_cobertura=row.get('Raio_Cobertura_km', 0))
    
    # Adicionar arestas baseadas em células de Voronoi adjacentes
    for i, j in vor.ridge_points:
        # Ignorar arestas ligadas aos pontos nos cantos
        if i >= len(pontos) or j >= len(pontos):
            continue
        
        # Calcular comprimento da aresta (distância entre ERBs)
        p1 = all_points[i]
        p2 = all_points[j]
        
        # Converter para radianos
        lon1, lat1 = p1
        lon2, lat2 = p2
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Fórmula de Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distancia = 6371 * c  # Raio da Terra em km
        
        # Adicionar aresta
        G.add_edge(i, j, weight=1.0/max(0.1, distancia), distance=distancia)
    
    # Visualizar
    plt.figure(figsize=(14, 10))
    
    # Obter posições dos nós
    pos = nx.get_node_attributes(G, 'pos')
    
    # Definir cores para operadoras
    cores_operadoras = {
        'CLARO': '#E02020',
        'OI': '#FFD700',
        'VIVO': '#9932CC',
        'TIM': '#0000CD',
        'N/A': '#CCCCCC'
    }
    
    # Agrupar nós por operadora
    operadoras = nx.get_node_attributes(G, 'operadora')
    grupos_operadoras = {}
    
    for node, operadora in operadoras.items():
        if operadora not in grupos_operadoras:
            grupos_operadoras[operadora] = []
        grupos_operadoras[operadora].append(node)
    
    # Desenhar células de Voronoi
    voronoi_plot_2d(vor, show_points=False, show_vertices=False, 
                   line_colors='gray', line_width=0.5, line_alpha=0.4, ax=plt.gca())
    
    # Desenhar nós por operadora
    for operadora, nós in grupos_operadoras.items():
        cor = cores_operadoras.get(operadora, '#CCCCCC')
        nx.draw_networkx_nodes(G, pos, nodelist=nós, node_color=cor, 
                              node_size=100, alpha=0.8, label=operadora)
    
    # Desenhar arestas
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.4, edge_color='gray')
    
    # Adicionar legenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cor, markersize=10, 
                        label=operadora) for operadora, cor in cores_operadoras.items()]
    plt.legend(handles=handles, title="Operadoras", loc='best')
    
    plt.title("Grafo de ERBs baseado em células de Voronoi", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafo de Voronoi criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    print(f"Visualização do grafo de Voronoi salva em {caminho_saida}")
    
    return G
