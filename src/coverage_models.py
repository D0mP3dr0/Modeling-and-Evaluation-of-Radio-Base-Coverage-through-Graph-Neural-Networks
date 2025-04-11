"""
Módulo para cálculos de cobertura de rádio das ERBs.
Contém funções para calcular a potência efetivamente irradiada (EIRP),
raio de cobertura, área de cobertura e geometrias dos setores de cobertura.
"""

import numpy as np
from shapely.geometry import Polygon, Point
import geopandas as gpd
import pandas as pd

# Constantes padrão
SENSIBILIDADE_RECEPTOR = -100  # dBm
ANGULO_SETOR = 120  # graus

def calcular_eirp(potencia_watts, ganho_antena):
    """
    Calcula a Potência Efetivamente Irradiada (EIRP) em dBm.
    
    Args:
        potencia_watts (float): Potência do transmissor em Watts
        ganho_antena (float): Ganho da antena em dBi
        
    Returns:
        float: EIRP em dBm ou np.nan se a potência for inválida
    """
    try:
        potencia = float(potencia_watts)
        if potencia <= 0:
            return np.nan
    except (ValueError, TypeError):
        return np.nan
    
    # Conversão de Watts para dBm e adição do ganho
    return 10 * np.log10(potencia * 1000) + ganho_antena

def calcular_raio_cobertura_aprimorado(eirp, freq_mhz, tipo_area='urbana'):
    """
    Calcula o raio de cobertura com base no EIRP, frequência e tipo de área.
    
    Args:
        eirp (float): Potência efetivamente irradiada em dBm
        freq_mhz (float): Frequência em MHz
        tipo_area (str): Tipo de área ('urbana_densa', 'urbana', 'suburbana', 'rural')
        
    Returns:
        float: Raio de cobertura em quilômetros ou np.nan se valores forem inválidos
    """
    if np.isnan(eirp) or np.isnan(freq_mhz) or freq_mhz <= 0:
        return np.nan
    
    # Fatores de atenuação por tipo de área
    atenuacao = {'urbana_densa': 22, 'urbana': 16, 'suburbana': 12, 'rural': 8}
    
    # Cálculo do raio base (fórmula de Friis simplificada)
    raio_base = 10 ** ((eirp - SENSIBILIDADE_RECEPTOR - 32.44 - 20 * np.log10(freq_mhz)) / 20)
    
    # Ajuste pelo tipo de área
    raio_ajustado = raio_base * 0.7 / (atenuacao.get(tipo_area, 16) / 10)
    
    # Limite de acordo com a frequência (frequências mais altas têm alcance menor)
    limite_freq = min(7, 15000 / freq_mhz) if freq_mhz > 0 else 5
    
    return min(raio_ajustado, limite_freq)

def calcular_area_cobertura(raio, angulo=ANGULO_SETOR):
    """
    Calcula a área de cobertura com base no raio e ângulo do setor.
    
    Args:
        raio (float): Raio de cobertura em quilômetros
        angulo (float): Ângulo do setor em graus (padrão = 120)
        
    Returns:
        float: Área de cobertura em quilômetros quadrados
    """
    if np.isnan(raio):
        return np.nan
    
    # Área do setor como fração de um círculo completo
    return (np.pi * raio**2 * angulo) / 360

def criar_setor_preciso(lat, lon, raio, azimute, angulo=ANGULO_SETOR, resolucao=30):
    """
    Cria um polígono representando o setor de cobertura da antena.
    
    Args:
        lat (float): Latitude do ponto central (ERB)
        lon (float): Longitude do ponto central (ERB)
        raio (float): Raio de cobertura em quilômetros
        azimute (float): Direção da antena em graus (0-360)
        angulo (float): Ângulo do setor em graus (padrão = 120)
        resolucao (int): Número de pontos a usar para o arco do setor
        
    Returns:
        Polygon or None: Geometria do setor ou None se raio/azimute for inválido
    """
    if np.isnan(raio) or np.isnan(azimute) or raio <= 0:
        return None
    
    # Ajustar azimute para o sistema de coordenadas geográficas
    azimute_rad = np.radians((450 - float(azimute)) % 360)
    metade_angulo = np.radians(angulo / 2)
    
    # Criar pontos para o polígono do setor
    pontos = [(lon, lat)]  # Ponto central
    
    # Criar arco com múltiplos pontos
    for i in range(resolucao + 1):
        angulo_atual = azimute_rad - metade_angulo + (i * 2 * metade_angulo / resolucao)
        
        # Adicionar pontos em diferentes distâncias para melhor definição
        for j in [0.8, 0.9, 0.95, 1.0]:
            dist = raio * j
            # Converter distância para graus decimais
            dx = dist * np.cos(angulo_atual) / 111.32  # 111.32 km = 1 grau de longitude no equador
            dy = dist * np.sin(angulo_atual) / (111.32 * np.cos(np.radians(lat)))  # Ajuste para latitude
            pontos.append((lon + dx, lat + dy))
    
    # Fechar o polígono
    pontos.append((lon, lat))
    
    try:
        return Polygon(pontos)
    except:
        return None

def calcular_tipo_area(gdf_erb, raio=0.01):
    """
    Estima o tipo de área (urbana, rural, etc.) com base na densidade de ERBs.
    
    Args:
        gdf_erb (GeoDataFrame): GeoDataFrame com as ERBs
        raio (float): Raio de busca (em graus decimais)
        
    Returns:
        GeoDataFrame: O mesmo GeoDataFrame com coluna 'tipo_area' adicionada
    """
    # Função para contar ERBs próximas
    def calcular_densidade_erb(ponto, gdf, raio):
        buffer = ponto.buffer(raio)
        return len(gdf[gdf.geometry.intersects(buffer)])
    
    # Calcular densidade para cada ERB
    gdf_erb = gdf_erb.copy()
    gdf_erb['densidade_erb'] = gdf_erb.geometry.apply(lambda p: calcular_densidade_erb(p, gdf_erb, raio))
    
    # Classificar o tipo de área com base na densidade
    gdf_erb['tipo_area'] = pd.cut(
        gdf_erb['densidade_erb'], 
        bins=[0, 3, 6, 10, float('inf')], 
        labels=['rural', 'suburbana', 'urbana', 'urbana_densa']
    )
    gdf_erb['tipo_area'] = gdf_erb['tipo_area'].fillna('urbana')
    
    return gdf_erb
