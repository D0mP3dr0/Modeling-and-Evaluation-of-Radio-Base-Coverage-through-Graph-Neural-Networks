"""
Módulo para visualizações avançadas de dados de ERBs.
Contém funções para criar mapas temáticos, visualizações de cobertura de sinal,
mapas de calor e outras representações visuais das ERBs e suas características.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import contextily as ctx
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects
from matplotlib.offsetbox import AnchoredSizeBar
import matplotlib.font_manager as fm
from datetime import datetime
import os
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

# Configuração padrão de cores para operadoras
CORES_OPERADORAS = {
    'CLARO': '#E02020',
    'OI': '#FFD700',
    'VIVO': '#9932CC',
    'TIM': '#0000CD'
}

def configurar_estilo_visualizacao():
    """Configura o estilo padrão para todas as visualizações."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2
    })

def adicionar_elementos_cartograficos(ax, titulo, fonte="Dados: Anatel (ERBs), OpenStreetMap (Base)"):
    """
    Adiciona elementos cartográficos padrão a um mapa matplotlib.
    
    Args:
        ax: Matplotlib axis onde os elementos serão adicionados
        titulo: Título do mapa
        fonte: Texto com a fonte dos dados
    """
    # Título
    ax.set_title(titulo, fontweight='bold', pad=20)
    
    # Escala
    scalebar = AnchoredSizeBar(ax.transData, 5000, '5 km', 'lower right', pad=0.5,
                              color='black', frameon=True, size_vertical=100)
    ax.add_artist(scalebar)
    
    # Norte
    x, y, arrow_length = 0.06, 0.12, 0.08
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=2, headwidth=8),
                ha='center', va='center', fontsize=14,
                xycoords='figure fraction',
                fontweight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Fonte e data
    ax.annotate(f"{fonte}\nGerado em: {datetime.now().strftime('%d/%m/%Y')}",
                xy=(0.01, 0.01), xycoords='figure fraction', fontsize=8,
                ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Grade e outros elementos
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.title.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='white')])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def criar_legenda_personalizada(ax, cores, titulo="Legenda"):
    """
    Cria uma legenda personalizada com as cores das operadoras.
    
    Args:
        ax: Matplotlib axis onde a legenda será adicionada
        cores: Dicionário com os nomes das operadoras e suas cores
        titulo: Título da legenda
    """
    elementos_legenda = [Patch(facecolor=cor, edgecolor='k', alpha=0.7, label=operadora)
                         for operadora, cor in cores.items()]
    legenda = ax.legend(handles=elementos_legenda, title=titulo, loc='upper right',
                      frameon=True, framealpha=0.8, edgecolor='k')
    legenda.get_frame().set_linewidth(0.8)
    plt.setp(legenda.get_title(), fontweight='bold')

def criar_mapa_posicionamento(gdf_erb, caminho_saida, cores_operadoras=CORES_OPERADORAS):
    """
    Cria um mapa mostrando a posição das ERBs por operadora.
    
    Args:
        gdf_erb: GeoDataFrame com as ERBs
        caminho_saida: Caminho onde salvar o mapa
        cores_operadoras: Dicionário de cores para cada operadora
    """
    print("Criando mapa de posicionamento das ERBs...")
    
    # Reprojetar para Web Mercator (EPSG:3857) para uso com contextily
    gdf_erb_3857 = gdf_erb.to_crs(epsg=3857)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Calcular tamanhos baseados no EIRP
    min_eirp = gdf_erb_3857['EIRP_dBm'].min()
    max_eirp = gdf_erb_3857['EIRP_dBm'].max()
    tamanhos = ((gdf_erb_3857['EIRP_dBm'] - min_eirp) / (max_eirp - min_eirp) * 120 + 30)
    
    # Plotar cada operadora com uma cor diferente
    for operadora, cor in cores_operadoras.items():
        subset = gdf_erb_3857[gdf_erb_3857['Operadora'] == operadora]
        if subset.empty:
            continue
            
        subset_tamanhos = tamanhos.loc[subset.index]
        
        # Adicionar efeito de glow
        ax.scatter(subset.geometry.x, subset.geometry.y,
                  s=subset_tamanhos * 1.5, color=cor, alpha=0.2, edgecolor='none')
                  
        # Adicionar pontos
        ax.scatter(subset.geometry.x, subset.geometry.y,
                  s=subset_tamanhos, color=cor, alpha=0.8, edgecolor='white', 
                  linewidth=0.5, label=operadora)
    
    # Adicionar mapa base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Adicionar elementos cartográficos
    adicionar_elementos_cartograficos(ax, 'Posicionamento das Estações Rádio Base (ERBs) por Operadora')
    
    # Adicionar legenda
    criar_legenda_personalizada(ax, cores_operadoras, "Operadoras")
    
    # Adicionar nota sobre tamanho dos pontos
    ax.annotate('Tamanho dos pontos proporcional à potência irradiada (EIRP)',
                xy=(0.5, 0.02), xycoords='figure fraction', ha='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Salvar figura
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Mapa de posicionamento salvo em {caminho_saida}")

def criar_mapa_cobertura_por_operadora(gdf_erb, gdf_setores, caminho_saida, cores_operadoras=CORES_OPERADORAS):
    """
    Cria um mapa com 4 subplots mostrando a cobertura de cada operadora.
    
    Args:
        gdf_erb: GeoDataFrame com as ERBs
        gdf_setores: GeoDataFrame com os setores de cobertura
        caminho_saida: Caminho onde salvar o mapa
        cores_operadoras: Dicionário de cores para cada operadora
    """
    print("Criando mapa de cobertura por operadora...")
    
    # Reprojetar para Web Mercator
    gdf_erb_3857 = gdf_erb.to_crs(epsg=3857)
    gdf_setores_3857 = gdf_setores.to_crs(epsg=3857)
    
    # Criar figura com 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    operadoras = ['CLARO', 'OI', 'VIVO', 'TIM']
    
    for i, operadora in enumerate(operadoras):
        ax = axes[i]
        
        # Filtrar dados para a operadora atual
        subset_erb = gdf_erb_3857[gdf_erb_3857['Operadora'] == operadora]
        subset_setores = gdf_setores_3857[gdf_setores_3857['Operadora'] == operadora]
        
        if subset_setores.empty:
            ax.set_title(f'Sem dados para {operadora}', fontsize=16)
            continue
            
        # Obter cor para a operadora
        cor_base = cores_operadoras[operadora]
        cor_setores = f"{cor_base}66"  # Cor com transparência (alfa = 66)
        
        # Plotar setores de cobertura
        subset_setores.plot(ax=ax, color=cor_setores, edgecolor=cor_base, linewidth=0.3, alpha=0.6)
        
        # Plotar ERBs
        subset_erb.plot(ax=ax, color=cor_base, markersize=50, marker='o',
                       edgecolor='white', linewidth=0.7, alpha=0.9)
        
        # Adicionar mapa base
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        # Adicionar elementos cartográficos
        adicionar_elementos_cartograficos(
            ax, 
            f'Cobertura da Operadora {operadora}'
        )
        
        # Adicionar informações estatísticas
        n_erbs = len(subset_erb)
        cobertura_media = subset_erb['Raio_Cobertura_km'].mean()
        densidade_cobertura = n_erbs / 325  # Aproximação da área em km²
        
        info_text = (f"Total de ERBs: {n_erbs}\n"
                    f"Raio médio: {cobertura_media:.2f} km\n"
                    f"Densidade: {densidade_cobertura:.2f} ERBs/km²")
                    
        ax.annotate(info_text, xy=(0.02, 0.96), xycoords='axes fraction',
                   fontsize=11, ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=cor_base, alpha=0.8))
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    
    # Salvar figura
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Mapa de cobertura por operadora salvo em {caminho_saida}")

def criar_mapa_sobreposicao(gdf_erb, gdf_setores, caminho_saida, cores_operadoras=CORES_OPERADORAS):
    """
    Cria um mapa mostrando a sobreposição de cobertura entre operadoras.
    
    Args:
        gdf_erb: GeoDataFrame com as ERBs
        gdf_setores: GeoDataFrame com os setores de cobertura
        caminho_saida: Caminho onde salvar o mapa
        cores_operadoras: Dicionário de cores para cada operadora
    """
    print("Criando mapa de sobreposição de cobertura...")
    
    try:
        # Reprojetar para Web Mercator
        gdf_erb_3857 = gdf_erb.to_crs(epsg=3857)
        gdf_setores_3857 = gdf_setores.to_crs(epsg=3857)
        
        # Obter limites (bounding box)
        bbox = gdf_erb_3857.total_bounds
        x_min, y_min, x_max, y_max = bbox
        
        # Definir tamanho da grade para rasterização
        grid_size = 500
        
        # Definir transformação afim para o raster
        transform = from_bounds(x_min, y_min, x_max, y_max, grid_size, grid_size)
        
        # Inicializar array de contagem com zeros
        contagem_sobreposicao = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Para cada operadora, rasterize seus polígonos de cobertura e some os resultados
        for operadora in cores_operadoras.keys():
            subset = gdf_setores_3857[gdf_setores_3857['Operadora'] == operadora]
            if not subset.empty:
                shapes = [(geom, 1) for geom in subset.geometry if geom.is_valid]
                if shapes:
                    mask = rasterize(
                        shapes=shapes,
                        out_shape=(grid_size, grid_size),
                        transform=transform,
                        all_touched=True,
                        fill=0,
                        dtype=np.uint8
                    )
                    contagem_sobreposicao += mask
        
        # Plotar o resultado
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plotar a matriz de sobreposição como imagem
        im = ax.imshow(contagem_sobreposicao, extent=[x_min, x_max, y_min, y_max],
                       cmap=plt.cm.viridis, origin='lower', alpha=0.7, interpolation='bilinear')
        
        # Opcional: plotar os limites dos setores para cada operadora
        for operadora in cores_operadoras.keys():
            subset = gdf_setores_3857[gdf_setores_3857['Operadora'] == operadora]
            if not subset.empty:
                subset.boundary.plot(ax=ax, color=cores_operadoras[operadora], linewidth=1, alpha=0.8)
        
        # Plotar as ERBs
        for operadora, cor in cores_operadoras.items():
            subset = gdf_erb_3857[gdf_erb_3857['Operadora'] == operadora]
            if not subset.empty:
                subset.plot(ax=ax, color=cor, markersize=20, edgecolor='white', linewidth=0.5)
        
        # Adicionar mapa base
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        # Adicionar elementos cartográficos
        adicionar_elementos_cartograficos(
            ax, 
            'Sobreposição de Cobertura entre Operadoras',
            "Operadoras sobrepostas: CLARO, OI, TIM, VIVO"
        )
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Número de Operadoras com Cobertura', fontsize=12)
        cbar.set_ticks(range(5))
        cbar.set_ticklabels(['Sem cobertura', '1 operadora', '2 operadoras', '3 operadoras', '4 operadoras'])
        
        # Adicionar legenda
        criar_legenda_personalizada(ax, cores_operadoras, "Operadoras")
        
        # Salvar figura
        plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Mapa de sobreposição salvo em {caminho_saida}")
        
    except Exception as e:
        print(f"Erro ao criar mapa de sobreposição: {e}")
        plt.close('all')

def criar_mapa_calor_potencia(gdf_erb, caminho_saida):
    """
    Cria um mapa de calor mostrando a intensidade da potência das ERBs.
    
    Args:
        gdf_erb: GeoDataFrame com as ERBs
        caminho_saida: Caminho onde salvar o mapa
    """
    print("Criando mapa de calor de potência...")
    
    # Reprojetar para Web Mercator
    gdf_erb_3857 = gdf_erb.to_crs(epsg=3857)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Obter limites (bounding box) com margem
    bbox = gdf_erb_3857.total_bounds
    x_min, y_min, x_max, y_max = bbox
    margin = 0.01
    x_min -= margin; y_min -= margin; x_max += margin; y_max += margin
    
    # Criar grade para interpolação
    grid_size = 500
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # Obter pontos e valores para interpolação
    pontos = np.array([(p.x, p.y) for p in gdf_erb_3857.geometry])
    valores = gdf_erb_3857['EIRP_dBm'].values
    
    # Interpolar valores na grade
    grid_potencia = griddata(pontos, valores, (xi, yi), method='cubic', fill_value=np.min(valores))
    
    # Limitar valores para melhor visualização (percentis 5 e 95)
    vmin = np.percentile(grid_potencia, 5)
    vmax = np.percentile(grid_potencia, 95)
    
    # Criar mapa de cores personalizado
    cmap = LinearSegmentedColormap.from_list('potencia',
                ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
                 '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'], N=256)
    
    # Plotar a interpolação como imagem
    im = ax.imshow(grid_potencia, extent=[x_min, x_max, y_min, y_max],
                  origin='lower', cmap=cmap, alpha=0.8, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Adicionar contornos
    contornos = ax.contour(xi, yi, grid_potencia, levels=5, colors='white', alpha=0.6, linewidths=0.8)
    plt.clabel(contornos, inline=1, fontsize=8, fmt='%.1f dBm')
    
    # Plotar as ERBs
    scatter = ax.scatter(gdf_erb_3857.geometry.x, gdf_erb_3857.geometry.y,
                        c=gdf_erb_3857['EIRP_dBm'], cmap=cmap, s=50, edgecolor='white',
                        linewidth=0.5, alpha=0.9, vmin=vmin, vmax=vmax)
    
    # Adicionar mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Adicionar elementos cartográficos
    adicionar_elementos_cartograficos(ax, 'Potência Efetivamente Irradiada (EIRP) das ERBs')
    
    # Adicionar barra de cores
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('EIRP (dBm)', fontsize=12)
    
    # Adicionar estatísticas
    eirp_min = gdf_erb_3857['EIRP_dBm'].min()
    eirp_max = gdf_erb_3857['EIRP_dBm'].max()
    eirp_media = gdf_erb_3857['EIRP_dBm'].mean()
    
    info_text = (f"EIRP Média: {eirp_media:.1f} dBm\n"
                f"EIRP Mínima: {eirp_min:.1f} dBm\n"
                f"EIRP Máxima: {eirp_max:.1f} dBm")
                
    ax.annotate(info_text, xy=(0.02, 0.96), xycoords='axes fraction',
                fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Salvar figura
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Mapa de calor de potência salvo em {caminho_saida}")

def criar_mapa_folium(gdf_erb, caminho_saida):
    """
    Cria um mapa interativo Folium com as ERBs.
    
    Args:
        gdf_erb: GeoDataFrame com as ERBs
        caminho_saida: Caminho onde salvar o mapa HTML
    """
    print("Criando mapa interativo Folium...")
    
    # Filtrar dados válidos
    geo_df = gdf_erb.dropna(subset=['Latitude', 'Longitude'])
    geo_df = geo_df[(geo_df['Latitude'] != 0) & (geo_df['Longitude'] != 0)]
    
    if geo_df.empty:
        print("Não há dados válidos para gerar o mapa Folium.")
        return
    
    # Centro do mapa (média das coordenadas)
    map_center = [geo_df['Latitude'].mean(), geo_df['Longitude'].mean()]
    
    # Criar mapa
    m = folium.Map(location=map_center, zoom_start=10, tiles='CartoDB positron')
    
    # Adicionar cluster de marcadores
    marker_cluster = MarkerCluster().add_to(m)
    
    # Para cada operadora, usar uma cor diferente
    cores_html = {
        'CLARO': 'red',
        'OI': 'orange',
        'VIVO': 'purple',
        'TIM': 'blue',
        'OUTRA': 'gray'
    }
    
    # Adicionar marcadores para cada ERB
    for idx, row in geo_df.iterrows():
        cor = cores_html.get(row.get('Operadora', 'OUTRA'), 'gray')
        
        # Criar texto pop-up
        popup_text = f"""
        <b>Operadora:</b> {row.get('NomeEntidade', 'N/A')}<br>
        <b>Tecnologia:</b> {row.get('Tecnologia', 'N/A')}<br>
        <b>Freq. Tx:</b> {row.get('FreqTxMHz', 'N/A')} MHz<br>
        <b>EIRP:</b> {row.get('EIRP_dBm', 'N/A')} dBm<br>
        <b>Raio:</b> {row.get('Raio_Cobertura_km', 'N/A')} km<br>
        <b>Azimute:</b> {row.get('Azimute', 'N/A')}°<br>
        <b>Altura:</b> {row.get('AlturaAntena', 'N/A')} m<br>
        """
        
        # Criar tooltip (texto que aparece ao passar o mouse)
        tooltip = f"{row.get('Operadora', 'ERB')}: {row.get('Tecnologia', '')}"
        
        # Adicionar marcador
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color=cor, icon='signal', prefix='fa')
        ).add_to(marker_cluster)
    
    # Salvar mapa
    m.save(caminho_saida)
    print(f"Mapa interativo Folium salvo em: {caminho_saida}")
