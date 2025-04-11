"""
Script para testar a importação e funcionalidade básica de todos os módulos do projeto.
Este arquivo ajuda a identificar problemas de importação ou outros erros fundamentais.
"""

import os
import sys
import logging
import importlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_modules")

def create_test_data():
    """Cria um GeoDataFrame de teste simples para verificar as funções"""
    # Criar dados fictícios
    data = {
        'Latitude': [-23.45, -23.46, -23.47, -23.48],
        'Longitude': [-47.55, -47.56, -47.57, -47.58],
        'Operator': ['CLARO', 'VIVO', 'TIM', 'OI'],
        'PotenciaTransmissorWatts': [40, 35, 30, 45],
        'FreqTxMHz': [850, 900, 1800, 2600],
        'GanhoAntena': [16, 17, 15, 18],
        'Azimute': [0, 120, 240, 90],
        'Tecnologia': ['4G', '4G', '3G', '5G'],
        'NomeEntidade': ['CLARO S.A.', 'TELEFÔNICA BRASIL S.A.', 'TIM S.A.', 'OI MÓVEL S.A.'],
        'data_licenciamento': ['2020-01-15', '2018-05-20', '2019-10-10', '2021-03-01']
    }
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    # Converter para GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf

def test_module_import(module_name):
    """Testa a importação de um módulo"""
    try:
        module = importlib.import_module(f"src.{module_name}")
        logger.info(f"✅ Módulo '{module_name}' importado com sucesso")
        return module
    except ImportError as e:
        logger.error(f"❌ Erro ao importar módulo '{module_name}': {e}")
        # Tentar importar diretamente (sem prefixo src)
        try:
            module = importlib.import_module(module_name)
            logger.info(f"✅ Módulo '{module_name}' importado diretamente com sucesso")
            return module
        except ImportError as e2:
            logger.error(f"❌ Erro ao importar módulo diretamente '{module_name}': {e2}")
            return None
    except Exception as e:
        logger.error(f"❌ Erro inesperado ao importar '{module_name}': {e}")
        return None

def test_core_modules():
    """Testa os módulos principais do projeto"""
    modules_to_test = [
        "data_processing",
        "analysis",
        "visualization",
        "graph_analysis",
        "coverage_models",
        "tech_frequency_analysis",
        "advanced_temporal_analysis",
        "correlation_analysis",
        "spatial_analysis",
        "integration_analysis",
        "prediction_module",
        "dashboard_interactive",
        "report_generator",
        "unit_tests"
    ]
    
    imported_modules = {}
    for module_name in modules_to_test:
        module = test_module_import(module_name)
        if module:
            imported_modules[module_name] = module
    
    return imported_modules

def test_module_functions(modules, test_data):
    """Testa as funções básicas dos módulos importados"""
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Testar tech_frequency_analysis
    if "tech_frequency_analysis" in modules:
        try:
            logger.info("Testando tech_frequency_analysis.run_tech_frequency_analysis()...")
            modules["tech_frequency_analysis"].run_tech_frequency_analysis(test_data, results_dir)
            logger.info("✅ run_tech_frequency_analysis executado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro em run_tech_frequency_analysis: {e}")
    
    # Testar advanced_temporal_analysis
    if "advanced_temporal_analysis" in modules:
        try:
            logger.info("Testando advanced_temporal_analysis.run_temporal_analysis()...")
            modules["advanced_temporal_analysis"].run_temporal_analysis(test_data, results_dir)
            logger.info("✅ run_temporal_analysis executado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro em run_temporal_analysis: {e}")

def main():
    """Função principal para executar todos os testes"""
    logger.info("Iniciando testes dos módulos...")
    
    # Criar dados de teste
    logger.info("Criando dados de teste...")
    test_data = create_test_data()
    
    # Testar importação dos módulos
    logger.info("Testando importação dos módulos...")
    imported_modules = test_core_modules()
    
    # Testar funções dos módulos
    logger.info("Testando funções dos módulos...")
    test_module_functions(imported_modules, test_data)
    
    logger.info("Testes concluídos")

if __name__ == "__main__":
    main() 