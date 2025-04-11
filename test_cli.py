"""
Script para testar a interface de linha de comando do projeto.
Gera dados de exemplo e executa o script principal para verificar o funcionamento.
"""

import os
import sys
import subprocess
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def create_test_data():
    """Cria dados de teste e salva como CSV e GeoJSON"""
    print("Gerando dados de teste...")
    
    # Criar dados fictícios
    data = {
        'Latitude': [-23.45, -23.46, -23.47, -23.48, -23.49, -23.50],
        'Longitude': [-47.55, -47.56, -47.57, -47.58, -47.59, -47.60],
        'Operator': ['CLARO', 'VIVO', 'TIM', 'OI', 'CLARO', 'VIVO'],
        'PotenciaTransmissorWatts': [40, 35, 30, 45, 38, 42],
        'FreqTxMHz': [850, 900, 1800, 2600, 2100, 700],
        'GanhoAntena': [16, 17, 15, 18, 16, 17],
        'Azimute': [0, 120, 240, 90, 180, 270],
        'Tecnologia': ['4G', '4G', '3G', '5G', '4G', '5G'],
        'NomeEntidade': ['CLARO S.A.', 'TELEFÔNICA BRASIL S.A.', 'TIM S.A.', 'OI MÓVEL S.A.', 
                        'CLARO S.A.', 'TELEFÔNICA BRASIL S.A.'],
        'data_licenciamento': ['2020-01-15', '2018-05-20', '2019-10-10', '2021-03-01', 
                              '2017-07-05', '2022-02-28']
    }
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    # Salvar como CSV
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/test_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV salvo em: {csv_path}")
    
    # Converter para GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Salvar como GeoJSON
    geojson_path = 'data/test_data.geojson'
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"GeoJSON salvo em: {geojson_path}")
    
    return csv_path, geojson_path

def run_cli_command(command):
    """Executa um comando CLI e retorna o resultado"""
    print(f"Executando comando: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Comando executado com sucesso!")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False, e.stderr

def test_cli_basic():
    """Testa funcionalidade básica da linha de comando"""
    csv_path, geojson_path = create_test_data()
    
    # Criar diretório para resultados
    os.makedirs('test_cli_results', exist_ok=True)
    
    # Lista de comandos para testar
    commands = [
        f"python src/main.py --input {csv_path} --output test_cli_results --basic",
        f"python src/main.py --input {geojson_path} --output test_cli_results --visualization",
        f"python src/main.py --input {csv_path} --output test_cli_results --tech-frequency",
        f"python src/main.py --input {csv_path} --output test_cli_results --temporal"
    ]
    
    # Executar cada comando
    success_count = 0
    for command in commands:
        success, _ = run_cli_command(command)
        if success:
            success_count += 1
    
    # Relatório final
    print(f"\nResultado dos testes: {success_count}/{len(commands)} comandos bem-sucedidos")
    
    return success_count == len(commands)

if __name__ == "__main__":
    print("Iniciando teste da interface de linha de comando...")
    success = test_cli_basic()
    
    if success:
        print("\n✅ Todos os testes da linha de comando foram bem-sucedidos!")
        sys.exit(0)
    else:
        print("\n❌ Alguns testes da linha de comando falharam.")
        sys.exit(1) 