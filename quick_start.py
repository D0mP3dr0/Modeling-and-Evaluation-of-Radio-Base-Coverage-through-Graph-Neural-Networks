#!/usr/bin/env python3
"""
Script de inicialização rápida para o Sistema de Análise de ERBs.
Permite executar facilmente o projeto com dados de exemplo e visualizar um resumo das funcionalidades.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path

# Verifica se o diretório data existe, se não, cria
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Verifica se o exemplo existe
example_file = data_dir / "example_data.csv"
if not example_file.exists():
    print("Arquivo de exemplo não encontrado. Crie o arquivo data/example_data.csv antes de continuar.")
    sys.exit(1)

def show_welcome():
    """Mostra a mensagem de boas-vindas e instrui sobre as opções disponíveis"""
    print("\n" + "=" * 80)
    print("SISTEMA DE ANÁLISE DE ESTAÇÕES RÁDIO BASE (ERBs)".center(80))
    print("=" * 80)
    print("\nEste script permite executar facilmente o projeto com dados de exemplo.")
    print("\nOpções disponíveis:")
    print("  1. Executar análise básica e visualizações")
    print("  2. Executar análise de tecnologia e frequência")
    print("  3. Executar análise temporal avançada")
    print("  4. Executar análise de correlação e espacial")
    print("  5. Executar todas as análises (pode demorar)")
    print("  6. Iniciar dashboard interativo")
    print("  7. Gerar relatório completo")
    print("  8. Verificar dependências e instalação")
    print("  9. Sair")

def run_command(command):
    """Executa um comando e retorna o resultado"""
    try:
        print(f"\nExecutando: {command}")
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando: {e}")
        return False

def check_dependencies():
    """Verifica as dependências instaladas"""
    print("\nVerificando dependências instaladas...")
    
    try:
        # Verificar se o requirements.txt existe
        if not os.path.exists("requirements.txt"):
            print("Arquivo requirements.txt não encontrado.")
            return False
        
        # Verificar pacotes instalados
        result = subprocess.run(
            "pip list --format=freeze", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        installed_packages = {
            line.split('==')[0].lower(): line.split('==')[1] 
            for line in result.stdout.splitlines() 
            if '==' in line
        }
        
        # Ler requirements
        with open("requirements.txt", "r") as f:
            requirements = [
                line.strip() for line in f 
                if line.strip() and not line.startswith('#') and '>=' in line
            ]
        
        missing = []
        for req in requirements:
            package = req.split('>=')[0].strip().lower()
            if package not in installed_packages:
                missing.append(package)
        
        if missing:
            print(f"Pacotes ausentes: {', '.join(missing)}")
            install = input("Deseja instalar os pacotes ausentes? (s/n): ").lower()
            if install == 's':
                run_command("pip install -r requirements.txt")
                print("Dependências instaladas.")
                return True
            else:
                print("Dependências incompletas. Algumas funcionalidades podem não funcionar.")
                return False
        else:
            print("Todas as dependências estão instaladas.")
            return True
    
    except Exception as e:
        print(f"Erro ao verificar dependências: {e}")
        return False

def main():
    """Função principal do script de inicialização rápida"""
    parser = argparse.ArgumentParser(description="Script de inicialização rápida para o Sistema de Análise de ERBs")
    parser.add_argument('--quick', action='store_true', help='Executa a análise básica diretamente, sem menu')
    args = parser.parse_args()
    
    if args.quick:
        # Executa análise básica diretamente
        results_dir = "quick_results"
        command = f"python src/main.py --input {example_file} --output {results_dir} --basic --visualization"
        success = run_command(command)
        
        if success:
            print(f"\nAnálise básica concluída. Resultados salvos em '{results_dir}'.")
            # Tenta abrir o diretório de resultados
            try:
                if sys.platform == 'darwin':  # macOS
                    run_command(f"open {results_dir}")
                elif sys.platform == 'win32':  # Windows
                    run_command(f"explorer {results_dir}")
                else:  # Linux
                    run_command(f"xdg-open {results_dir}")
            except:
                pass
        return
    
    while True:
        show_welcome()
        choice = input("\nEscolha uma opção (1-9): ")
        
        results_dir = "results"
        
        if choice == '1':
            run_command(f"python src/main.py --input {example_file} --output {results_dir} --basic --visualization")
        elif choice == '2':
            run_command(f"python src/main.py --input {example_file} --output {results_dir} --tech-frequency")
        elif choice == '3':
            run_command(f"python src/main.py --input {example_file} --output {results_dir} --temporal")
        elif choice == '4':
            run_command(f"python src/main.py --input {example_file} --output {results_dir} --correlation --spatial")
        elif choice == '5':
            run_command(f"python src/main.py --input {example_file} --output {results_dir} --all")
        elif choice == '6':
            print("\nIniciando dashboard interativo. Pressione Ctrl+C no terminal para encerrar.")
            run_command(f"python src/main.py --input {example_file} --dashboard")
        elif choice == '7':
            run_command(f"python src/main.py --input {example_file} --output {results_dir} --report")
            # Tenta abrir o relatório PDF gerado
            latest_dir = subprocess.run(
                f"ls -td {results_dir}/analysis_* | head -1", 
                shell=True, 
                capture_output=True, 
                text=True
            ).stdout.strip()
            if latest_dir:
                pdf_path = f"{latest_dir}/reports/erb_report.pdf"
                if os.path.exists(pdf_path):
                    try:
                        webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
                    except:
                        print(f"Relatório gerado em: {pdf_path}")
        elif choice == '8':
            check_dependencies()
        elif choice == '9':
            print("\nEncerrando. Obrigado por usar o Sistema de Análise de ERBs!")
            break
        else:
            print("\nOpção inválida. Por favor, escolha de 1 a 9.")
        
        input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    main() 