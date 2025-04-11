import pandas as pd

def load_and_clean_data(input_path: str, output_path: str) -> pd.DataFrame | None:
    """Carrega o CSV de licenciamento, remove colunas desnecessárias e salva o resultado."""
    
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada não encontrado em {input_path}")
        return None
    except Exception as e:
        print(f"Erro ao ler o CSV de entrada: {e}")
        return None

    colunas_para_excluir = [
        'NumFistel', 'NumServico', 'NumAto', 'CodDebitoTFI', '_id',
        'NumFistelAssociado', 'NumRede', 'NumEstacao',
        'DataLicenciamento', 'DataPrimeiroLicenciamento', 'DataValidade',
        'CodTipoAntena', 'CodEquipamentoAntena', 'CodEquipamentoTransmissor',
        'CodTipoClasseEstacao', 'DesignacaoEmissao', 'ClassInfraFisica',
        'CompartilhamentoInfraFisica', 'NomeEntidadeAssociado',
        'FrenteCostaAntena', 'AnguloMeiaPotenciaAntena'
    ]

    colunas_existentes = [col for col in colunas_para_excluir if col in df.columns]
    colunas_nao_encontradas = set(colunas_para_excluir) - set(colunas_existentes)

    if colunas_nao_encontradas:
        print(f"Aviso: As seguintes colunas não foram encontradas e não serão excluídas: {colunas_nao_encontradas}")

    df_filtrado = df.drop(columns=colunas_existentes)

    try:
        df_filtrado.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Arquivo limpo salvo com sucesso em {output_path}")
        print(f"Colunas mantidas: {list(df_filtrado.columns)}")
        print(f"Número de ERBs no dataset: {len(df_filtrado)}")
    except Exception as e:
        print(f"Erro ao salvar o CSV de saída: {e}")
        return None
        
    return df_filtrado
