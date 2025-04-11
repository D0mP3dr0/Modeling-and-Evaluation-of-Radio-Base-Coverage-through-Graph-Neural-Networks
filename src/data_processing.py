import pandas as pd

def load_and_clean_data(input_path: str, output_path: str) -> pd.DataFrame | None:
    """Loads the licensing CSV, removes unnecessary columns and saves the result."""
    
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

    columns_to_exclude = [
        'NumFistel', 'NumServico', 'NumAto', 'CodDebitoTFI', '_id',
        'NumFistelAssociado', 'NumRede', 'NumEstacao',
        'DataLicenciamento', 'DataPrimeiroLicenciamento', 'DataValidade',
        'CodTipoAntena', 'CodEquipamentoAntena', 'CodEquipamentoTransmissor',
        'CodTipoClasseEstacao', 'DesignacaoEmissao', 'ClassInfraFisica',
        'CompartilhamentoInfraFisica', 'NomeEntidadeAssociado',
        'FrenteCostaAntena', 'AnguloMeiaPotenciaAntena'
    ]

    existing_columns = [col for col in columns_to_exclude if col in df.columns]
    columns_not_found = set(columns_to_exclude) - set(existing_columns)

    if columns_not_found:
        print(f"Warning: The following columns were not found and will not be excluded: {columns_not_found}")

    filtered_df = df.drop(columns=existing_columns)

    try:
        filtered_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Clean file successfully saved to {output_path}")
        print(f"Retained columns: {list(filtered_df.columns)}")
        print(f"Number of RBS in the dataset: {len(filtered_df)}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        return None
        
    return filtered_df
