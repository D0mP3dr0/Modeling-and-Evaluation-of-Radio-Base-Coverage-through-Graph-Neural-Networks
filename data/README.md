# Pasta de Dados

Esta pasta deve conter os arquivos de dados utilizados pelo projeto. Devido ao tamanho, estes arquivos não são versionados no Git.

## Arquivos necessários

1. **csv_licenciamento_bruto.csv** - Arquivo CSV com dados de licenciamento de ERBs da Anatel

   * Este arquivo deve ser baixado diretamente do site da Anatel ou obtido de outras fontes oficiais.
   * Renomeie-o para `csv_licenciamento_bruto.csv` ou ajuste o caminho em `main.py`.

## Formato esperado

O arquivo CSV deve conter, no mínimo, as seguintes colunas:

* `Latitude` - Latitude da ERB em graus decimais
* `Longitude` - Longitude da ERB em graus decimais
* `NomeEntidade` - Nome da operadora/entidade
* `FreqTxMHz` - Frequência de transmissão em MHz
* `PotenciaTransmissorWatts` - Potência do transmissor em Watts
* `GanhoAntena` - Ganho da antena em dBi
* `Azimute` - Azimute da antena em graus

## Fontes de dados

* **Anatel (Brasil)**: Através do [sistema de consulta pública](https://sistemas.anatel.gov.br/se/public/view/b/srd.php) ou [Dados Abertos](https://www.gov.br/anatel)
* **Mosaik Solutions**: Provê dados globais de ERBs (requer licença)
* **OpenCelliD**: Projeto open source com dados de ERBs (pode ser menos preciso)

## Arquivos gerados

Durante o processamento, o seguinte arquivo será criado nesta pasta:

* `erb_sorocaba_limpo.csv` - Versão limpa e filtrada do arquivo original 