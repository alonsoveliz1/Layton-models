import os
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def find_csv_files(base_folder) -> [str]:
    """Metodo para ver cuantos archivos CSV componen el CIC-BCCC-NRC-TabularIoT-2024
    # Busco en el directorio base y subdirectorios (donde dentro estan los csv que componen cada subdataset)"""

    print(f"Buscando ficheros csv dentro de {base_folder}\n")
    csv_files = []
    for subdir, _, files in os.walk(base_folder):
        for file in files:
            if isinstance(file, str) and file.endswith(".csv"):
                csv_path = os.path.join(subdir, file)
                csv_files.append(csv_path)
    print(f"El dataset CIC-BCCC-TabularIoT-2024-TCP se compone de {len(csv_files)} CSV files.\n")
    return csv_files



def verificar_consistencia(csv_files, reference_csv) -> [(str,str)]:  # Devuelve una lista de tuplas (archivo no consistente, motivo)
    """Vamos a ver cuantos atributos tiene cada csv de cada dataset, debemos garantizar que todos tengan los mismos atributos, en el mismo orden
       y con los mismos tipos. Para ello se elige un csv de referencia"""

    print(f"Usando el archivo de referencia ({reference_csv})")

    df_referencia = pd.read_csv(reference_csv, nrows=100) # Leemos 100 filas solo para inferir los tipos
    reference_columns = df_referencia.columns.tolist()  # Atributos del dataset que tomamos como referencia
    reference_types = df_referencia.dtypes.to_dict()  # Tipos del dataframe de referencia

    print(f"El dataset de referencia {reference_csv} tiene {df_referencia.shape[1]} atributos\n")
    print(f"Los atributos y tipos son del {reference_csv} son:")
    for attr, attr_type in reference_types.items():
        print(f" - {attr}: {attr_type}")
    print("\n")

    mismatched_files = []  # Array para almacenar cuales son los archivos que no cumplen consistencia
    for csv_file_path in csv_files:
        if csv_file_path == reference_csv: # Skippeamos el csv que hemos tomado como referencia
            continue

        # Leemos atributos y tipos de cada uno de los csv (los referenciamos como tmp para iterarlos)
        curr_csv_header = pd.read_csv(csv_file_path, nrows=100)
        curr_columns = curr_csv_header.columns.tolist()

        # Y ahora pasamos a comparar el csv que estamos iterando sobre el que hemos escodigo de referencia
        if len(curr_columns) != len(reference_columns):  # Si tienen mas o menos atributos
            mismatched_files.append((csv_file_path,
                                     f"Tiene diferente numero de atributos ya que aparece un nuevo atributo {set(curr_columns) - set(reference_columns)}"))
            continue

        # Si tiene los mismos tipos pero estan en diferente orden
        if curr_columns != reference_columns:
            mismatched_files.append((csv_file_path, "Diferente orden en los atributos o diferentes nombres"))
            continue

        # Y faltaria comprobar los tipos, extraemos los tipos del csv que estamos iterando
        currr_types = curr_csv_header.dtypes.to_dict()
        for col in reference_columns:
            if reference_types[col] != currr_types[col]:
                mismatched_files.append(
                    (csv_file_path,
                     f"Tipo diferente para el atributo '{col}' (Tipo de referencia: {reference_types[col]}, Tipo en el csv: {currr_types[col]})"))


    return mismatched_files



def encontrar_csv_con_nulos(csv_files: [str]) -> [(str, str, dict)]:
    """Metodo para encontrar que csvs tienen valores nulos en alguno de sus atributos y devolverlos en la forma [(archivo, total nulos, columnas_con_nulos: nulos_columna)]"""

    csvs_with_nulls = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        null_counts = df.isna().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            null_columns = {col: count for col, count in null_counts.items() if count > 0}
            csvs_with_nulls.append((csv_file, total_nulls, null_columns))
            print(f"Hay {total_nulls} valores nulos {os.path.basename(csv_file)}")

    return csvs_with_nulls



def run_consistency_check(base_folder: str, reference_csv: str) -> None:
    """Metodo para invocar al proceso de verifcar la consistencia y mostrar su resultado. En caso de que no haya consistencia se muestra
       cual es el archivo que causa el problema, y el motivo."""

    csv_files = find_csv_files(base_folder) # Array con los archivos csv
    csv_inconsistentes = verificar_consistencia(csv_files, reference_csv) # Devuelve una lista de tuplas (archivo no consistente, motivo)
    csvs_with_nulls = encontrar_csv_con_nulos(csv_files)

    if not csv_inconsistentes:
        print("Todos los CSV son consistentes en número de columnas, orden, tipo y tipo de los atributos.")
    else:
        print("Hay CSV que no son consistentes:")
        for file_path, reason in csv_inconsistentes:
            print(f"- {file_path}: {reason}")
    print("\n")

    if csvs_with_nulls:
        print("\nArchivos CSV con valores nulos:")
        for file_path, total_nulls, null_columns in csvs_with_nulls:
            print(f"- {file_path}: {total_nulls} valores nulos en total")
            for col, count in null_columns.items():
                print(f"  • {col}: {count} valores nulos")
    else:
        print("Ningún CSV tiene valores nulos en los atributos de sus filas.")




def main():
    # DESCOMENTAR SI QUIERO ANALIZAR EL DATASET ORIGINAL
    base_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\raw\\CIC-BCCC-NRC-TabularIoT-2024"
    reference_csv = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\raw\\CIC-BCCC-NRC-TabularIoT-2024\\CIC-BCCC-ACI-IOT-2023\\Benign Traffic.csv"

    # DESCOMENTAR SI QUIERO ANALIZAR EL DATASET PROCESADO (comprobacion de que este correcto tras procesarlo)
    #base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    #reference_csv = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD\\CIC-BCCC-ACI-IOT-2023\\Benign Traffic.csv"

    run_consistency_check(base_folder, reference_csv)

if __name__ == "__main__":
    main()
