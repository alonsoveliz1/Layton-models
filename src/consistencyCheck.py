import os
from typing import List, Dict, Tuple

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def find_csv_files(base_folder) -> List[str]:
    # Metodo para ver cuantos archivos CSV componen el CIC-BCCC-NRC-TabularIoT-2024
    # Busco en el directorio base y subdirectorios (donde dentro estan los csv que componen cada dataset)

    csv_files: List[str] = []
    for subdir, _, files in os.walk(base_folder):
        for file in files:
            if isinstance(file, str) and file.endswith(".csv"):
                csv_path = os.path.join(subdir, file)
                csv_files.append(csv_path)
    print(f"El dataset CIC-BCCC-TabularIoT-2024-TCP se compone de {len(csv_files)} CSV files.\n")
    return csv_files



def get_file_info(csv_file: str) -> Dict:
    # Recoger informacion de los archivos csv para luego comprobar consistencia:
    # {nombre, numero de atributos, nombre de las columnas, diccionario con los tipos de cada atributo {attr:type}}

    df = pd.read_csv(csv_file, nrows=10)
    return {
        "archivo": csv_file,
        "num_columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
    }



def check_file_consistency(csv_files, reference_csv) -> List[Tuple[str, str]]:  # Devuelve una lista de tuplas {archivo no consistente, motivo}
    """Ahora que ya sabemos cuantos atributos tiene cada fichero csv, debemos garantizar que todos tengan los mismos atributos, en el mismo orden
       y con los mismos tipos. Para ello se elige un csv de referencia al igual que antes"""

    print(f"Usando el archivo de referencia ({reference_csv})")

    df_reference_header = pd.read_csv(reference_csv, nrows=100) # Leemos 100 filas para inferir los tipos de los atributos
    reference_columns = df_reference_header.columns.tolist()  # Atributos del dataset que tomamos como referencia
    reference_types = df_reference_header.dtypes.to_dict()  # Tipos del dataframe de referencia

    print(f"\nAtributos y tipos del csv de referencia {reference_csv}:")
    for col, dt in reference_types.items():
        print(f" - {col}: {dt}")
    print("\n")

    mismatched_files = []  # Array para almacenar cuales son los archivos que no cumplen consistencia
    for csv_path in csv_files:
        if csv_path == reference_csv: # Skippeamos el csv que hemos tomado como referencia
            continue

        # De la misma manera leemos atributos y tipos de cada uno de los csv (los referenciamos como tmp para iterarlos)
        curr_df_header = pd.read_csv(csv_path, nrows=100)
        curr_columns = curr_df_header.columns.tolist()

        if len(curr_columns) != len(reference_columns):  # Si tienen mas o menos atributos
            mismatched_files.append((csv_path,
                                     f"Tiene diferente numero de atributos ya que aparece un nuevo atributo {set(curr_df_header.columns) - set(df_reference_header.columns)}"))
            continue

        if curr_columns != reference_columns:
            mismatched_files.append((csv_path, "Diferente orden en los atributos o diferentes nombres"))
            continue

        tmp_dtypes = curr_df_header.dtypes.to_dict()
        for col in reference_columns:
            if reference_types[col] != tmp_dtypes[col]:
                mismatched_files.append(
                    (csv_path,
                     f"Tipo diferente para el atributo '{col}' (Tipo de referencia: {reference_types[col]}, Tipo en el csv: {tmp_dtypes[col]})"))

    return mismatched_files

def find_nulls(csv_files: list[str]) -> list[tuple[str, str, dict]]:
    """Metodo para encontrar que csvs tienen valores nulos en alguno de sus atributos y devolverlos en la forma [archivo, total nulos, columnas_con_nulos: num_null_columna"""
    csvs_with_nulls = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        null_counts = df.isna().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            null_columns = {col: count for col, count in null_counts.items() if count > 0}
            csvs_with_nulls.append((csv_file, total_nulls, null_columns))
            print(f"Found {total_nulls} nulls in {os.path.basename(csv_file)}")
    return csvs_with_nulls


def run_consistency_check(base_folder: str, reference_csv: str):
    """Metodo para invocar al proceso de verifcar la consistencia y mostrar su resultado. En caso de que no haya consistencia se muestra
       cual es el archivo que causa el problema, y el motivo."""

    csv_files = find_csv_files(base_folder) # Array con los archivos csv
    mismatched_files = check_file_consistency(csv_files, reference_csv)
    csvs_with_nulls = find_nulls(csv_files)

    if not mismatched_files:
        print("Todos los CSV son consistentes en número de columnas, orden, tipo y tipo de los atributos.")
    else:
        print("Hay CSV que no son consistentes:")
        for file_path, reason in mismatched_files:
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

    # DESCOMENTAR SI QUIERO ANALIZAR EL DATASET PROCESADO (comprobacion de que este correcto)
    #base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    #reference_csv = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD\\CIC-BCCC-ACI-IOT-2023\\Benign Traffic.csv"

    run_consistency_check(base_folder, reference_csv)

if __name__ == "__main__":
    main()
