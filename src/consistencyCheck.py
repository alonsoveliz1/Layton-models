import os
from typing import List, Dict, Tuple

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def find_csv_files(base_folder) -> List:
    # Metodo para ver cuantos archivos CSV componen el CIC-BCCC-NRC-TabularIoT-2024
    # Busco en el directorio base y subdirectorios (donde dentro estan los csv que componen cada dataset)

    csv_files = []
    for subdir, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(subdir, file)
                csv_files.append(csv_path)
    print(f"El dataset CIC-BCCC-NRC-TabularIoT-2024 se compone de {len(csv_files)} CSV files.\n")
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


def analyze_column_consistency(csv_files, reference_csv):
    """Metodo para analizar cuantas columnas tiene cada fichero csv. Como ya sabemos que 85 son las esperadas, y que hay 2 archivos con 86
       se incorpora funcionalidad para saber cual es exactamente los que difieren del resto, eligiendo un csv como referencia y comparando sobre el"""

    info_csv_files = []
    for csv_file in csv_files:
        info_csv_files.append(get_file_info(csv_file))

    # Convertimos en un df para poder representarlo
    df_info = pd.DataFrame(info_csv_files)

    print("\nDistribucion del numero de atributos en los diferentes archivos csv:")
    print(df_info["num_columns"].value_counts())

    # Si hay mas de una fila en el df (hay archivos con diferentes atributos)
    if len(df_info["num_columns"].value_counts()) != 1:
        # Leemos los atributos del csv que nombramos de "referencia"
        df_reference_csv = pd.read_csv(reference_csv, nrows=0)

        # Vemos cuantos tiene
        num_columns_reference = len(df_reference_csv.columns)

        # Mostramos cuales son los que tienen un numero diferente a el
        print("\nLos ficheros que tienen un numero diferente de atributos son:")
        print(f" {df_info["archivo"][df_info["num_columns"] != num_columns_reference]}\n")


def check_file_consistency(csv_files, reference_csv) -> List[
    Tuple[str, str]]:  # Devuelve una lista de tuplas {archivo no consistente, motivo}
    """Ahora que ya sabemos cuantos atributos tiene cada fichero csv, debemos garantizar que todos tengan los mismos atributos, en el mismo orden
       y con los mismos tipos. Para ello se elige un csv de referencia al igual que antes"""

    print(f"Usando el archivo de referencia ({reference_csv})")

    # Leemos 100 filas para inferir los tipos de los atributos
    df_reference_header = pd.read_csv(reference_csv, nrows=100)
    reference_types = df_reference_header.dtypes.to_dict()  # Tipos del dataframe de referencia

    print(f"\nAtributos y tipos del csv de referencia {reference_csv}:")
    for col, dt in reference_types.items():
        print(f" - {col}: {dt}")
    print("\n")

    reference_columns = df_reference_header.columns.tolist()

    mismatched_files = []  # Array para almacenar cuales son los archivos que no cumplen consistencia
    for csv_path in csv_files:
        if csv_path == reference_csv:
            continue

        # De la misma manera leemos atributos y tipos de cada uno de los csv (los referenciamos como tmp para iterarlos)
        df_tmp_header = pd.read_csv(csv_path, nrows=10)
        tmp_columns = df_tmp_header.columns.tolist()

        if len(tmp_columns) != len(reference_columns):  # Si tienen mas o menos atributos
            mismatched_files.append((csv_path,
                                     f"Tiene diferente numero de atributos ya que aparece un nuevo atributo {set(df_tmp_header.columns) - set(df_reference_header.columns)}"))
            continue

        if tmp_columns != reference_columns:
            mismatched_files.append((csv_path, "Diferente orden en los atributos o diferente nombres"))
            continue

        tmp_dtypes = df_tmp_header.dtypes.to_dict()
        for col in reference_columns:
            if reference_types[col] != tmp_dtypes[col]:
                mismatched_files.append(
                    (csv_path,
                     f"Tipo diferente para el atributo '{col}' (Tipo de referencia: {reference_types[col]}, Tipo en el csv: {tmp_dtypes[col]})"))

    return mismatched_files


def run_consistency_check(base_folder: str, reference_csv: str):
    """Metodo para invocar al proceso de verifcar la consistencia y mostrar su resultado. En caso de que no haya consistencia se muestra
       cual es el archivo que causa el problema, y el motivo."""

    csv_files = find_csv_files(base_folder)
    analyze_column_consistency(csv_files, reference_csv)
    mismatched_files = check_file_consistency(csv_files, reference_csv)

    if not mismatched_files:
        print("Todos los CSV son consistentes en número de columnas, orden, tipo y tipo de los atributos.")
    else:
        print("Hay CSV que no son consistentes:")
        for file_path, reason in mismatched_files:
            print(f"- {file_path}: {reason}")


def main():
    # DESCOMENTAR SI QUIERO ANALIZAR EL DATASET ORIGINAL
    #base_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\raw\\CIC-BCCC-NRC-TabularIoT-2024"
    #reference_csv = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\raw\\CIC-BCCC-NRC-TabularIoT-2024\\CIC-BCCC-ACI-IOT-2023\\Benign Traffic.csv"

    # DESCOMENTAR SI QUIERO ANALIZAR EL DATASET PROCESADO (comprobacion de que este correcto)
    base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    reference_csv = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD\\CIC-BCCC-ACI-IOT-2023\\Benign Traffic.csv"

    run_consistency_check(base_folder, reference_csv)


if __name__ == "__main__":
    main()
