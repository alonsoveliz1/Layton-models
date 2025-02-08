from typing import List, Dict, Tuple, Set
import os
import pandas as pd

reference_csv = os.path.join(base_folder, "CIC-BCCC-ACI-IOT-2023", "Benign Traffic.csv")
#
def find_csv_files(base_folder):
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
    # Recoger informacion de los archivos csv para luego comprobar consistencia {filename, attribute_count, attribute_name, dict {attr:type}}
    df = pd.read_csv(csv_file, nrows=10)
    return {
        "archivo": csv_file,
        "num_columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
    }


def analyze_column_consistency(csv_files, reference_csv):
    """Metodo para analizar cuantas columnas tiene cada fichero csv. Como ya sabemos que 85 son las esperadas, y que hay 2 archivos con 86
       se incorpora funcionalidad para saber cuales son exactamente los que difieren del resto"""
    info_csv_files = []
    for csv_file in csv_files:
        file_info = get_file_info(csv_file)
        if file_info:
            info_csv_files.append(file_info)

    df_info = pd.DataFrame(info_csv_files)
    print("\nDistribucion del numero de atributos en los diferentes archivos csv:")
    print(df_info["num_columns"].value_counts())
    df_reference_csv = pd.read_csv(reference_csv, nrows=0)
    num_columns_reference = len(df_reference_csv.columns)
    if len(df_info["num_columns"].value_counts()) != 1:
        print("\nLos ficheros que tienen un numero diferente de atributos son:")
        print(df_info[df_info["num_columns"] != num_columns_reference])
    return df_info


def check_file_consistency(csv_files, reference_csv):
    """Ahora que ya sabemos cuantos atributos tiene cada fichero csv, debemos garantizar que todos tengan los mismos atributos, en el mismo orden
       y con los mismos tipos. Para ello se elige un csv de referencia """
    df_ref_header = pd.read_csv(reference_csv,
                                nrows=100)  # Leemos 100 filas para inferir correctamente los tipos de los atributos
    reference_columns = df_ref_header.columns.tolist()
    reference_types = df_ref_header.dtypes.to_dict()  # Tipos del dataframe de referencia

    print("\nAtributos y tipos del csv de referencia (CIC-BCCC-ACI-IOT-2023):")
    for col, dt in reference_types.items():
        print(f" - {col}: {dt}")
    print("\n")

    mismatched_files = []  # Array para almacenar cuales son los archivos que no cumplen consistencia
    for csv_path in csv_files:
        if csv_path == reference_csv:
            continue

        # De la misma manera leemos atributos y tipos de cada uno de los csv
        df_tmp_header = pd.read_csv(csv_path, nrows=10)
        tmp_columns = df_tmp_header.columns.tolist()

        if len(tmp_columns) != len(reference_columns):  # Si tienen mas o menos atributos
            mismatched_files.append((csv_path, "Tiene diferente numero de atributos"))
            continue

        if tmp_columns != reference_columns:
            mismatched_files.append((csv_path, "Diferente orden en los atributos o diferente nombres"))
            continue

        tmp_dtypes = df_tmp_header.dtypes.to_dict()
        for col in reference_columns:
            if reference_types[col] != tmp_dtypes[col]:
                mismatched_files.append((csv_path,
                                         f"Tipo diferente para el atributo '{col}' (Tipo de referencia: {reference_types[col]}, Tipo en el csv: {tmp_dtypes[col]})"))

    return mismatched_files


def run_consistency_check(base_folder: str):
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
    base_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024"
    #base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024-MOD"

    # Check dataset consistency
    run_consistency_check(base_folder)

if __name__ == "__main__":
    main()