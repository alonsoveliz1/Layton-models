import pandas as pd
from a_verificar_consistencia import find_csv_files


def main():
    base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    files = find_csv_files(base_folder)
    df_combinado = pd.DataFrame()

    for csv_file in files:
        df_combinado = pd.concat([df_combinado, pd.read_csv(csv_file)], ignore_index=True)

    # Eliminar columnas con valores a 0
    df_combinado = df_combinado.drop(["Bwd PSH Flags", "Bwd URG Flags", "Fwd Bytes/Bulk Avg","Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg"], axis = 1)
    df_combinado.to_csv(base_folder + "\\combinado.csv", index=False)

    # A partir de este dataset iteraremos para cada layer del clasificador creando conjuntos de datos especificos para cada uno de ellos
if __name__ == "__main__":
    main()