import pandas as pd
from a_verificar_consistencia import find_csv_files


def main():
    base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    files = find_csv_files(base_folder)
    df_combinado = pd.DataFrame()

    for csv_file in files:
        df_combinado = pd.concat([df_combinado, pd.read_csv(csv_file)], ignore_index=True, )

    df_combinado.to_csv(base_folder + "\\combinado.csv", index=False)


if __name__ == "__main__":
    main()