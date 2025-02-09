import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def process_all_datasets(base_folder: str) -> None:
    """Metodo lanzadera para procesar cada uno de los datasets a nivel individual."""

    """Queremos generar graficos con un color diferente para cada tipo de ataque, y que en caso de que ese tipo de ataque se repita
        entre csvs, que el color sea el mismo por lo que utilizaremos un diccionario"""

    print("Getting all attack types in the CIC-BCCC-NRC-TabularIoT-2024")
    all_attacks = get_all_attack_types(base_folder)
    colors = generate_extended_colors(len(all_attacks))

    attack_colors = dict(zip(all_attacks, colors))

    # Process each dataset
    for subfolder in sorted(os.listdir(base_folder)):
        process_single_dataset(base_folder, subfolder,attack_colors)


def get_all_attack_types(base_folder: str):
    """Metodo para saber que tipos de ataque hay en cada dataset."""
    all_attacks = set()
    for subfolder in sorted(os.listdir(base_folder)):
        dataset_path = os.path.join(base_folder, subfolder)
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

        for csv_file in csv_files:
            df_temp = pd.read_csv(csv_file, low_memory=False)
            all_attacks.update(df_temp[df_temp['Label'] != 0]['Attack Name'].unique())
            del df_temp

    print(f"\nEn el dataset se presentan los siguientes tipos de ataque: {all_attacks}")
    return all_attacks



def generate_extended_colors(n_attacks: int) -> np.ndarray:
    """Metodo para generar un conjunto de colores mas amplio, ya que disponemos de 48 tipos de ataque no nos sirve con una unica paleta."""
    colormaps = [
        plt.colormaps['tab20'](np.linspace(0, 1, 20)),
        plt.colormaps['Set3'](np.linspace(0, 1, 12)),
        plt.colormaps['Set2'](np.linspace(0, 1, 8)),
        plt.colormaps['Paired'](np.linspace(0, 1, 12))
    ]

    all_colors = np.vstack(colormaps)

    if n_attacks > len(all_colors):
        additional_colors = plt.colormaps['hsv'](np.linspace(0, 1, n_attacks - len(all_colors)))
        all_colors = np.vstack([all_colors, additional_colors])

    np.random.shuffle(all_colors)
    return all_colors



def process_single_dataset(base_folder, subfolder: str, attack_colors) -> None:
    """El CIC-BCCC-ACI-IOT-2023, esta compuesto por 9 datasets, por lo que es interesante obtener informacion
       a nivel individual de cada uno de ellos. Metodo lanzadera para cada dataset."""

    dataset_path = os.path.join(base_folder, subfolder)
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

    dataframes = []
    for csv_file in csv_files:
        df_temp = pd.read_csv(csv_file)
        dataframes.append(df_temp)

    # Creamos un df con todos los csv de un unico dataset
    df_subfolder = pd.concat(dataframes)

    # Ejecucion de cada parte del analisis
    write_dataset_info(df_subfolder, subfolder) # Informacion general en formato txt

    if 'Service' in df_subfolder.columns:
        plot_service_attacked(df_subfolder, subfolder) # Distribucion de los servicios que reciben trafico

    plot_label_distribution(df_subfolder, subfolder) # Distribucion del tipo de trafico de cada dataset (1 = malicioso 0 = benigno)
    plot_attack_distribution(df_subfolder, subfolder,attack_colors) # Distribucion del tipo de ataque dentro de cada dataset
    boxplot_flow_duration(df_subfolder, subfolder) # Boxplot
    plot_numerical_attributes_distribution(df_subfolder, subfolder)



def write_dataset_info(df: pd.DataFrame, subfolder: str) -> None:
    """Vamos a guardar informacion individual de cada dataset en un txt, numero de filas, numero de filas duplicadas,
       distribucion de Label (tambien se hara mas adelante una grafica), valores nulos dentro del dataset y distribucion
       de los tipos de ataque presentes en el dataset. Tambien vamos a analizar para cada variable numerica sus outliers"""

    # Creamos un archivo txt con el nombre del dataset para escribir
    file_path = f'../analysis/text/{subfolder}_analysis.txt'
    with open(file_path, 'w') as f:
        pd.options.display.min_rows = 100
        def write_to_file(text):
            f.write(str(text) + '\n')

        write_to_file("\n" + "#" * 50)
        write_to_file(f"DATASET INFORMATION: {subfolder}")
        write_to_file(f"#" * 50)

        # Info del dataset
        df.info(buf=f)

        # Información básica
        write_to_file(f"\nTotal rows: {df.shape[0]}")
        write_to_file(f"Duplicate rows: {df.duplicated().sum()}")

        # Análisis de valores nulos
        columns_with_nulls = df.columns[df.isnull().any()].tolist()
        if columns_with_nulls :
            write_to_file(f"\nAttributes with missing values: {columns_with_nulls}")
            rows_with_nulls = df.isnull().any(axis=1).sum()
            write_to_file(f"\nRows with null values: {rows_with_nulls}")
        else:
            write_to_file("There are no attributes with missing values\n")

        # Análisis de valores negativos
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        sum_rows_with_negatives = df[numeric_cols].lt(0).any(axis=1).sum()
        write_to_file(f"Number of rows with negative values: {sum_rows_with_negatives}")

        if(sum_rows_with_negatives > 0):
            df_negative_mask = df[numeric_cols] < 0
            negative_columns = numeric_cols[df_negative_mask.any()]
            write_to_file(f"Attribues with negative values {list(negative_columns)}: \n")
            del df_negative_mask

        # Análisis temporal
        write_to_file("\nTIMESTAMP ANALYSIS:")
        write_to_file("-" * 90)
        if df['Timestamp'].dtype == 'object':
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
        write_to_file(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

        # Estadísticas de flujo
        flow_cols = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
                     'Flow Bytes/s', 'Flow Packets/s']
        write_to_file("\nBASIC FLOW STATISTICS:")
        write_to_file("-" * 30)
        write_to_file(df[flow_cols].describe().to_string())

        # Estadísticas de paquetes
        packet_cols = ['Total Length of Fwd Packet', 'Total Length of Bwd Packet',
                       'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
                       'Packet Length Std', 'Average Packet Size']
        write_to_file("\nPACKET STATISTICS:")
        write_to_file("-" * 30)
        write_to_file(df[packet_cols].describe().to_string())

        # Estadísticas de flags
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
                     'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                     'CWR Flag Count', 'ECE Flag Count']
        write_to_file("\nFLAG COUNTS:")
        write_to_file("-" * 30)
        write_to_file(df[flag_cols].agg(['sum', 'mean']).to_string())

        # IAT estadísticas
        iat_cols = ['Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                    'Fwd IAT Mean', 'Bwd IAT Mean']
        write_to_file("\nINTER ARRIVAL TIME STATISTICS:")
        write_to_file("-" * 30)
        write_to_file(df[iat_cols].describe().to_string())

        # Estadísticas de ventana
        window_cols = ['FWD Init Win Bytes', 'Bwd Init Win Bytes']
        write_to_file("\nWINDOW SIZE STATISTICS:")
        write_to_file("-" * 30)
        write_to_file(df[window_cols].describe().to_string())

        # Estadísticas de actividad
        activity_cols = ['Active Mean', 'Active Std', 'Active Max', 'Active Min',
                         'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
        write_to_file("\nACTIVITY STATISTICS:")
        write_to_file("-" * 30)
        write_to_file(df[activity_cols].describe().to_string())

        # Distribución de etiquetas
        write_to_file("\nLabel distribution:")
        write_to_file(df["Label"].value_counts(dropna=False).to_string())
        write_to_file("\nAttack Name distribution:")
        write_to_file(df["Attack Name"].value_counts(dropna=False).to_string())

        write_to_file("\n" + "#" * 50 + "\n")

        # Valores anomalos -- POR IMPLEMENTAR
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
        info_valores_anomalos = []

    print(f"Analysis saved to: analysis/{subfolder}_analysis.txt")



"""POR IMPLEMENTAR: HACER QUE SE GUARDE EN SU RESPECTIVA CARPETA"""
def plot_label_distribution(df: pd.DataFrame, subfolder: str) -> None:
    """Metodo para generar un grafico sobre la distribucion del atributo Label (tipo de conexión)."""

    plt.figure(figsize=(10, 6))
    counts = df['Label'].value_counts()
    colors = ['red','green']

    ax = counts.plot(kind='bar', color=colors)
    plt.title(f"Label Distribution in {subfolder}")
    plt.xlabel("Connection Type (1 = malicious, 0 = benign)")
    plt.ylabel("Frequency")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



"""POR IMPLEMENTAR: HACER QUE SE GUARDE EN SU RESPECTIVA CARPETA"""
def plot_attack_distribution(df: pd.DataFrame, subfolder: str, attack_colors) -> None:
    """Metodo para generar un grafico sobre la distribucion del atributo "Tipo de Ataque"."""

    df_attacks = df[df['Label'] != 0]
    counts = df_attacks['Attack Name'].value_counts()
    current_colors = [attack_colors[attack] for attack in counts.index]

    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind='bar', color=current_colors, width=0.8)
    plt.title(f"Attack Type Distribution in {subfolder}")
    plt.xlabel("Attack Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.show()
    del df_attacks



"""POR IMPLEMENTAR: HACER QUE SE GUARDE EN SU RESPECTIVA CARPETA"""
def plot_service_attacked(df: pd.DataFrame, subfolder: str) -> None:
    """Metodo para generar un grafico sobre la distribucion del atributo "Tipo de Ataque"."""
    df_attacks = df[df['Label'] != 0]
    counts = df_attacks['Service'].value_counts()

    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind='bar', width=0.8)
    plt.title(f"Service distribution in {subfolder}")
    plt.xlabel("Service")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.show()



"""POR IMPLEMENTAR: HACER QUE SE GUARDE EN SU RESPECTIVA CARPETA"""
def boxplot_flow_duration(df: pd.DataFrame, subfolder: str) -> None:
    """Metodo para generar un boxplot para los valores de "Flow Duration" en cada dataset"""
    df_seconds = df.copy()
    df_seconds["Flow Duration"] = df_seconds["Flow Duration"] / 1000000
    plt.figure(figsize=(10, 8))

    ax = df_seconds.boxplot(column="Flow Duration", showmeans=True, vert=False,
                    meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"red", "markersize":10},
                    flierprops={"marker":"o", "markerfacecolor":"gray", "markersize":4},
                    medianprops={"color":"red"},
                    boxprops={"color":"black"},
                    whiskerprops={"color":"black"})

    ax.set_xlabel("Duración (segundos)")
    ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    plt.title(f"Duracion del flujo de comunicacion en {subfolder} ")
    plt.show()



def plot_numerical_attributes_distribution(df: pd.DataFrame, subfolder: str) -> None:
    carpeta_proyecto = "C:\\Users\\avelg\\PycharmProjects\\NIDS"
    target_path = os.path.join(carpeta_proyecto,"analysis","diagrams",subfolder)

    os.makedirs(target_path, exist_ok=True)
    sample_df = df.sample(n = 10000, random_state = 42)
    df_numeric = sample_df.select_dtypes(include=["float64", "int64"])

    for col in df_numeric.columns:
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        df_numeric[col].hist(grid=False)
        plt.ylabel("Count")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df_numeric[col], width=0.5, showmeans=True,
                    meanprops={"marker":"o",
                          "markerfacecolor":"white",
                          "markeredgecolor":"black",
                          "markersize":"10"},
                    flierprops={"marker":"D",
                           "markerfacecolor":"red",
                           "markeredgecolor":"black",
                           "markersize":"5"},
                    medianprops={"color": "red"},
                    boxprops={"facecolor": "skyblue",
                         "edgecolor": "black"},
                    whiskerprops={"color": "black",
                            "linestyle": "--"})

        plt.title(f"Distribucion de {col} en {subfolder}")

        valid_col_name = col.replace("/", "_").replace("\\", "_") # Para el atributo Flow Bytes/s por el tema de la barra en el path de windows
        filename = f"{valid_col_name}.png"
        filepath = os.path.join(target_path, filename)
        plt.savefig(filepath)

        plt.close()

    print(f"Saved the diagrams for {subfolder}")



def main():
    base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    # Solo evaluar el dataset ya procesado, en caso contrario explota ya que hay csv que en el campo 'Date' tienen diferente formato
    process_all_datasets(base_folder)


if __name__ == "__main__":
    main()
