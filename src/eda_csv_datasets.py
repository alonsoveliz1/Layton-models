import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Set
import seaborn as sns

class DatasetConsistencyChecker:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.csv_files = []

        self.reference_csv = os.path.join(base_folder, "CIC-BCCC-ACI-IOT-2023", "Benign Traffic.csv")

    #
    def find_csv_files(self):
        # Metodo para ver cuantos archivos CSV componen el CIC-BCCC-NRC-TabularIoT-2024
        for subdir, _, files in os.walk(self.base_folder):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(subdir, file)
                    self.csv_files.append(csv_path)
        print(f"El dataset CIC-BCCC-NRC-TabularIoT-2024 se compone de {len(self.csv_files)} CSV files.\n")

    def get_file_info(self, csv_file: str) -> Dict:
        # Recoger informacion de los archivos csv para luego comprobar consistencia {filename, attribute_count, attribute_name, dict {attr:type}}
        df = pd.read_csv(csv_file, nrows=10)
        return {
            "archivo": csv_file,
            "num_columns": df.shape[1],
            "column_names": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
        }

    def analyze_column_consistency(self):
        """Metodo para analizar cuantas columnas tiene cada fichero csv. Como ya sabemos que 85 son las esperadas, y que hay 2 archivos con 86
           se incorpora funcionalidad para saber cuales son exactamente los que difieren del resto"""
        info_csv_files = []
        for csv_file in self.csv_files:
            file_info = self.get_file_info(csv_file)
            if file_info:
                info_csv_files.append(file_info)

        df_info = pd.DataFrame(info_csv_files)
        print("\nDistribucion del numero de atributos en los diferentes archivos csv:")
        print(df_info["num_columns"].value_counts())
        df_reference_csv = pd.read_csv(self.reference_csv, nrows=0)
        num_columns_reference = len(df_reference_csv.columns)
        if len(df_info["num_columns"].value_counts()) != 1:
            print("\nHay ficheros que tienen un numero diferente de atributos:")
            print(df_info[df_info["num_columns"] != num_columns_reference])
        return df_info

    def check_file_consistency(self):
        """Ahora que ya sabemos cuantos atributos tiene cada fichero csv, debemos garantizar que todos tengan los mismos atributos, en el mismo orden
           y con los mismos tipos. Para ello se elige un csv de referencia """
        df_ref_header = pd.read_csv(self.reference_csv, nrows=100) # Leemos 100 filas para inferir correctamente los tipos de los atributos
        reference_columns = df_ref_header.columns.tolist()
        reference_types = df_ref_header.dtypes.to_dict() # Tipos del dataframe de referencia

        print("\nAtributos y tipos del csv de referencia (CIC-BCCC-ACI-IOT-2023):")
        for col, dt in reference_types.items():
            print(f" - {col}: {dt}")
        print("\n")

        mismatched_files = [] # Array para almacenar cuales son los archivos que no cumplen consistencia
        for csv_path in self.csv_files:
            if csv_path == self.reference_csv:
                continue

            # De la misma manera leemos atributos y tipos de cada uno de los csv
            df_tmp_header = pd.read_csv(csv_path, nrows=10)
            tmp_columns = df_tmp_header.columns.tolist()

            if len(tmp_columns) != len(reference_columns): # Si tienen mas o menos atributos
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

    def run_consistency_check(self):
        """Metodo para invocar al proceso de verifcar la consistencia y mostrar su resultado. En caso de que no haya consistencia se muestra
           cual es el archivo que causa el problema, y el motivo."""
        self.find_csv_files()
        self.analyze_column_consistency()
        mismatched_files = self.check_file_consistency()

        if not mismatched_files:
            print("Todos los CSV son consistentes en número de columnas, orden, tipo y tipo de los atributos.")
        else:
            print("Hay CSV que no son consistentes:")
            for file_path, reason in mismatched_files:
                print(f"- {file_path}: {reason}")


class DatasetProcessor:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        """Queremos generar graficos con un color diferente para cada tipo de ataque, y que en caso de que ese tipo de ataque se repita
        entre csvs, que el color sea el mismo por lo que utilizaremos un diccionario"""
        self.attack_colors: Dict = {}

    def process_all_datasets(self) -> None:
        """Metodo lanzadera para procesar cada uno de los datasets a nivel individual."""
        # Initialize attack colors
        all_attacks = self.get_all_attack_types()
        colors = self.generate_extended_colors(len(all_attacks))
        self.attack_colors.update(dict(zip(all_attacks, colors)))

        # Process each dataset
        for subfolder in sorted(os.listdir(self.base_folder)):
            self.process_single_dataset(subfolder)

    #
    def process_single_dataset(self, subfolder: str) -> None:
        """El CIC-BCCC-ACI-IOT-2023, esta compuesto por 9 datasets, por lo que es interesante obtener informacion
           a nivel individual de cada uno de ellos. Metodo lanzadera para cada dataset.
           Metodo lanzadera para procesar cada dataset a nivel individual"""

        dataset_path = os.path.join(self.base_folder, subfolder)
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

        dataframes = []
        for csv_file in csv_files:
            df_temp = pd.read_csv(csv_file)
            dataframes.append(df_temp)

        df_subfolder = pd.concat(dataframes)

        self._write_dataset_info(df_subfolder, subfolder)

        if 'Service' in df_subfolder.columns:
            self._plot_service_attacked(df_subfolder, subfolder)

        self._plot_label_distribution(df_subfolder, subfolder)
        self._plot_attack_distribution(df_subfolder, subfolder)
        self._boxplot_flow_duration(df_subfolder, subfolder)
        self.plot_numerical_attributes_distribution(df_subfolder, subfolder)

    def generate_extended_colors(self, n_colors: int) -> np.ndarray:
        """Metodo para generar un conjunto de colores mas amplio, ya que disponemos de 48 tipos de ataque no nos sirve con una unica paleta."""
        colormaps = [
            plt.colormaps['tab20'](np.linspace(0, 1, 20)),
            plt.colormaps['Set3'](np.linspace(0, 1, 12)),
            plt.colormaps['Set2'](np.linspace(0, 1, 8)),
            plt.colormaps['Paired'](np.linspace(0, 1, 12))
        ]

        all_colors = np.vstack(colormaps)

        if n_colors > len(all_colors):
            additional_colors = plt.colormaps['hsv'](np.linspace(0, 1, n_colors - len(all_colors)))
            all_colors = np.vstack([all_colors, additional_colors])

        np.random.shuffle(all_colors)
        return all_colors

    def get_all_attack_types(self):
        """Metodo para saber que tipos de ataque hay en cada dataset."""
        all_attacks = set()
        for subcarpeta in sorted(os.listdir(self.base_folder)):
            dataset_path = os.path.join(self.base_folder, subcarpeta)
            csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))

            for csv_file in csv_files:
                df_temp = pd.read_csv(csv_file, low_memory=False)
                all_attacks.update(df_temp[df_temp['Label'] != 0]['Attack Name'].unique())

        print(f"\nEn el dataset se presentan los siguientes tipos de ataque: {all_attacks}")
        return all_attacks

    def _write_dataset_info(self, df: pd.DataFrame, subfolder: str) -> None:
        """Vamos a guardar informacion individual de cada dataset en un txt, numero de filas, numero de filas duplicadas,
           distribucion de Label (tambien se hara mas adelante una grafica), valores nulos dentro del dataset y distribucion
           de los tipos de ataque presentes en el dataset. Tambien vamos a analizar para cada variable numerica sus outliers"""

        # Crear directorio 'analysis' si no existe
        os.makedirs('../analysis', exist_ok=True)

        # Abrir archivo para escribir
        with open(f'analysis/{subfolder}_analysis.txt', 'w') as f:
            # Redireccionar stdout al archivo
            def write_to_file(text):
                f.write(str(text) + '\n')

            write_to_file("\n" + "#" * 50)
            write_to_file(f"DATASET INFORMATION: {subfolder}")
            write_to_file("#" * 50)

            df.info(buf=f)

            # Información básica
            write_to_file(f"\nTotal rows: {df.shape[0]}")
            write_to_file(f"Duplicate rows: {df.duplicated().sum()}\n")
            write_to_file(f"Total null values:\n {df.isnull().sum()}\n")

            # Análisis de valores negativos
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            sum_rows_with_negatives = df[numeric_cols].lt(0).any(axis=1).sum()
            write_to_file(f"Number of rows with negative values: {sum_rows_with_negatives}\n")
            if(sum_rows_with_negatives > 0):
                df_negative_mask = df[numeric_cols] < 0
                negative_columns = numeric_cols[df_negative_mask.any()]
                write_to_file(f"Attribues with negative values {list(negative_columns)}: \n")



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

            # Valores anomalos
            df_numeric = df.select_dtypes(include=['int64', 'float64'])
            info_valores_anomalos = []

        # Imprimir confirmación en terminal
        print(f"Analysis saved to: analysis/{subfolder}_analysis.txt")

    def _plot_label_distribution(self, df: pd.DataFrame, subfolder: str) -> None:
        """Metodo para generar un grafico sobre la distribucion del atributo Label (tipo de conexión)."""
        plt.figure(figsize=(10, 6))
        counts = df['Label'].value_counts()
        colors = ['green' if idx == 0 else 'red' for idx in counts.index]

        ax = counts.plot(kind='bar', color=colors)
        plt.title(f"Label Distribution in {subfolder}")
        plt.xlabel("Connection Type (1 = malicious, 0 = benign)")
        plt.ylabel("Frequency")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def _plot_attack_distribution(self, df: pd.DataFrame, subfolder: str) -> None:
        """Metodo para generar un grafico sobre la distribucion del atributo "Tipo de Ataque"."""
        df_attacks = df[df['Label'] != 0]
        counts = df_attacks['Attack Name'].value_counts()
        current_colors = [self.attack_colors[attack] for attack in counts.index]

        plt.figure(figsize=(10, 6))
        ax = counts.plot(kind='bar', color=current_colors, width=0.8)
        plt.title(f"Attack Type Distribution in {subfolder}")
        plt.xlabel("Attack Type")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.show()

    def _plot_service_attacked(self, df: pd.DataFrame, subfolder: str) -> None:
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

    def _boxplot_flow_duration(self, df: pd.DataFrame, subfolder: str) -> None:
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

    def plot_numerical_attributes_distribution(self, df: pd.DataFrame, subfolder: str) -> None:
        carpeta_proyecto = "C:\\Users\\avelg\\PycharmProjects\\NIDS"
        target_path = os.path.join(carpeta_proyecto,"diagrams",subfolder)
        os.makedirs(target_path, exist_ok=True)
        sample_df = df.sample(n = 1000, random_state = 42)
        df_numeric = sample_df.select_dtypes(include=["float64", "int64"])

        for col in df_numeric.columns:
            plt.figure(figsize=(15, 4))
            plt.subplot(1, 2, 1)
            df_numeric[col].hist(grid=False)
            plt.ylabel("Count")
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df_numeric[col])
            plt.title(f"Distribucion de {col} en {subfolder}")

            valid_col_name = col.replace("/", "_").replace("\\", "_")
            filename = f"{valid_col_name}.png"
            filepath = os.path.join(target_path, filename)
            plt.savefig(filepath)

            plt.close()




def main():
    #base_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024"
    base_folder= "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024-MOD"

    # Check dataset consistency
    checker = DatasetConsistencyChecker(base_folder)
    checker.run_consistency_check()

    # Process and analyze datasets
    processor = DatasetProcessor(base_folder)
    processor.process_all_datasets()


if __name__ == "__main__":
    main()
