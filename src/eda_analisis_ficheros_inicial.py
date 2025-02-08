import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

attack_colors = {} # Diccionario para mappear cada tipo de ataque a un color

# Opciones de visualizacion de pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

# Ruta de la carpeta madre con los datasets
carpetaBCC = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024"

# Rutas de los ficheros csv
csv_files = []

############################################################################################################
# Función para obtener información global: número archivos csv, discrepancias en columnas,atributos y tipos#
############################################################################################################

def comprobar_consistencia_ficheros():
    # Carga de las rutas de los ficheros csv en el array csv_files
    for subdir, dirs, files in os.walk(carpetaBCC):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(subdir, file)
                csv_files.append(csv_path)

    print(f"Se han encontrado {len(csv_files)} archivos CSV.\n")

    # Ahora que sabemos con que ficheros trabajamos, necesitamos garantizar que la informacion que tenemos sea consistente entre los archivos csv
    # Creamos un vector para guardar informacion del nombre de los archivos, y del numero, nombre y tipo de las columnas
    info_csv_files = []

    for csv_file in csv_files:
        try:
            # Cargamos 10 filas para inferir el tipo de los atributos
            df = pd.read_csv(csv_file, nrows=10)
        except Exception as e:
            print(f"Error leyendo el archivo {csv_file}: {e}")
            continue

        num_columns = df.shape[1]
        column_names = list(df.columns)

        # Creamos un diccionario de la forma "nombre_columna : tipo columna"
        dtypes = df.dtypes.to_dict()

        info_csv_files.append({
            "archivo": csv_file,
            "num_columns": num_columns,
            "column_names": column_names,
            "dtypes": dtypes,
        })

    # Metemos de nuevo el array dentro de un DataFrame con Pandas
    df_info = pd.DataFrame(info_csv_files)

    # Mostramos cuantas columnas tiene el dataframe generado, en caso de que haya mas de un valor, hay archivos con diferentes atributos
    print(df_info["num_columns"].value_counts())

    # Podemos ver que hay 2 archivos csv que tienen 86 atributos mientras que el resto tienen 85, por lo que veamos cuales son
    print(df_info[df_info["num_columns"] != 85],"\n")

    # Podemos averiguar que atributo es comparando un csv de los que tienen 86 columnas con otro de los que tiene 85
        # expected_cols = set(df_menosCols.columns)
        # unexpected_cols = set(df_masCols.columns)
        # extra_cols = unexpected_cols - expected_cols
    # Y tras haberlo hecho podemos ver que hay un atributo extra que es "Device"
        # columna_deMas = "Device"

    # Ademas, queremos saber si todas las columnas estan en el mismo orden y tienen el mismo tipo de atributos
    # Escogemos uno de los csv con menos columnas como referencia

    ruta_csv_referencia = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024\\CIC-BCCC-ACI-IOT-2023\\Benign Traffic.csv"

    # 1. Cargamos el csv de referencia para inferir los tipos de cada columna
    df_ref_header = pd.read_csv(ruta_csv_referencia, nrows=100)
    reference_columns = df_ref_header.columns.tolist() # Extraemos los atributos que seran referencia
    reference_dtypes = df_ref_header.dtypes.to_dict()  # Y hacemos un diccionario de la forma {nombre_columna: dtype}

    print("\nAtributo:tipo del archivo de referencia:")
    for col, dt in reference_dtypes.items():
        print(f" - {col}: {dt}")
    print("\n")

    # 2. Creamos una lista para almacenar archivos con discrepancias sobre la referencia inicial
    mismatched_files = []

    # 3. Recorremos cada CSV y comparamos
    for csv_path in csv_files:
        if csv_path == ruta_csv_referencia:
            # Saltamos el CSV de referencia para no compararlo consigo mismo
            continue

        try:
            # Leemos las primeras filas para ver columnas y tipos
            df_tmp_header = pd.read_csv(csv_path, nrows=10)
        except Exception as e:
            # Si hay error leyendo, lo registramos y continuamos
            mismatched_files.append((csv_path, f"Error de lectura: {e}"))
            continue

        tmp_columns = df_tmp_header.columns.tolist()

        # 3.1) Commprobamos si las filas tienen el mismo numero de columnas
        if len(tmp_columns) != len(reference_columns):
            mismatched_files.append((csv_path, "Difiere en numero de columnas"))
            continue

        # 3.2) Comprobamos que el orden y los nombres son los mismos
        if tmp_columns != reference_columns:
            mismatched_files.append((csv_path, "Difiere en orden o nombres de columnas"))
            continue

        # 3.3) Comprobar que los tipos coinciden
        tmp_dtypes = df_tmp_header.dtypes.to_dict()  # pasamos de la misma manera los tipos del csv actual a un diccionario para comparar
        # Vamos a verificar columna por columna
        for col in reference_columns:
            ref_type = reference_dtypes[col]
            tmp_type = tmp_dtypes[col]

            # Si difieren, registramos la discrepancia
            if ref_type != tmp_type:
                mismatched_files.append((csv_path, f"Difiere el tipo de '{col}' (Ref: {ref_type}, Actual: {tmp_type})"))

    # 4. Mostramos los resultados del analisis
    if not mismatched_files:
        print("Todas las columnas coinciden en todos los CSV en número, nombre, orden y el tipo de sus atributos es el mismo.")
    else:
        print("Existen CSVs con discrepancias:")
        for file_path, reason in mismatched_files:
            print(f"- {file_path}: {reason}")

    # Como fin del resumen podemos ver que hay una discrepancia de tipos en el atributo BWD IAT Total



# Funcion para generar una paleta de colores mas grande (Tipo de ataque: color)
def generate_extended_colors(n_colors):
    # Combinamos varios colormaps
    colormap1 = plt.colormaps['tab20'](np.linspace(0, 1, 20))
    colormap2 = plt.colormaps['Set3'](np.linspace(0, 1, 12))
    colormap3 = plt.colormaps['Set2'](np.linspace(0, 1, 8))
    colormap4 = plt.colormaps['Paired'](np.linspace(0, 1, 12))

    # Combinamos todos los colormaps
    all_colors = np.vstack([colormap1, colormap2, colormap3, colormap4])

    # Aseguramos que tenemos suficientes colores
    if n_colors > len(all_colors):
        # Si necesitamos más colores, generamos variaciones
        additional_colors = plt.colormaps['hsv'](np.linspace(0, 1, n_colors - len(all_colors)))
        all_colors = np.vstack([all_colors, additional_colors])

    np.random.shuffle(all_colors) # Barajamos los colores para evitar colores similares consecutivos
    return all_colors



###########################################################################################################
# Tambien queremos generar informacion de cada uno de los datasets                                        #
###########################################################################################################


def process_datasets(carpetaBCC):
    # Creamos un set para guardar los tipos de ataque
    all_attacks = set()
    for subcarpeta in sorted(os.listdir(carpetaBCC)):
        ruta_dataset = os.path.join(carpetaBCC, subcarpeta)
        csv_files = glob.glob(os.path.join(ruta_dataset, "*.csv"))

    # Guardamos cada tipo de ataque que vemos en el dataset
        for csv_file in csv_files:
            try:
                df_temp = pd.read_csv(csv_file)
                all_attacks.update(df_temp[df_temp['Label'] != 0]['Attack Name'].unique())
            except Exception as e:
                print(f"Error leyendo {csv_file}: {e}")

    # Y generamos colores para todos los tipos de ataque
    colors = generate_extended_colors(len(all_attacks))
    attack_colors.update(dict(zip(all_attacks, colors)))

    # Procesamos cada dataset
    for subcarpeta in sorted(os.listdir(carpetaBCC)):
        ruta_dataset = os.path.join(carpetaBCC, subcarpeta)
        csv_files = glob.glob(os.path.join(ruta_dataset, "*.csv"))

        dataframes = []
        for csv_file in csv_files:
            try:
                df_temp = pd.read_csv(csv_file)
                dataframes.append(df_temp)
            except Exception as e:
                print(f"Error leyendo {csv_file}: {e}")

        df_subcarpeta = pd.concat(dataframes)

        print("")
        print("#" * 50)
        print(f"INFORMACIÓN DEL DATASET {subcarpeta}")
        print("#" * 50)
        print(f"\nFilas totales: {df_subcarpeta.shape[0]}")
        print(f"Filas duplicadas: {df_subcarpeta.duplicated().sum()}\n")
        print(f"Distribución de 'Label' :")
        print(df_subcarpeta["Label"].value_counts(dropna=False))

        nulos_totales = df_subcarpeta.isna().sum().sum()
        print(f"\nTotal de valores nulos: {nulos_totales}\n")

        print(f"Distribución de 'Attack Name':")
        print(df_subcarpeta["Attack Name"].value_counts(dropna=False))
        print("")



        # Primera grafica: distribución de los tipos de flujo (1 = malicioso, 0 = benigno)
        plt.figure(figsize=(10, 6))
        counts = df_subcarpeta['Label'].value_counts()
        colors = ['green' if idx == 0 else 'red' for idx in counts.index]
        ax = counts.plot(kind='bar', color=colors)
        plt.title(f"Distribución de 'Label' en el dataframe {subcarpeta}")
        plt.xlabel("Tipo de conexión (1 = malicioso, 0 = benigno)")
        plt.ylabel("Frecuencia")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Segunda gráfica: Distribución del tipo de ataque
        df_ataque = df_subcarpeta[df_subcarpeta['Label'] != 0]
        counts = df_ataque['Attack Name'].value_counts()

        # Crear lista de colores en el orden de los ataques actuales
        current_colors = [attack_colors[attack] for attack in counts.index]

        # Crear la figura para ataques
        plt.figure(figsize=(10, 6))
        ax = counts.plot(
            kind='bar',
            color=current_colors,
            width=0.8
        )

        plt.title(f"Distribución de 'Attack Name' en el dataframe {subcarpeta}")
        plt.xlabel("Tipo de Ataque")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        plt.show()

comprobar_consistencia_ficheros()
process_datasets(carpetaBCC)