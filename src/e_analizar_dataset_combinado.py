import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)



def extraer_caracteristicas_txt(dataframe_final: pd.DataFrame) -> None:
    """Metodo para extraer caracteristicas del dataset combinado y ver si se ha procesado correctamente
       CARACTERISTICAS: tipos attr, distribucion maligno/benigno, distribucion ataques, desequilibrio de clases
       filas duplicadas y filas duplicadas por cada tipo de ataque"""

    print("COMENZANDO CON LA EXTRACCION DE CARACTERISTICAS DEL DATASET")
    print("*" * 50)

    file_path = f'../analysis/text/CIC-BCCC-NRC-TabularIoT-2024-MOD.txt'
    with open(file_path, 'w') as f:

        # Comprobacion de que los tipos se han inferido correctamente
        print("Comprobando que los tipos del dataset final se han inferido correctamente...\n")
        reference_dtypes = dataframe_final.dtypes.to_dict()
        f.write("Tipos de cada uno de los atributos:\n")
        for col, dt in reference_dtypes.items():
            f.write(f" - {col}: {dt} ")
            f.write("\n")

        # Numero de filas benignas vs filas maliciosas
        print("Extrayendo cantidad de instancias de flujos benignos frente a maliciosos... \n")
        num_benign = len(dataframe_final[dataframe_final["Label"] == 0])
        num_malicious = len(dataframe_final[dataframe_final["Label"] == 1])
        f.write(f"Filas benignas: {num_benign}\n")
        f.write(f"Filas maliciosas: {num_malicious}\n\n")

        # Numero de filas benignas vs filas maliciosas
        print("Extrayendo cantidad de clases en el dataset... \n")
        for categoria in dataframe_final["Attack Category"].unique():
            f.write(f"Filas de la categoria {categoria}: {len(dataframe_final[dataframe_final["Attack Category"] == categoria])}\n")

        # Ahora vamos a indagar sobre los tipos de ataques que hay en el dataset combinado
        # Creo un nuevo df con solo los ataques:

        print("Contando el numero de ataques del dataset final...\n")
        df_malicioso = dataframe_final[dataframe_final["Label"] == 1]
        num_ataques = df_malicioso["Attack Name"].nunique()
        f.write(f"Cantidad de ataques diferentes: {num_ataques}\n")

        print("Contando cuantas instancias tiene cada tipo de ataque... \n")
        f.write("Distribucion los tipos de ataque en el dataset\n:")
        total_malicioso = df_malicioso["Attack Name"].value_counts()
        f.write(f"{df_malicioso["Attack Name"].value_counts()}\n") # Cuenta los valores para cada ataque

        # Calculo del desequilibrio de clases
        print("Calculando ratios de desequilibrio de clases... \n")
        ratio = num_benign / num_malicious
        f.write(f"\nRatio trafico benigno/malicioso: {ratio}\n")

        porcentaje_malicioso_total = df_malicioso["Attack Name"].value_counts(normalize=True) * 100
        f.write("Porcentaje de cada tipo de ataque respecto al total de ataques:\n")
        for ataque, porcentaje in porcentaje_malicioso_total.items():
            f.write(f" - {ataque}: {porcentaje:.2f}% ({total_malicioso[ataque]} instancias)")
            f.write("\n\n")

        # Analisis de las filas duplicadas en el dataset
        print("Examinando si existen filas duplicadas en el dataset...\n")
        df_dup = dataframe_final.duplicated()
        sum_dup = df_dup.sum()
        df_malicioso_dup = df_malicioso.duplicated().sum()
        df_malicioso_benign = sum_dup - df_malicioso_dup

        f.write(f"Numero de filas duplicadas: {sum_dup} \n")
        f.write(f"Numero de filas duplicadas de trafico malicioso: {df_malicioso_dup} \n")
        f.write(f"Numero de filas benignas: {df_malicioso_benign}\n\n")

        df_duplicated = dataframe_final[dataframe_final.duplicated()]
        f.write("Distribucion de cuantas filas hay duplicadas por cada tipo de ataque \n")
        f.write(f" {df_duplicated['Attack Name'].value_counts()}\n")

        print("*" * 50)
        print(f"EXTRACCION DE CARACTERISTICAS DEL DATASET COMPLETADA, FICHERO GUARDADO EN\n")
        print(f"../analysis/text/CIC-BCCC-NRC-TabularIoT-2024-MOD.txt\n")
        print("*" * 50)



def procesar_dataset_graficos(dataframe_final: pd.DataFrame) -> None:
    """Metodo para generar 3 graficos de los atributos categoricos del dataset combinado (Distribuciones Label, Attack Type
       y service). Y histogramas y boxplots para los atributos numericos"""

    print("COMENZANDO CON LA GENERACION DE GRAFICOS RESUMEN DEL DATASET")
    print("*" * 50)

    # Primera grafica: distribución de los tipos de flujo (1 = malicioso, 0 = benigno)
    plt.figure(figsize=(10, 6))
    counts = dataframe_final['Label'].value_counts()
    colors = ['green' if idx == 0 else 'red' for idx in counts.index]
    ax = counts.plot(kind='bar', color=colors)
    plt.title(f"Distribución de 'Label' en el Dataset CIC-BCCC-NRC-TabularIoT-2024-TCP")
    plt.xlabel("Tipo de conexión (1 = malicioso, 0 = benigno)")
    plt.ylabel("Frecuencia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=0)
    plt.tight_layout()
    os.makedirs("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/", exist_ok=True)
    plt.savefig("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/Label.png")
    plt.close()

    # Segunda grafica: distribucion de los tipos de ataque
    df_ataques = dataframe_final[dataframe_final['Label'] == 1]
    pd_series_ataques = df_ataques['Attack Name'].value_counts() # Como diccionario {ataque:conteo}
    num_categorias_ataque = len(pd_series_ataques.index)

    # Usamos la paleta turbo que tiene colores suficientes para los 40 tipos de ataque
    cmap = plt.get_cmap('turbo', num_categorias_ataque)
    color_list = [cmap(i) for i in range(num_categorias_ataque)]

    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind='bar', color=color_list)
    plt.title("Distribución de 'Attack Name' en conexiones maliciosas (Label=1)")
    plt.xlabel("Tipo de Ataque")
    plt.ylabel("Frecuencia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/Attack Type.png")
    plt.close()

    # Segunda grafica: distribucion de los servicios de trafico benigno
    df_benigno = dataframe_final[dataframe_final['Label'] == 0]
    pd_series_servicio = df_benigno['Service'].value_counts()  # Como diccionario {ataque:conteo}
    num_servicios = len(pd_series_servicio.index)

    # Usamos la paleta turbo que tiene colores suficientes para los tipos de servicio
    cmap = plt.get_cmap('turbo', num_categorias_ataque)
    color_list = [cmap(i) for i in range(num_servicios)]

    plt.figure(figsize=(10, 6))
    ax = pd_series_servicio.plot(kind='bar', color=color_list)
    plt.title("Distribución de 'Servicio' en conexiones benignas (Label=0)")
    plt.xlabel("Tipo de Servicio")
    plt.ylabel("Frecuencia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/Service Type Benign.png")
    plt.close()


    # Tercera grafica: distribucion de los servicios
    counts = dataframe_final['Service'].value_counts()
    num_categorias_servicio = len(counts.index)

    # Usamos la paleta que si tiene colores de sobra
    cmap = plt.get_cmap('tab20', num_categorias_servicio)
    color_list = [cmap(i) for i in range(num_categorias_servicio)]

    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind='bar', color=color_list)
    plt.title("Distribución de 'Service' en el BCCC-NRC-TabularIoT-2024-TCP")
    plt.xlabel("Servicio")
    plt.ylabel("Frecuencia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    # Ajustamos la rotación de las etiquetas del eje X para poder leer bien los servicios igual que antes
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/Service.png")
    plt.close()

    # Graficas para atributos numericos
    sample_size = min(500000, len(dataframe_final))
    sample_df = dataframe_final.sample(sample_size, random_state=42)

    df_numeric = sample_df.select_dtypes(include=["float64", "int64"])
    for col in df_numeric.columns:
        plt.figure(figsize=(15, 4)) # Tamanyo del lienzo
        # Elaboracion del histograma y configuracion de sus caracteristicas
        plt.subplot(1, 2, 1)
        df_numeric[col].hist(grid=False, color="royalblue", edgecolor="black")
        plt.title(f"Histograma de {col}")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df_numeric[col], width=0.5, showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "royalblue",
                               "markersize": "10"},
                    flierprops={"marker": "D", "markerfacecolor": "royalblue", "markeredgecolor": "black",
                                "markersize": "5"},
                    medianprops={"color": "royalblue"},
                    boxprops={"facecolor": "skyblue", "edgecolor": "black"},
                    whiskerprops={"color": "black", "linestyle": "--"})
        plt.title(f"Distribucion de {col} en CIC-BCC-NRC-TabularIoT-2024-TCP")


        # Para el atributo Flow Bytes/s por el tema de la barra en el path de windows
        valid_col_name = col.replace("/", "_").replace("\\", "_")
        filename = f"{valid_col_name}.png"
        filepath = os.path.join("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/", filename)
        plt.savefig(filepath)
        plt.close()

    print("*" * 50)
    print(f"GENERACION DE GRAFICOS COMPLETADA, FICHEROS GUARDADOS EN")
    print(f"../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/")
    print("*" * 50)

def main():
    df = pd.read_csv("../data/processed/CIC-BCCC-NRC-TabularIoT-2024-MOD/combinado.csv", low_memory=False)
    extraer_caracteristicas_txt(df)
    procesar_dataset_graficos(df)

main()
