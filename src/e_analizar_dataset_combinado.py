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
        
        
        # Análisis de valores nulos
        columns_with_nulls = dataframe_final.columns[dataframe_final.isnull().any()].tolist()
        if columns_with_nulls :
            f.write(f"\nAtributos con valores nulos: {columns_with_nulls}")
            rows_with_nulls = dataframe_final.isnull().any(axis=1).sum()
            f.write(f"\nFilas con valores nulos: {rows_with_nulls}")
        else:
            f.write("No hay atributos con valores nulos\n")

        # Análisis de valores negativos
        numeric_cols = dataframe_final.select_dtypes(include=['int64', 'float64']).columns # Filtramos solo para columnas numericas
        sum_rows_with_negatives = dataframe_final[numeric_cols].lt(0).any(axis=1).sum()
        f.write(f"Numero de filas con valores negativos: {sum_rows_with_negatives}")

        if sum_rows_with_negatives > 0:
            dataframe_final_negative_mask = dataframe_final[numeric_cols] < 0
            negative_columns = numeric_cols[dataframe_final_negative_mask.any()]
            f.write(f"Los atributos que tienen valores negativos son {list(negative_columns)}: \n")
            del dataframe_final_negative_mask

        ### ANALISIS DE LOS ATRIBUTOS DEL DATASET PARA VER QUE VALORES TIENEN


        """ESTADISTICAS DE LOS FLUJOS"""
        flow_cols = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Flow Bytes/s', 'Flow Packets/s']
        f.write("\nEstadisticas de los fluijos TCP:")
        f.write("-" * 30)
        f.write(dataframe_final[flow_cols].describe().to_string())

        """ESTADISTICAS DE LOS PAQUETES"""
        packet_cols = ['Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'Average Packet Size', 'Fwd Act Data Pkts']
        packet_cols_f = ['Total Length of Fwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Fwd Packets/s']
        packet_cols_b = ['Total Length of Bwd Packet', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std','Bwd Packets/s']

        f.write("\nEstadisticas de paquetes TCP:")
        f.write(dataframe_final[packet_cols].describe().to_string())
        f.write("\nEstadisticas de paquetes TCP (Forward):")
        f.write(dataframe_final[packet_cols_f].describe().to_string())
        f.write("\n+ Estadisticas de paquetes TCP (Backward):")
        f.write(dataframe_final[packet_cols_b].describe().to_string())
        f.write("-" * 30)

        """ESTADISTICAS DE LAS CABECERAS"""
        header_cols = ['Fwd Header Length', 'Bwd Header Length']
        f.write("\nEstadisticas de las cabeceras de los paquetes:")
        f.write(dataframe_final[header_cols].describe().to_string())

        """ESTADISTICAS DE FLAGS"""
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                     'CWR Flag Count', 'ECE Flag Count']
        flag_cols_f = ['Fwd PSH Flags', 'Fwd URG Flags']
        flag_cols_b = ['Bwd PSH Flags', 'Bwd URG Flags']

        f.write("\nEstadisticas de las flags de los paquetes:")
        f.write(dataframe_final[flag_cols].describe().to_string())
        f.write("\nEstadisticas de las flags de los paquetes (Forward):")
        f.write(dataframe_final[flag_cols_f].describe().to_string())
        f.write("\nEstadisticas de las flags de los paquetes (Backward):")
        f.write(dataframe_final[flag_cols_b].describe().to_string())
        f.write("-" * 30)

        """ESTADISTICAS IAT (Inter Arrival Time)"""
        iat_cols = ['Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std']
        iat_cols2 = ['Bwd IAT Max', 'Bwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Std']

        f.write("\nEstadisticas de Inter Arrival Time:")
        f.write(dataframe_final[iat_cols].describe().to_string())
        f.write("\nEstadisticas de Inter Arrival Time 2:")
        f.write(dataframe_final[iat_cols2].describe().to_string())
        f.write("-" * 30)

        """ESTADISTICAS DE LA VENTANA"""
        # CANTIDAD DE DATOS QUE EL DESPOSITIVO DE DESTINO VA A PODER PROCESAR
        window_cols = ['FWD Init Win Bytes', 'Bwd Init Win Bytes']
        f.write("\nEstadisticas del tamaño de ventana:")
        f.write("-" * 30)
        f.write(dataframe_final[window_cols].describe().to_string())

        """ESTADISTICAS DE ACTIVIDAD"""
        activity_cols = ['Active Mean', 'Active Std', 'Active Max', 'Active Min',
                         'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min','Down/Up Ratio']
        f.write("\nEstadisticas de actividad del flujo:")
        f.write("-" * 30)
        f.write(dataframe_final[activity_cols].describe().to_string())

        """ESTADISTICAS DE SEGMENTO"""
        segment_cols = ['Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Seg Size Min']
        f.write("\nEstadisticas de segmento:")
        f.write("-" * 30)
        f.write(dataframe_final[segment_cols].describe().to_string())

        """ESTADISTICAS DE TRANSMISION EN BLOQUES (RAFAGAS)"""
        bulk_cols = ['Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg']
        f.write("\nEstadisticas de rafagas de transmision:")
        f.write("-" * 30)
        f.write(dataframe_final[bulk_cols].describe().to_string())

        """ESTADISTICAS DE SUBFLOWS"""
        subflow_cols = ['Subflow Fwd Packets', 'Subflow Bwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes']
        f.write("\nEstadisticas de los sub-flujos:")
        f.write("-" * 30)
        f.write(dataframe_final[subflow_cols].describe().to_string())

        # Al analizar cuales eran las correlaciones entre atributos se ha visto que hay atributos que tienen 0 en todas sus  filas! En todos los datasets
        # Por lo que parece que CICFlowMeter no recoge bien esta informacion
        """ESTADISTICAS DE ATRIBUTOS CON VALORES EXTRAÑOS"""
        atributos_extraños = ["Bwd PSH Flags", "Bwd URG Flags", "Fwd Bytes/Bulk Avg",
                            "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg"]
        f.write("\nEstadisticas de los flujos anormales (Todos los valores son 0):")
        f.write("-" * 30)
        f.write(dataframe_final[atributos_extraños].describe().to_string())

        # Distribución de etiquetas
        f.write("\nDistribucion del tipo de trafico en el dataset:")
        f.write(dataframe_final["Label"].value_counts(dropna=False).to_string())
        f.write("\nDistribucion del tipo de ataque en el dataset:")
        f.write(dataframe_final["Attack Name"].value_counts(dropna=False).to_string())

        f.write("\nPuertos de destino más frecuentes:")
        f.write("-" * 30)
        
        # Numero de filas benignas vs filas maliciosas
        print("Extrayendo cantidad de instancias de flujos benignos frente a maliciosos... \n")
        num_benign = len(dataframe_final[dataframe_final["Label"] == 0])
        num_malicious = len(dataframe_final[dataframe_final["Label"] == 1])
        f.write(f"Filas benignas: {num_benign}\n")
        f.write(f"Filas maliciosas: {num_malicious}\n\n")

        # Numero de filas benignas vs filas maliciosas
        print("Extrayendo cantidad de clases en el dataset... \n")
        for categoria in dataframe_final["Attack Category"].unique():
            f.write(f"Filas de la categoria {categoria}: {len(dataframe_final[dataframe_final['Attack Category'] == categoria])}\n")

        # Ahora vamos a indagar sobre los tipos de ataques que hay en el dataset combinado
        # Creo un nuevo dataframe_final con solo los ataques:

        print("Contando el numero de ataques del dataset final...\n")
        dataframe_final_malicioso = dataframe_final[dataframe_final["Label"] == 1]
        num_ataques = dataframe_final_malicioso["Attack Name"].nunique()
        f.write(f"Cantidad de ataques diferentes: {num_ataques}\n")

        print("Contando cuantas instancias tiene cada tipo de ataque... \n")
        f.write("Distribucion los tipos de ataque en el dataset\n:")
        total_malicioso = dataframe_final_malicioso["Attack Name"].value_counts()
        f.write(f"{dataframe_final_malicioso['Attack Name'].value_counts()}\n") # Cuenta los valores para cada ataque

        # Calculo del desequilibrio de clases
        print("Calculando ratios de desequilibrio de clases... \n")
        ratio = num_benign / num_malicious
        f.write(f"\nRatio trafico benigno/malicioso: {ratio}\n")

        porcentaje_malicioso_total = dataframe_final_malicioso["Attack Name"].value_counts(normalize=True) * 100
        f.write("Porcentaje de cada tipo de ataque respecto al total de ataques:\n")
        for ataque, porcentaje in porcentaje_malicioso_total.items():
            f.write(f" - {ataque}: {porcentaje:.2f}% ({total_malicioso[ataque]} instancias)")
            f.write("\n\n")

        # Analisis de las filas duplicadas en el dataset
        print("Examinando si existen filas duplicadas en el dataset...\n")
        df_final_dup = dataframe_final.duplicated()
        sum_dup = df_final_dup.sum()
        df_malicioso_dup = dataframe_final_malicioso.duplicated().sum()
        dataframe_final_malicioso_benign = sum_dup - df_malicioso_dup

        f.write(f"Numero de filas duplicadas: {sum_dup} \n")
        f.write(f"Numero de filas duplicadas de trafico malicioso: {df_malicioso_dup} \n")
        f.write(f"Numero de filas benignas: {dataframe_final_malicioso_benign}\n\n")

        dataframe_final_duplicated = dataframe_final[dataframe_final.duplicated()]
        f.write("Distribucion de cuantas filas hay duplicadas por cada tipo de ataque \n")
        f.write(f" {dataframe_final_duplicated['Attack Name'].value_counts()}\n")

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
    if 'Attack Name' in dataframe_final.columns:
        dataframe_final_ataques = dataframe_final[dataframe_final['Label'] == 1]
        pd_series_ataques = dataframe_final_ataques['Attack Name'].value_counts() # Como diccionario {ataque:conteo}
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
    dataframe_final_benigno = dataframe_final[dataframe_final['Label'] == 0]
    pd_series_servicio = dataframe_final_benigno['Service'].value_counts()  # Como diccionario {ataque:conteo}
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


    # Paleta de colores
    colors = sns.color_palette("viridis", n_colors=num_categorias_servicio)

    plt.figure(figsize=(14, 8))
    ax = counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    plt.title("Distribución de 'Service' en el BCCC-NRC-TabularIoT-2024-TCP",
              fontsize=16, pad=20)
    plt.xlabel("Servicio", fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/Service.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Graficas para atributos numericos
    sample_size = min(500000, len(dataframe_final))
    sample_dataframe_final = dataframe_final.sample(sample_size, random_state=42)

    dataframe_final_numeric = sample_dataframe_final.select_dtypes(include=["float64", "int64"])
    for col in dataframe_final_numeric.columns:
        plt.figure(figsize=(15, 4)) # Tamanyo del lienzo
        # Elaboracion del histograma y configuracion de sus caracteristicas
        plt.subplot(1, 2, 1)
        dataframe_final_numeric[col].hist(grid=False, color="royalblue", edgecolor="black")
        plt.title(f"Histograma de {col}")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=dataframe_final_numeric[col], width=0.5, showmeans=True,
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


def generar_boxplots_por_categoria_ataque(dataframe_final: pd.DataFrame) -> None:
    """Genera boxplots de cada atributo numérico comparando por 'Attack Category'"""

    print("GENERANDO BOXPLOTS POR CATEGORIA DE ATAQUE")
    print("*" * 50)

    # Asegurar que Attack Category está presente
    if 'Attack Category' not in dataframe_final.columns:
        print(
            "La columna 'Attack Category' no se encuentra en el DataFrame. Se omiten boxplots por categoría de ataque.")
        return

    # Seleccionar solo atributos numéricos
    numeric_cols = dataframe_final.select_dtypes(include=["float64", "int64"]).columns
    dataframe_final_for_plotting = dataframe_final[list(numeric_cols) + ['Attack Category']]

    os.makedirs("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/boxplots_por_categoria_ataque", exist_ok=True)

    # Obtener número de categorías para la paleta de colores
    num_categories = dataframe_final_for_plotting['Attack Category'].nunique()
    colors = sns.color_palette("viridis", n_colors=num_categories)

    for col in numeric_cols:
        plt.figure(figsize=(14, 8))

        # Crear boxplot con estilo profesional
        ax = sns.boxplot(data=dataframe_final_for_plotting, x='Attack Category', y=col,
                         palette=colors,
                         showmeans=True,
                         meanprops={"marker": "D", "markerfacecolor": "white",
                                    "markeredgecolor": "black", "markersize": "6"},
                         flierprops={"marker": "o", "markerfacecolor": "lightcoral",
                                     "markeredgecolor": "darkred", "markersize": "3", "alpha": 0.6},
                         boxprops={"edgecolor": "black", "linewidth": 1.2},
                         whiskerprops={"color": "black", "linewidth": 1.2},
                         medianprops={"color": "darkblue", "linewidth": 2},
                         capprops={"color": "black", "linewidth": 1.2})

        # Configurar títulos y etiquetas
        plt.title(f"Distribución de {col} por Categoría de Ataque",
                  fontsize=16, pad=20)
        plt.xlabel("Categoría de Ataque", fontsize=14)
        plt.ylabel(col, fontsize=14)

        # Mejorar legibilidad de etiquetas
        plt.xticks(rotation=45, ha='right')

        # Añadir grid sutil
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Formatear eje Y si los valores son grandes
        if dataframe_final_for_plotting[col].max() > 1000:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        plt.tight_layout()

        # Guardar con nombre válido y alta calidad
        valid_col_name = col.replace("/", "_").replace("\\", "_")
        filename = f"boxplot_{valid_col_name}.png"
        filepath = os.path.join("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/boxplots_por_categoria_ataque",
                                filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    print("*" * 50)
    print("BOXPLOTS POR CATEGORIA DE ATAQUE GUARDADOS EN:")
    print("../analysis/diagrams/CIC-BCC-NRC-TabularIoT-2024/boxplots_por_categoria_ataque/")
    print("*" * 50)


def main():
    dataframe_final = pd.read_csv("../data/raw/CIC-BCCC-NRC-TabularIoT-2024/combinado.csv", low_memory=False)
    extraer_caracteristicas_txt(dataframe_final)
    #procesar_dataset_graficos(dataframe_final)
    #generar_boxplots_por_categoria_ataque(dataframe_final)
main()
