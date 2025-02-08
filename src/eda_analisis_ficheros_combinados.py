import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

attack_colors = {}

# Lectura del dataset
df = pd.read_csv("dataset_combinado.csv", low_memory=False)

# Comprobacion de que los tipos se han inferido correctamente
reference_dtypes = df.dtypes.to_dict()
print("\nTipos de cada uno de los atributos:")
for col, dt in reference_dtypes.items():
    print(f" - {col}: {dt}")

# Numero de filas duplicadas
print(f"\nFilas duplicadas: {df.duplicated().sum()}\n")

# Numero de filas benignas vs filas maliciosas
num_benign = len(df[df["Label"] == 0])
num_malicious = len(df[df["Label"] == 1])
print(f"\nFilas benignas: {num_benign}")
print(f"Filas maliciosas: {num_malicious}\n")

# Realmente el puerto destino lo podemos mapear para cada servicio, de manera que el modelo no se tome el puerto como una variable ordinal
# Veamos cuantos valores diferentes tenemos para los puertos no efímeros
unique_ports = df[df["Dst Port"] < 49152]["Dst Port"].nunique()
print(f"Numero de puertos distintos < 49152: ", {unique_ports})

# Ahora vamos a indagar sobre los tipos de ataques que hay en el dataset combinado
# Creo un nuevo df con solo los ataques:
df_malicioso = df[df["Label"] == 1]
cantidadTiposAtaque = df_malicioso["Attack Name"].nunique()
print(f"\nCantidad de ataques diferentes: {cantidadTiposAtaque} ")
print("\nDistribución de 'Attack Name' para tráfico malicioso:")
print(df_malicioso["Attack Name"].value_counts(), "\n")

####################################
# Analisis de las filas duplicadas #
####################################

# En primer lugar buscamos todas las filas duplicadas (keep=False -> todas las ocurrencias).
duplicates_df = df[df.duplicated(keep=False)]
print(f"Total de filas duplicadas: {len(duplicates_df)}")

# Ver que tipos de ataque aparecen en estas filas
print("\nDistribución de 'Attack Name' en las filas duplicadas:")
print(duplicates_df["Attack Name"].value_counts())

# Elaboracion de graficas resumen (al igual que en los datasets individuales)
def process_dataset_bccc():

    # Primera grafica: distribución de los tipos de flujo (1 = malicioso, 0 = benigno)
    plt.figure(figsize=(10, 6))
    counts = df['Label'].value_counts()
    colors = ['green' if idx == 0 else 'red' for idx in counts.index]
    ax = counts.plot(kind='bar', color=colors)
    plt.title(f"Distribución de 'Label' en el Dataset CIC-BCCC-NRC-TabularIoT-2024")
    plt.xlabel("Tipo de conexión (1 = malicioso, 0 = benigno)")
    plt.ylabel("Frecuencia")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Segunda grafica: distribucion de los tipos de ataque
    df_ataque = df[df['Label'] != 0]
    counts = df_ataque['Attack Name'].value_counts()

    # Número de categorías de ataque
    num_categories = len(counts.index)

    # Creamos la paleta de colores. Usamos la paleta 'tab20' con num_categories colores distintos
    cmap = plt.get_cmap('tab20', num_categories)
    color_list = [cmap(i) for i in range(num_categories)]

    plt.figure(figsize=(10, 6))
    ax = counts.plot(kind='bar', color=color_list)

    # Título y ejes
    plt.title("Distribución de 'Attack Name' en conexiones maliciosas (Label=1)")
    plt.xlabel("Tipo de Ataque")
    plt.ylabel("Frecuencia")

    # Formateamos el eje Y con separadores de miles
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Ajustamos la rotación de las etiquetas del eje X (opcional, 90° si muchos ataques)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


process_dataset_bccc()