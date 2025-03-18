import datetime
import os
import pandas as pd

def parsear_fecha_flujo(timestamp_str) -> datetime:

    # Primero, eliminamos los espacios adicionales
    timestamp_str = timestamp_str.strip()

    timestamp_formats = [
        '%d/%m/%Y %I:%M:%S %p',
        '%d/%m/%y %H:%M',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %I:%M:%S %p',
        '%Y-%m-%d %H:%M:%S'
    ]

    for formato in timestamp_formats:
        try:
            return pd.to_datetime(timestamp_str, format=formato)
        except ValueError as e:
            # print(f"Error con formato {formato}: {e}")
            continue

    # Como último recurso si no se puede parsear con los formatos que le paso, que lo intente parsear automaticamente
    try:
        print("Intentando parseo automático")
        return pd.to_datetime(timestamp_str)
    except Exception as e:
        print(f"Todos los formatos fallaron: {e}")
        return timestamp_str



def procesar_ficheros_csv(archivo_csv: str, target_path: str) -> None:
    """Procesamiento del dataset incluye el mapeo de puertos en los servicios utilizados, eliminacion de columnas innecesarias
        reconversion de tipos de los atributos para que sean consistentes,,,"""
    # Mapeo de puertos a servicios (se irá iterando), 24 servicios al principio entre los mas reconocidos (luego habra que tener en cuenta Otros y los Ephemeral) Ahora menos
    port_service_map = {
        22: "SSH",
        80: "HTTP",
        81: "HTTP",
        8000: "HTTP",
        8081: "HTTP",
        1883: "MQTT",
        443: "HTTPS",
    }

    # Mapeo de cada clase de ataque
    dos_map = {'DoS TCP Flood', 'DoS SYN Flood', 'SYN Flood', 'DoS DNS Flood'}
    ddos_map = {'DDoS RSTFIN Flood', 'DDoS TCP SYN Flood', 'ACK Flood', 'DDoS ACK Fragmentation', 'DDoS PSHACK Flood', 'DDoS HTTP Flood'}
    brute_force_map = {'MQTT Brute Force', 'Sparta SSH Brute Force','Password Attack', 'Dictionary Brute Force','Telnet Brute Force'}
    scanning_map = {'Recon Port Scan','Recon OS Scan','Recon Vulnerability Scan','Recon Ping Sweep','Scan Aggressive','Port Scanning','Scan Port OS','Scan Host Port','Vulnerability Scanner','Recon Host Discovery','OS Fingerprinting'}
    application_map = {'XSS','SQL Injection','Uploading Attack'}
    mitm_map = {'MITM','MITM ARP Spoofing'}
    mirai_map = {'Mirai ACK Flood','Mirai HTTP Flood','Mirai Host Brute Force'}
    mqtt_map = {'MQTT DDoS Publish Flood','MQTT DoS Connect Flood','MQTT Malformed','MQTT DoS Publish Flood'}
    other_map = {'Backdoor','Ransomware'}
    benign_map = {'Benign Traffic'}

    def classify_attack(attack_name):
        if attack_name in dos_map:
            return 'DoS'
        elif attack_name in ddos_map:
            return 'DDoS'
        elif attack_name in brute_force_map:
            return 'Brute Force'
        elif attack_name in scanning_map:
            return 'Scanning'
        elif attack_name in application_map:
            return 'Application'
        elif attack_name in mitm_map:
            return 'Mitm'
        elif attack_name in mirai_map:
            return 'Mirai'
        elif attack_name in mqtt_map:
            return 'MQTT'
        elif attack_name in other_map:
            return 'Other'
        elif attack_name in benign_map:
            return 'Benign'
        else:
            return 'Unknown'



    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Device']

    float_columns = ['Idle Std', 'Idle Max', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
                     'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Bwd Packet Length Max',
                     'Bwd Packet Length Min', 'Packet Length Min', 'Packet Length Max', 'Down/Up Ratio',
                     'Idle Mean', 'Idle Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                     'Bwd IAT Min']

    def map_service(port):
        if port in port_service_map:
            return port_service_map[port]
        elif 8000 <= port <= 9000:
            return "Puerto tipico IoT"
        elif 49152 <= port <= 65535:
            return "Puertos Efimeros/Dinamicos"
        else:
            return "Otros"

    # Leer el archivo CSV completo
    df = pd.read_csv(archivo_csv, low_memory=False)

    # Re-conversion de tipos a float (pandas los convierte a int de manera inconsistente)
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Procesar la columna timestamp si existe (spoiler: existe siempre)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].apply(parsear_fecha_flujo)

    # Añadir columna de Service
    df['Service'] = df['Dst Port'].apply(map_service)

    # Añadir columna de Categoria de ataque
    df['Attack Category'] = df['Attack Name'].apply(classify_attack)

    # Eliminar columnas especificadas (no aportan al modelo o contienen informacion que se ha explicado en nuevas columnas)
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Eliminar las columnas negativas tambien
    df_numeric = df.select_dtypes(include='number')

    mask_negative = (df_numeric < 0).any(axis=1)

    df_sin_negativos = df[~mask_negative]
    # Guardar el resultado
    df_sin_negativos.to_csv(target_path, index=False)



def procesar_csvs(base_folder: str, target_folder: str) -> None:
    """Metodo lanzadera para procesar los archivos csv y que se guarden en su respectiva carpeta"""

    for subdir, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".csv"):

                source_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(source_path, base_folder)
                target_path = os.path.join(target_folder, relative_path)
                # print(f"Source path: {source_path}")
                # print(f"Target path: {relative_path}")
                # print(f"Target path: {target_path}")

                target_subfolder = os.path.dirname(target_path)
                os.makedirs(target_subfolder, exist_ok=True)
                # print(f"Target subfolder: {target_subfolder}")

                print(f"Intentando procesar {source_path}")
                try:
                    procesar_ficheros_csv(source_path, target_path)
                    print(f"Guardado: {target_path}\n")

                except Exception as e:
                    print(f"Error procesando {source_path}: {e}")



def main():
    base_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\raw\\CIC-BCCC-NRC-TabularIoT-2024"
    target_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    procesar_csvs(base_folder, target_folder)



if __name__ == "__main__":
    main()