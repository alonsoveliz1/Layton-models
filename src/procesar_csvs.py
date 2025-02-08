import os
import pandas as pd
import gc

def parse_timestamp(timestamp_str):
    timestamp_formats = [
        '%d/%m/%Y %I:%M:%S %p',
        '%d/%m/%y %H:%M',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %I:%M:%S %p',
    ]

    for fmt in timestamp_formats:
        try:
            return pd.to_datetime(timestamp_str, format=fmt)
        except ValueError:
            continue

    return timestamp_str

def procesar_ficheros_csv(archivo_csv: str, target_path: str):
    # Mapeo de puertos a servicios, 32 servicios entre los mas reconocidos (luego habra que tener en cuenta Otros y los Ephemeral)
    port_service_map = {
        20: "FTP-Data", 21: "FTP-Control", 22: "SSH", 23: "Telnet", 25: "SMTP",
        53: "DNS", 67: "DHCP-Server", 68: "DHCP-Client", 69: "TFTP", 80: "HTTP",
        110: "POP3", 123: "NTP", 135: "RPC", 137: "NetBIOS-NS", 138: "NetBIOS-DGM",
        139: "NetBIOS-SSN", 143: "IMAP", 161: "SNMP", 443: "HTTPS", 445: "SMB",
        465: "SMTPS", 587: "SMTP-Submit", 993: "IMAPS", 995: "POP3S",
        1433: "MSSQL", 1521: "Oracle DB", 3306: "MySQL", 3389: "RDP",
        5432: "PostgreSQL", 5900: "VNC", 8080: "HTTP-Alt", 8888: "HTTP-Alt"
    }

    def map_service(port):
        if port in port_service_map:
            return port_service_map[port]
        elif 49152 <= port <= 65535:
            return "Ephemeral"
        else:
            return "Others"

    # Columnas a procesar
    columns_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Device']
    float_columns = ['Idle Std', 'Idle Max', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
                    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Bwd Packet Length Max',
                    'Bwd Packet Length Min', 'Packet Length Min', 'Packet Length Max', 'Down/Up Ratio',
                    'Idle Mean', 'Idle Min']

    # Leer el archivo CSV completo
    df = pd.read_csv(archivo_csv, low_memory=False)

    # Manejo especial para archivos de Port Scanning
    if "Port Scanning.csv" in archivo_csv:
        bwd_iat_columns = ['Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
                          'Bwd IAT Max', 'Bwd IAT Min']
        for col in bwd_iat_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)

    # Procesar timestamp si existe
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)

    # Añadir columna de Service
    df['Service'] = df['Dst Port'].apply(map_service)

    # Eliminar columnas especificadas
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Convertir columnas a float
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Guardar el resultado
    df.to_csv(target_path, index=False)

    gc.collect()

def process_csv_files(base_folder: str, target_folder: str):
    for subdir, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".csv"):
                source_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(source_path, base_folder)
                target_path = os.path.join(target_folder, relative_path)

                target_subfolder = os.path.dirname(target_path)
                os.makedirs(target_subfolder, exist_ok=True)

                print(f"Processing {source_path}")

                try:
                    procesar_ficheros_csv(source_path, target_path)
                    print(f"Saved: {target_path}\n")

                except Exception as e:
                    print(f"Error processing {source_path}: {e}")

def main():
    base_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024"
    target_folder = "C:\\Users\\avelg\\PycharmProjects\\NIDS\\CIC-BCCC-NRC-TabularIoT-2024-MOD"
    process_csv_files(base_folder, target_folder)


if __name__ == "__main__":
    main()