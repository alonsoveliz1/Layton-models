import pandas as pd
from sklearn.utils import resample
from pathlib import Path

def generar_dataset_balanceado(df) -> None:
    df_malicious = df[df["Label"] == 1]
    df_benign = df[df["Label"] == 0]

    n_instances = 200000
    n_categories = df_malicious['Attack Category'].nunique()

    muestras_por_categoria = int(n_instances / n_categories)
    balanced_categories = []

    # Balance by attack type
    for attack_name, df_cat in df_malicious.groupby('Attack Category'):
        print(f"Processing {attack_name}")
        subattacks = df_cat['Attack Name'].unique()
        n_subattacks = len(subattacks)

        muestras_por_subataque = int(muestras_por_categoria/n_subattacks)
        balanced_subattacks = []

        print(f"Muestras por subataque {muestras_por_subataque}")
        for subattack_name, df_sub in df_cat.groupby('Attack Name'):
            print(f"Processing {subattack_name}")
            replace_needed = (len(df_sub) < muestras_por_subataque)
            print(f"Replace needed {replace_needed}")
            df_sub_bal = resample(
                df_sub,
                replace = replace_needed,
                n_samples = muestras_por_subataque,
                random_state = 42,
            )

            balanced_subattacks.append(df_sub_bal)

        df_cat_bal = pd.concat(balanced_subattacks, ignore_index=True)
        balanced_categories.append(df_cat_bal)

    balanced_malicious = pd.concat(balanced_categories, ignore_index=True)

    # We also have to create a balanced subset of the benign traffic, since most of it is from the MQTT service so bias ain't introduced in the model
    balanced_services = []
    n_services = df_benign['Service'].nunique()
    n_instances_benign = n_instances * n_categories

    for service_type, df_serv in df_benign.groupby('Service'):
        replace_needed = (len(df_serv) < n_instances_benign)
        df_benign_bal = resample(
            df_serv,
            replace = replace_needed,
            n_samples = int(n_instances_benign/n_services),
            random_state = 42
        )
        balanced_services.append(df_benign_bal)

    balanced_benign = pd.concat(balanced_services, ignore_index=True)

    balanced_total = pd.concat([balanced_benign,balanced_malicious], ignore_index=True)
    base_path = Path(__file__).resolve().parent.parent
    final_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" /  "combinado_balanceado.csv"

    # Shuffle
    balanced_total = balanced_total.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_total.to_csv(final_path, index = False)

    print(balanced_benign['Service'].value_counts())
    print(balanced_total['Service'].value_counts())

def main():
    base_path = Path(__file__).resolve().parent.parent
    combinado_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" / "combinado.csv"
    df = pd.read_csv(combinado_path , low_memory=False)
    generar_dataset_balanceado(df)

main()