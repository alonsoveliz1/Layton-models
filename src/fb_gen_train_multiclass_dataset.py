import pandas as pd
from sklearn.utils import resample
from pathlib import Path

def generar_dataset_balanceado(df) -> None:
    df_malicious = df[df["Label"] == 1]

    n_instances = 200000
    n_categories = df_malicious['Attack Category'].nunique()

    muestras_por_categoria = int(n_instances / n_categories)
    balanced_categories = []

    # Balancear por tipo de ataque
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
            if replace_needed:
                print(f"Replace needed {replace_needed}")
                print(f"Muestras repetidas {muestras_por_subataque - len(df_sub)}")

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


    base_path = Path(__file__).resolve().parent.parent
    final_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" /  "combinado_balanceado_ataques.csv"

    # Mezclar
    balanced_total = balanced_malicious.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_total.to_csv(final_path, index = False)

    print(balanced_malicious['Service'].value_counts())

def main():
    base_path = Path(__file__).resolve().parent.parent
    combinado_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" / "combinado.csv"
    df = pd.read_csv(combinado_path , low_memory=False)
    generar_dataset_balanceado(df)

main()