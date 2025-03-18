import pandas as pd
from sklearn.utils import resample

def generar_dataset_balanceado(df) -> None:
    df_malicious = df[df["Label"] == 1]
    df_benign = df[df["Label"] == 0]

    n_instances = 100000
    n_categories = df_malicious['Attack Category'].nunique()

    muestras_por_categoria = int(n_instances / n_categories)
    balanced_categories = []

    for attack_name, df_cat in df_malicious.groupby('Attack Category'):
        subattacks = df_cat['Attack Name'].unique()
        n_subattacks = len(subattacks)

        muestras_por_subataque = int(muestras_por_categoria/n_subattacks)
        balanced_subattacks = []

        for subattack_name, df_sub in df_cat.groupby('Attack Name'):

            replace_needed = (len(df_sub) < muestras_por_subataque)
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

    df_benign_bal = resample(
        df_benign,
        replace = False,
        n_samples = n_instances * n_categories,
        random_state = 42
    )

    balanced_total = pd.concat([df_benign_bal,balanced_malicious], ignore_index=True)
    balanced_total.to_csv("C:\\Users\\avelg\\PycharmProjects\\NIDS\\data\\processed\\CIC-BCCC-NRC-TabularIoT-2024-MOD\\combinado_balanceado.csv", index = False)

def main():
    df = pd.read_csv("../data/processed/CIC-BCCC-NRC-TabularIoT-2024-MOD/combinado.csv", low_memory=False)
    generar_dataset_balanceado(df)

main()