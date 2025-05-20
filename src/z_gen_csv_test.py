from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


def generar_dataset_balanceado(df) -> None:
    df_malicious = df[df["Label"] == 1]
    df_benign = df[df["Label"] == 0]

    n_instances = 100
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

    balanced_test_total = pd.concat([balanced_benign,balanced_malicious], ignore_index=True)

    service_column = balanced_test_total[['Service']]
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(service_column)

    service_encoded = ohe.transform(service_column)
    ohe_cols = ohe.get_feature_names_out(['Service'])
    service_ohe = pd.DataFrame(service_encoded, columns=ohe_cols)
    balanced_test_final = pd.concat([balanced_test_total.drop(columns='Service'), service_ohe], axis=1)

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / 'data' / 'processed'
    output_file = output_dir / 'balanced_test_total.csv'
    balanced_test_final.to_csv(output_file, index = False)

def main():
    df = pd.read_csv("../data/processed/CIC-BCCC-NRC-TabularIoT-2024-MOD/combinado.csv", low_memory=False)
    generar_dataset_balanceado(df)

main()