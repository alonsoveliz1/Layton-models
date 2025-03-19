import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

def main():
    df = pd.read_csv("../data/processed/CIC-BCCC-NRC-TabularIoT-2024-MOD/combinado_balanceado.csv")
    # Attributes that are not necessary for binary classifier
    df = df.drop(columns=['Attack Name','Attack Category'])

    # Drop Label since it'll be our predicting value and timestamp
    X = df.drop(columns=['Label','Timestamp'])
    Y = df['Label']

    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)
    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size = 1/3, random_state = 42, stratify = y_temp)

    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Eval set: {X_eval.shape}, {y_eval.shape}\n")

    print("Train Label Distribution:\n", y_train.value_counts(normalize=True))
    print("\nTest Label Distribution:\n", y_test.value_counts(normalize=True))
    print("\nEval Label Distribution:\n", y_eval.value_counts(normalize=True))

    numerical_features = X_train.select_dtypes(include=np.number).columns
    categorical_features = X_train.select_dtypes(exclude=np.number).columns

    # One hot encoding of categorical attributes
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[categorical_features])

    X_train_ohe = ohe.transform(X_train[categorical_features])
    X_test_ohe = ohe.transform(X_test[categorical_features])
    X_eval_ohe = ohe.transform(X_eval[categorical_features])

    ohe_cols = ohe.get_feature_names_out(categorical_features)
    X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe_cols, index=X_train.index)
    X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe_cols, index=X_test.index)
    X_eval_ohe = pd.DataFrame(X_eval_ohe, columns=ohe_cols, index=X_eval.index)

    X_train_final = pd.concat([X_train.drop(columns=categorical_features), X_train_ohe], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=categorical_features), X_test_ohe], axis=1)
    X_eval_final = pd.concat([X_eval.drop(columns=categorical_features), X_eval_ohe], axis=1)

    numerical_features = X_train_final.select_dtypes(include=np.number).columns
    categorical_features = X_train_final.select_dtypes(exclude=np.number).columns

    print(f"{numerical_features}\n")
    print(f"{numerical_features.size}\n")
    print(f"{categorical_features}\n")

    """ CORRELATION BETWEEN ATTRIBUTES"""
    # PLOT
    corr_matrix = X_train_final.corr()
    sns.clustermap(corr_matrix, cmap='coolwarm', linewidths =0.5,  figsize=(36, 36))
    plt.show()

    # PRINT > 0.7 THRESHOLD
    threshold = 0.7  # or whatever is interesting
    high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    print("Highly correlated pairs (>0.7 or < -0.7):")
    for pair in high_corr_pairs:
        print(pair, corr_matrix.loc[pair[0], pair[1]])


    """FEATURE IMPORTANCE"""
    X_train_scaled = robust_scaler_with_epsilon(X_train_final, numerical_features)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Extract importances
    importances = rf_model.feature_importances_
    features = X_train_scaled.columns

    # Sort them from highest to lowest
    sorted_idx = np.argsort(importances)[::-1]

    print("Feature Importances (highest to lowest):")
    for idx in sorted_idx:
        print(f"{features[idx]}: {importances[idx]}")

    top_n = 20
    top_features = [features[i] for i in sorted_idx[:top_n]]
    top_importances = [importances[i] for i in sorted_idx[:top_n]]

    # Optional: quick bar plot
    plt.figure(figsize=(20, 12))
    plt.title("Feature Importances (RandomForest)")
    plt.bar([top_features[i] for i in range(len(top_features))], [top_importances[i] for i in range(len(top_importances))])
    plt.xticks(rotation=90)
    plt.show()

def robust_scaler_with_epsilon(df, columns):
    df_scaled = df.copy()
    for col in columns:
        median_val = df[col].median()
        q1, q3 = np.percentile(df[col], [25, 75])
        iqr_val = q3 -q1
        if iqr_val < 1e-8:
            iqr_val += 1e-8
        df_scaled[col] = (df[col] - median_val)/iqr_val
        return df_scaled

main()