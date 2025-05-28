import time
from pathlib import Path
from datetime import datetime

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from onnxconverter_common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from mlxtend.evaluate import paired_ttest_5x2cv

def main():
    pd.set_option('display.max_columns', None)
    base_path = Path(__file__).parent.parent
    output_dir = base_path / 'output'
    dataset_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" / "combinado_balanceado.csv"
    df = pd.read_csv(dataset_path)

    # TRAIN TEST EVAL SPLIT
    x_train, x_test, x_eval, y_train, y_test, y_eval, attack_names_test = train_test_eval_split(df)

    # PREPROCESSING
    x_train_ohe, x_test_ohe, x_eval_ohe = preprocess_df(x_train, x_test, x_eval)

    # # FEATURE SELECTION - Extract correlation info
    corr_threshold = 0.9
    # extract_correlation_info(x_train_ohe, corr_threshold, base_path, "full_correlation_matrix.png", "full_correlation_info.txt") # Corr Matrix && List

    # DIMENSIONALITY REDUCTION (> Threshold but I only drop the attributes less related to the label)
    x_train_reduced, dropped_features = remove_correlated_features(x_train_ohe, y_train, corr_threshold)
    x_test_reduced = x_test_ohe.drop(columns = dropped_features)
    x_eval_reduced = x_eval_ohe.drop(columns= dropped_features)

    # # PLOT AGAIN CORRELATION MATRIX WITH REDUCED ATTRIBUTES
    # extract_correlation_info(x_train_reduced, corr_threshold, base_path, "reduced_correlation_matrix.png", "reduced_correlation_info.txt")

    # # FEATURE MAP FOR ONNX MODEL CONVERSION, FORMAT f{i}
    # original_columns = x_train_reduced.columns
    # feature_map_path =  output_dir / 'feature_map_reduced.txt'
    # with open(feature_map_path, 'w') as f:
    #     for i, name in enumerate(original_columns):
    #         f.write(f'{i} {name}\n')
    # f.close()

    # RENAMING ATTRIBUTES TO A F+{i} MAP SO THEN XGB MODEL CAN BE EXPORTED TO ONNX RUNTIME
    x_train_reduced.columns = [f'f{i}' for i in range(x_train_reduced.shape[1])]
    x_test_reduced.columns = [f'f{i}' for i in range(x_test_reduced.shape[1])]
    x_eval_reduced.columns = [f'f{i}' for i in range(x_eval_reduced.shape[1])]


    # # PAIRED_T_TEST_5x2CV FOR MODEL SELECTION
    # print("Starting paired T_TEST_5x2CV...\n")
    # models = {
    #     "LogReg": Pipeline(steps=[("scaler", StandardScaler()),
    #                               ("clf", LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000))]),
    #     "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
    #     "XGBoost": XGBClassifier(random_state=42),
    #     "LightGBM": LGBMClassifier(device = "cuda", random_state=42, verbose = -1, deterministic = True),
    # }
    #
    # apply_paired_ttest(models, x_eval_reduced, y_eval)

    results = {}

    # Convert to DMatrix for GPU processing
    d_train = xgb.DMatrix(x_train_reduced, label=y_train)
    d_test = xgb.DMatrix(x_test_reduced, label=y_test)
    d_eval = xgb.DMatrix(x_eval_reduced, label=y_eval)

    # Convert to CuPy array for CUDA parallel computing processing
    x_train_cuda = cp.asarray(x_train_reduced.values)
    x_test_gpu = cp.asarray(x_test_reduced.values)
    x_eval_cuda = cp.asarray(x_eval_reduced.values)

    # XGBoost model
    try:
        print("\nTRAINING XGBOOST\n")
        start_time = time.time()

        # Hyperparameter tuning with optuna API
        def objective(trial):
            params = {
               'objective': 'binary:logistic',
               'eval_metric': 'auc', # logloss
               'device': 'cuda',
               'max_depth': trial.suggest_int('max_depth', 3, 15),                  # Max tree depth, model complexity
               'eta': trial.suggest_float('eta', 0.01, 0.3, log = True),            # Learning rate, lower better generalz.
               'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),  # Minimun sum of weights in a child node
               'subsample': trial.suggest_float('subsample', 0.6, 1.0),             # Fraction training to sample per tree
               'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5, 15),  # Balance classes (9-1 in my madeup dataset)
               'gamma': trial.suggest_float('gamma', 0, 5),                         # Minimum loss reduction to make a partition leaf
               'alpha': trial.suggest_float('alpha', 0, 5),                         # Regularization on leaf weights
               'lambda': trial.suggest_float('lambda', 0, 5)                        # Regularization on leaf weights
            }

            prunning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
            model = xgb.train(
                        params,
                        d_train,
                        num_boost_round=1000,
                        evals = [(d_eval, 'validation')],
                        callbacks=[prunning_callback],
                        early_stopping_rounds=50,
                        verbose_eval=False
            )

            return model.best_score # Eval metric was auc (tradeoff between TPR and FPR) Imbalanced data (logloss?)


        # Create optuna study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

        # Optimize
        study.optimize(objective, n_trials=50, timeout=3600)  # 50 intentos o 1 hora

        # Get best params of the study
        best_params = study.best_params

        # TRAIN MODEL WITH BEST PARAMS
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'device': 'cuda',
            **best_params
        }

        model = xgb.train(
            final_params,
            d_train,
            num_boost_round=1000,
            evals=[(d_train, 'train'), (d_eval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        preds_proba = model.inplace_predict(x_train_cuda)

        # Find best threshold to minimize __objective__
        best_score = float('inf')
        best_threshold = 0.5

        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (preds_proba > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_train, preds.get()).ravel()

            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # We want to attack the fp_rate (way more costly than tn_rate)
            score = fp_rate

            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_fp_rate = fp_rate
                best_fn_rate = fn_rate


        # Make predictions on test set with best_threshold to maximize score
        y_pred_proba = model.inplace_predict(x_test_gpu)
        y_pred = (y_pred_proba > best_threshold).astype(int).get() # If > 0,5 TRUE

        # Calculate metrics, classificatioin report
        acc = accuracy_score(y_test, y_pred)

        print(f"XGBoost training completed in {time.time() - start_time:.2f} seconds")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results['xgboost'] = {
            'accuracy': acc,
            'model': model,
            'training_time': time.time() - start_time
        }

        # Get feature importance
        if hasattr(model, 'get_score'):
            print("\nTop 10 Feature Importance:")
            importance = model.get_score(importance_type='gain')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_importance[:10]:
                print(f"{feature}: {score}")

        # Store predictions with the original data
        evaluation_results = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'probability': y_pred_proba.get(),
            "attack_name": attack_names_test
        })

        # Check which are the attacks that are not classified properly
        evaluation_results['attack_name'] = attack_names_test

        # Find all false negatives -> missed attacks
        missed_attacks = evaluation_results[(evaluation_results['true_label'] == 1) &
                                            (evaluation_results['predicted_label'] == 0)]

        # Count missed attacks by type
        missed_by_type = missed_attacks['attack_name'].value_counts()
        total_by_type = evaluation_results[evaluation_results['true_label'] == 1]['attack_name'].value_counts()

        all_attack_types = set(missed_by_type.index) | set(total_by_type.index)
        miss_rates = {}

        for attack in all_attack_types:
            missed = missed_by_type.get(attack, 0)
            total = total_by_type.get(attack, 0)

            # Avoid division by zero
            if total > 0:
                miss_rates[attack] = (missed / total) * 100
            else:
                miss_rates[attack] = 0.0
                print(f"Warning: No instances of {attack} in total_by_type")

        miss_rate = pd.Series(miss_rates).sort_values(ascending=False)

        # Filter only attack types with miss rate > 0
        miss_rate = miss_rate[miss_rate > 0]

        print("Miss Rates by Attack Type (excluding 0%):")
        for attack, rate in miss_rate.items():
            if attack in total_by_type:
                print(f"{attack}: {rate:.2f}% missed ({missed_by_type.get(attack, 0)} out of {total_by_type[attack]})")
            else:
                print(f"{attack}: No instances in evaluation set")

        # Plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if rate > 20 else 'orange' if rate > 10 else 'green' for rate in miss_rate.values]
        bars = plt.bar(range(len(miss_rate)), miss_rate.values, color=colors)

        # X-axis labels
        plt.xticks(range(len(miss_rate)), miss_rate.index, rotation=45, ha='right')
        plt.xlabel("Attack Type")
        plt.ylabel("Miss Rate (%)")
        plt.title("Miss Rates by Attack Type (excluding 0%)")
        plt.tight_layout()

        # Annotate bars with text
        for i, (attack, rate) in enumerate(miss_rate.items()):
            count = missed_by_type.get(attack, 0)
            total = total_by_type.get(attack, 0)
            plt.text(i, rate + 1, f'{rate:.1f}%\n({count}/{total})',
                     ha='center', va='bottom', fontsize=9)

        plt.show()

        # ROC Curve from test
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba.get())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=( 8 , 6 ))
        plt.plot(fpr, tpr, label =f' ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        plt .close()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.title(f'Confusion matrix')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels =['Benigno (0)', 'Ataque (1)'], yticklabels=['Benigno (0)', 'Ataque (1)'])
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.savefig(base_path / "output" / "confusion_matrix" / f"confusion_matrix{objective}.png")

        # 4. Heatmap de Rendimiento por Tipo de Ataque
        # Calcular métricas para cada tipo de ataque
        attack_types = evaluation_results['attack_name'].unique()
        attack_metrics = {}
        for attack in attack_types:
            attack_data = evaluation_results[evaluation_results['attack_name'] == attack]
            if len(attack_data[attack_data['true_label'] == 1]) > 0:
                tn = len(attack_data[(attack_data['true_label'] == 0) & (attack_data['predicted_label'] == 0)])
                fp = len(attack_data[(attack_data['true_label'] == 0) & (attack_data['predicted_label'] == 1)])
                fn = len(attack_data[(attack_data['true_label'] == 1) & (attack_data['predicted_label'] == 0)])
                tp = len(attack_data[(attack_data['true_label'] == 1) & (attack_data['predicted_label'] == 1)])

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                attack_metrics[attack] = {
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Samples': len(attack_data[attack_data['true_label'] == 1])
                }

        # Crear DataFrame y heatmap
        metrics_df = pd.DataFrame(attack_metrics).T
        metrics_df = metrics_df.sort_values('Recall')  # Ordenar por recall

        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics_df[['Precision', 'Recall', 'F1-Score']],
                    annot=True, fmt='.3f', cmap='RdYlGn',
                    cbar_kws={'label': 'Valor de la Métrica'},
                    vmin=0, vmax=1)
        plt.title('Métricas de Rendimiento por Tipo de Ataque')
        plt.xlabel('Métrica')
        plt.ylabel('Tipo de Ataque')
        plt.tight_layout()
        plt.savefig(base_path / "output" / f"attack_metrics_heatmap.png", dpi=300)
        plt.show()
        try:
            print(f"X_Train_reduced.shape {x_train_reduced.shape}")
            num_features = x_train_reduced.shape[1]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            onnx_path = base_path / "output" / "models" / f"xgboost_flow_classifier_l1{timestamp}.onnx"
            onnx_model = onnxmltools.convert_xgboost(model, initial_types=[('input', FloatTensorType([None, num_features]))])
            with open(onnx_path,"wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved as {onnx_path}")
        except Exception as e:
            print(f"Error with ONNX Conversion: {e}")

        # Free memory
        del model, d_train, d_test, d_eval

    except Exception as e:
        print(f"Error with XGBoost: {e}")


def train_test_eval_split(df):

    print(f"Combined_balanced shape: {df.shape}") # Print the dataset rows && nª attrbs
    print(f"Combined_balanced attributes: {df.columns}") # Print the dataset attributes

    # (1) PREPARE DATA - Attributes that are not necessary for binary classifier (72 left)
    df = df.drop(columns=["Attack Category","Timestamp"])

    # (2) Drop Label since it'll be our predicting value
    X = df.drop(columns=["Label"])
    Y = df["Label"]

    # (3) Train-test-eval (70-20-10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = df["Attack Name"])

    # (3.1) Auxiliary df to be able to do a second stratification by Attack Name
    df_temp = df.loc[X_temp.index].copy()
    # print(f"DF_temp shape: {df_temp.shape}")
    # print(df_temp.columns)

    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size = 1/3, random_state = 42, stratify = df_temp['Attack Name'])

    attack_names_test = X_test["Attack Name"].copy() # auxiliary for later
    # print(eval_attack_names)

    X_train = X_train.drop(columns=['Attack Name'])
    X_test = X_test.drop(columns=['Attack Name'])
    X_eval = X_eval.drop(columns=['Attack Name'])

    # print(f"Train set: {X_train.shape}, {y_train.shape}")
    # print(f"Test set: {X_test.shape}, {y_test.shape}")
    # print(f"Eval set: {X_eval.shape}, {y_eval.shape}\n")

    # print("Train Label Distribution:\n", y_train.value_counts(normalize=True))
    # print("\nTest Label Distribution:\n", y_test.value_counts(normalize=True))
    # print("\nEval Label Distribution:\n", y_eval.value_counts(normalize=True))
    # print("\n")

    return X_train, X_test ,X_eval , y_train, y_test, y_eval, attack_names_test

def preprocess_df(X_train, X_test, X_eval):

    # PREPROCESSING - One hot encoding of categorical attributes (Service)
    categorical_features = X_train.select_dtypes(exclude=np.number).columns
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[categorical_features])

    X_train_ohe = ohe.transform(X_train[categorical_features])
    X_test_ohe = ohe.transform(X_test[categorical_features])
    X_eval_ohe = ohe.transform(X_eval[categorical_features])

    ohe_cols = ohe.get_feature_names_out(categorical_features)
    X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe_cols, index=X_train.index)
    X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe_cols, index=X_test.index)
    X_eval_ohe = pd.DataFrame(X_eval_ohe, columns=ohe_cols, index=X_eval.index)

    # Dropping Service from the train/test/eval and concatenating with the OHE new features
    X_train_final = pd.concat([X_train.drop(columns=categorical_features), X_train_ohe], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=categorical_features), X_test_ohe], axis=1)
    X_eval_final = pd.concat([X_eval.drop(columns=categorical_features), X_eval_ohe], axis=1)

    return X_train_final, X_test_final, X_eval_final

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


def get_robust_scaler(df, columns):
    stats = {}
    for col in columns:
        median_val = df[col].median()
        q1, q3 = np.percentile(df[col], [25, 75])
        iqr_val = q3 -q1
        if iqr_val < 1e-8:
            iqr_val += 1e-8
        stats[col] = {"median" : median_val, "iqr" : iqr_val}
    return stats


def transform_robust_scaler_with_epsilon(df, stats):
    df_scaled = df.copy()
    for col, vals in stats.items():
        df_scaled[col] = (df[col] - vals["median"]) / vals["iqr"]
    return df_scaled

def remove_correlated_features(X, y, threshold: float):
    corr_matrix = X.corr().abs()

    label_correlations = {}
    for feature in X.columns:
        corr_with_label = np.abs(pd.concat([X[feature],y], axis = 1).corr().iloc[0,-1])
        label_correlations[feature] = corr_with_label

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    to_drop = set()

    for column in upper.columns:
        if column in to_drop:
            continue

        correlated_features = [
            row for row in upper.index
            if upper.loc[row, column] > threshold and row != column
        ]

        if not correlated_features:
            continue

        correlated_features.append(column)
        best_feature = max(correlated_features, key=lambda x: label_correlations[x])

        # Add all but the best feature to the drop list
        to_drop.update([f for f in correlated_features if f != best_feature])

    print(f"Removing {len(to_drop)} highly correlated features...")
    print(f"Features dropped: {sorted(to_drop)}")

    # Return the dataframe with reduced features
    return X.drop(columns=list(to_drop)), list(to_drop)



def extract_correlation_info(df, threshold, project_path, plotname, filename):
    # Plot full correlation matrix
    corr_matrix = df.corr()
    sns.clustermap(corr_matrix, cmap='coolwarm', linewidths =0.5,  figsize=(36, 36))
    corr_matrix_path = project_path / "output" / "correlation_plots" / plotname
    plt.savefig(corr_matrix_path)
    plt.close()

    # Print pairs which are grather than > 0.7 THRESHOLD
    high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    resume_file = project_path / "output" / filename
    with open(resume_file, "w") as f:
        f.write("Highly correlated pairs (>0.95 or < -0.95):\n")
        for pair in high_corr_pairs:
            f.write(f"{pair}, {corr_matrix.loc[pair[0], pair[1]]}\n")
    f.close()

    print(f"Correlation matrix saved to {corr_matrix_path}")
    print(f"Correlation report saved to {resume_file}")



def apply_paired_ttest(models, x_train_reduced, y_train):
    results_ttest = []
    names = list(models.keys())
    n_models = len(names)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            print(f"Comparing {names[i]} vs {names[j]}...")
            name_a, estimator_a = names[i], models[names[i]]
            name_b, estimator_b = names[j], models[names[j]]

            t,p = paired_ttest_5x2cv(
                estimator1 = estimator_a,
                estimator2 = estimator_b,
                X = x_train_reduced,
                y = y_train,
                scoring = "f1_weighted",
                random_seed=42)
            results_ttest.append((name_a, name_b, t, p))

    for a, b, t, p in sorted(results_ttest, key=lambda x: x[3]):
        print(f"{a:15s} vs {b:15s}   t = {t:+.3f}   p = {p:.4g}")


main()

