import time

import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from onnxconverter_common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("../data/processed/CIC-BCCC-NRC-TabularIoT-2024-MOD/combinado_balanceado.csv")

    # TRAIN TEST EVAL SPLIT
    X_train, X_test, X_eval, y_train, y_test, y_eval, eval_attack_names = train_test_eval_split(df)

    # PREPROCESSING
    X_train_final, X_test_final, X_eval_final = preprocess_df(X_train, X_test, X_eval)

    # FEATURE SELECTION - Extract correlation info
    corr_threshold = 0.7
    extract_correlation_info(X_train_final, corr_threshold) # Corr Matrix && List

    # DIMENSIONALITY REDUCTION (> Threshold but I only drop the attributes less related to the label)
    X_train_reduced, dropped_features = remove_correlated_features(X_train_final, y_train, corr_threshold)
    X_test_reduced = X_test_final.drop(columns = dropped_features)
    X_eval_reduced = X_eval_final.drop(columns= dropped_features)

    original_columns = X_train_reduced.columns.tolist()
    feature_map_path = 'feature_map.txt'
    with open(feature_map_path, 'w') as f:
        for i, name in enumerate(original_columns):
            f.write(f'{i} {name}\n')


    # Plot again the full correlated matrix with less attributes (41 less)
    extract_correlation_info(X_train_reduced, corr_threshold)

    """
    # FEATURE SCALING -> NO NEEDED SINCE USING A TREE BASED MODEL HAS IN THIS CASE A PERFORMANCE DROP (ONE HOT ENCODED ATTRS)
    numerical_features = X_train_reduced.select_dtypes(include=np.number).columns
    scaler_stats = get_robust_scaler(X_train_reduced, numerical_features)
    X_train_scaled = transform_robust_scaler_with_epsilon(X_train_reduced, scaler_stats)
    X_test_scaled = transform_robust_scaler_with_epsilon(X_test_reduced, scaler_stats)
    X_eval_scaled = transform_robust_scaler_with_epsilon(X_eval_reduced, scaler_stats)
    """

    # Extract importances (RF)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_reduced, y_train)
    importances = rf_model.feature_importances_
    features = X_train_reduced.columns

    sorted_idx = np.argsort(importances)[::-1]
    print("Feature Importances (highest to lowest):")
    for idx in sorted_idx:
        print(f"{features[idx]}: {importances[idx]}")

    top_n = 20
    top_features = [features[i] for i in sorted_idx[:top_n]]
    top_importances = [importances[i] for i in sorted_idx[:top_n]]

    plt.figure(figsize=(20, 12))
    plt.title("Feature Importances (RandomForest)")
    plt.bar([top_features[i] for i in range(len(top_features))], [top_importances[i] for i in range(len(top_importances))])
    plt.xticks(rotation=45)
    plt.show()


    # RENAMING ATTRIBUTES TO A F+{i} MAP SO THEN XGB MODEL CAN BE EXPORTED TO ONNX RUNTIME
    X_train_reduced.columns = [f'f{i}' for i in range(X_train_reduced.shape[1])]
    X_test_reduced.columns = [f'f{i}' for i in range(X_test_reduced.shape[1])]
    X_eval_reduced.columns = [f'f{i}' for i in range(X_eval_reduced.shape[1])]

    # CROSS VALIDATION
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device':'cuda'
    }

    models = {
        "LogReg" : Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced"))]),
        "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1),
        "XGB": XGBClassifier(**base_params, n_jobs=-1)
    }

    cv_folds = 5
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        fold_metrics = []
        for train_index, val_index in skf.split(X_train_reduced, y_train):
            X_tr, X_val = X_train_reduced.iloc[train_index], X_train_reduced.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # train the correct model
            model.fit(X_tr, y_tr)

            # get probabilities (all three support predict_proba)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            fold_metrics.append({
                "accuracy": accuracy_score(y_val, y_pred),
                "auc": roc_auc_score(y_val, y_proba),
                "f1": f1_score(y_val, y_pred)
            })
        avg = pd.DataFrame(fold_metrics).mean().to_dict()
        cv_results[name] = avg
        del model, y_proba, y_pred

    print(cv_results)

    for name, res in cv_results.items():
        print(
            f"{name:6s} — "
            f"Accuracy={res['accuracy']:.4f}, "
            f"AUC={res['auc']:.4f}, "
            f"F1={res['f1']:.4f}"
        )

    results = {}



    # XGBoost model
    try:
        print("\nTRAINING XGBOOST\n")

        start_time = time.time()

        # Convert to DMatrix for faster processing
        dtrain = xgb.DMatrix(X_train_reduced, label=y_train)
        dtest = xgb.DMatrix(X_test_reduced, label=y_test)
        deval = xgb.DMatrix(X_eval_reduced, label=y_eval)

        # Hyperparameter tuning with optuna API
        def objective(trial):
            params = {
               'objective': 'binary:logistic',
               'eval_metric': 'auc', # logloss
               'device': 'cuda',
               'max_depth': trial.suggest_int('max_depth', 3, 15),
               'eta': trial.suggest_float('eta', 0.01, 0.3, log = True), # Learning rate
               'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
               'subsample': trial.suggest_float('subsample', 0.6, 1.0),
               'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5, 15),
               'gamma': trial.suggest_float('gamma', 0, 5),
               'alpha': trial.suggest_float('alpha', 0, 5),
               'lambda': trial.suggest_float('lambda', 0, 5)
            }

            prunning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
            model = xgb.train(params, dtrain, num_boost_round=1000, evals = [(deval, 'validation')], callbacks=[prunning_callback], early_stopping_rounds=50, verbose_eval=False)
            preds_proba = model.predict(deval)

            # Best prediction threshold
            precision, recall, thresholds = precision_recall_curve(y_eval, preds_proba)
            avg_precision = average_precision_score(y_eval, preds_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='best')
            plt.show()

            best_score = float('-inf')
            best_threshold = 0.5
            best_fp_rate = 1.0
            best_fn_rate = 1.0

            for threshold in np.arange(0.3, 0.7, 0.01):
                preds = (preds_proba > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()
                fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0


                # We want to attack the fp_rate but keeping a good recall

                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                # score = recall - (0.5 * fp_rate)
                score = f1_score

                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_fp_rate = fp_rate
                    best_fn_rate = fn_rate

            # Save best threshold and best fp_rate
            trial.set_user_attr("best_threshold", best_threshold)
            trial.set_user_attr("false_negative_rate", best_fn_rate)
            trial.set_user_attr("false_positive_rate", best_fp_rate)
            return best_score



        # Create optuna study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

        # Optimize
        study.optimize(objective, n_trials=50, timeout=3600)  # 50 intentos o 1 hora

        # Get best params of the study
        best_params = study.best_params
        best_threshold = study.best_trial.user_attrs["best_threshold"]
        best_fp_rate = study.best_trial.user_attrs["false_positive_rate"]

        print("Mejores parámetros encontrados:")
        print(best_params)
        print(f"Mejor umbral: {best_threshold}")
        print(f"Mejor tasa de falsos positivos: {best_fp_rate:.4f}")

        # TRAIN MODEL WITH BEST PARAMS
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'device': 'cuda',
            'tree_method': 'gpu_hist',
            **best_params
        }

        model = xgb.train(
            final_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (deval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # Make predictions
        y_pred_proba = model.predict(dtest) # The models give me the output (prediction)
        y_pred = (y_pred_proba > best_threshold).astype(int) # If > 0,5 TRUE

        # Calculate metrics
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
            'probability': y_pred_proba
        })

        # Check which are the attacks that are not classified properly
        evaluation_results['attack_name'] = eval_attack_names

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

        # Convert to Series for sorting
        miss_rate = pd.Series(miss_rates).sort_values(ascending=False)

        print("Miss rate by attack type:")
        for attack, rate in miss_rate.items():
            if attack in total_by_type:
                print(f"{attack}: {rate:.2f}% missed ({missed_by_type.get(attack, 0)} out of {total_by_type[attack]})")
            else:
                print(f"{attack}: No instances in evaluation set")


        # ROC Curve from test
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=( 8 , 6 ))
        plt.plot(fpr, tpr, label =f' ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.show()

        try:
            print(f"X_Train_reduced.shape {X_train_reduced.shape}")
            num_features = X_train_reduced.shape[1]
            feature_names = [f'f{i}' for i in range(num_features)] # Onnx wants a digit for each attr
            feature_map_path = 'feature_map.txt'
            with open(feature_map_path, 'w') as f:
                for i,name in enumerate(X_train_reduced.columns):
                    f.write(f'{i} {name} \n')
            f.close()

            onnx_path = "xgboost_flow_classifier_l1.onnx"
            onnx_model = onnxmltools.convert_xgboost(model, initial_types=[('input', FloatTensorType([None, num_features]))])
            with open("xgboost_model.onnx","wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"Model saved as {onnx_path}")
        except Exception as e:
            print(f"Error with ONNX Conversion: {e}")

        # Free memory
        del model, dtrain, dtest, deval

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

    eval_attack_names = X_eval["Attack Name"].copy() # auxiliary for later
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

    return X_train, X_test ,X_eval , y_train, y_test, y_eval, eval_attack_names

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

def extract_correlation_info(df, threshold):
    # Plot full correlation matrix
    corr_matrix = df.corr()
    sns.clustermap(corr_matrix, cmap='coolwarm', linewidths =0.5,  figsize=(36, 36))
    plt.show()

    # Print pairs which are grather than > 0.7 THRESHOLD
    high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    print("Highly correlated pairs (>0.7 or < -0.7):")
    for pair in high_corr_pairs:
        print(pair, corr_matrix.loc[pair[0], pair[1]])



main()

