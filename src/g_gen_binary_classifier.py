import time
from pathlib import Path
from datetime import datetime
import json

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_curve, auc, f1_score, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ------------------------------
# Helpers
# ------------------------------
def train_test_eval_split(df):
    X = df.drop(columns=["Label"])
    Y = df["Label"]

    # Train 70 / Temp 30 (estratificado por attack Name)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=df["Attack Name"]
    )

    # Segundo split: Test 20, Eval 10
    df_temp = df.loc[X_temp.index].copy()
    X_test, X_eval, y_test, y_eval = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=df_temp["Attack Name"]
    )

    attack_names_test = X_test["Attack Name"].copy()

    # Borrar attack Name al estratificar
    X_train = X_train.drop(columns=["Attack Name"])
    X_test  = X_test.drop(columns=["Attack Name"])
    X_eval  = X_eval.drop(columns=["Attack Name"])

    return X_train, X_test, X_eval, y_train, y_test, y_eval, attack_names_test


def drop_and_encode(X_train, X_test, X_eval):
    # Antes si servia, ahora ya no utilizamos service
    for frame in (X_train, X_test, X_eval):
        if "Service" in frame.columns:
            frame.drop(columns="Service", inplace=True)
    return X_train, X_test, X_eval


def remove_correlated_features(X, y, threshold: float = 0.9, save_path: Path = None):
    """
    Eliminar atributos correlacionados y guardar el analisis en un fichero json

    Args:
        X: Feature dataframe
        y: Target variable
        threshold: Correlation threshold (default 0.9)
        save_path: Path to save correlation analysis results

    Returns:
        X_reduced: DataFrame with correlated features removed
        to_drop: List of dropped feature names
    """
    corr_matrix = X.corr(numeric_only=True).abs()

    # Calcular correlacion con label
    label_corr = {}
    for col in X.columns:
        label_corr[col] = np.abs(pd.concat([X[col], y], axis=1).corr().iloc[0, -1])

    # Matriz triangular de la matrix de correlacion
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlation_pairs = []
    to_drop = set()

    for col in upper.columns:
        if col in to_drop:
            continue

        correlated = []
        # Busco todas las parejas que superen el threshold
        for r in upper.index:
            if r != col and upper.loc[r, col] > threshold:
                correlated.append(r)
                # Store the correlation pair
                correlation_pairs.append({
                    'feature_1': col,
                    'feature_2': r,
                    'correlation': float(upper.loc[r, col]),
                    'feature_1_label_corr': float(label_corr.get(col, 0)),
                    'feature_2_label_corr': float(label_corr.get(r, 0))
                })

        if not correlated:
            continue


        correlated.append(col)

        # Me quedo solo las que mas esten correlacionadas con label
        keep = max(correlated, key=lambda c: label_corr.get(c, 0))

        # Marcamos que atributos eliminamos
        for c in correlated:
            if c != keep:
                to_drop.add(c)
                # Y actualizamos
                for pair in correlation_pairs:
                    if (pair['feature_1'] == c and pair['feature_2'] == keep) or \
                            (pair['feature_1'] == keep and pair['feature_2'] == c):
                        pair['kept_feature'] = keep
                        pair['removed_feature'] = c

    if save_path:
        # Report final
        analysis_results = {
            'threshold': threshold,
            'total_features_original': len(X.columns),
            'total_features_removed': len(to_drop),
            'total_features_kept': len(X.columns) - len(to_drop),
            'removed_features': sorted(list(to_drop)),
            'correlation_pairs': sorted(correlation_pairs,
                                        key=lambda x: x['correlation'],
                                        reverse=True)
        }

        json_path = save_path / 'correlation_analysis.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    return X.drop(columns=list(to_drop)), sorted(to_drop)


def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)



def main():
    pd.set_option("display.max_columns", None)

    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" / "combinado_balanceado.csv"
    out_dir   = base_path / "output"
    mdl_dir   = out_dir / "models"
    plots_dir = out_dir / "plots"
    cm_dir    = out_dir / "confusion_matrix"
    corr_dir = out_dir / "correlation"
    ensure_dirs(out_dir, mdl_dir, plots_dir, cm_dir, corr_dir)

    df = pd.read_csv(data_path, low_memory=False)

    if "Attack Category" in df.columns:
        df = df.drop(columns="Attack Category")

    # Split
    X_train, X_test, X_eval, y_train, y_test, y_eval, attack_names_test = train_test_eval_split(df)

    # Preprocessing
    X_train, X_test, X_eval = drop_and_encode(X_train, X_test, X_eval)

    # Correlation pruning
    X_train, dropped = remove_correlated_features(X_train, y_train, threshold=0.9, save_path=corr_dir)
    X_test  = X_test.drop(columns=dropped, errors="ignore")
    X_eval  = X_eval.drop(columns=dropped, errors="ignore")

    # Guardamos el mapeo de atributos al formato f{i} que es como lo espera ONNX
    feature_map_path = mdl_dir / "feature_map_reduced.txt"
    orig_feature_names = list(X_train.columns)
    with open(feature_map_path, "w") as f:
        for i, name in enumerate(X_train.columns):
            f.write(f"{i} {name}\n")

    # Y las actualizamos
    X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_test.columns  = [f"f{i}" for i in range(X_test.shape[1])]
    X_eval.columns  = [f"f{i}" for i in range(X_eval.shape[1])]

    # DMatrix + GPU arrays
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_eval  = xgb.DMatrix(X_eval,  label=y_eval)
    d_test  = xgb.DMatrix(X_test,  label=y_test)

    X_eval_cu = cp.asarray(X_eval.values)
    X_test_cu = cp.asarray(X_test.values)

    # scale_pos_weight from TRAIN
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight (from train): {scale_pos_weight:.3f}")

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "device": "cuda",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight",
                                                    0.5 * scale_pos_weight,
                                                    2.0 * scale_pos_weight),
        }
        model = xgb.train(
            params,
            d_train,
            num_boost_round=2000,
            evals=[(d_eval, "validation")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        # El mejor score es AUCPR on eval
        return model.best_score

    print("\n>>> Optuna: maximizing AUCPR on eval")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print("Best params:", best_params)

    # Parametros de entrenamiento
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",  # still AUCPR
        "device": "cuda",
        **best_params,
    }

    print("\n>>> Training final model")
    start = time.time()
    model = xgb.train(
        final_params,
        d_train,
        num_boost_round=4000,
        evals=[(d_train, "train"), (d_eval, "eval")],
        early_stopping_rounds=150,
        verbose_eval=200,
    )
    train_time = time.time() - start
    print(f"Training done in {train_time:.1f}s. Best iters: {model.best_iteration}")

    # Eleccion del THRESHOLD operacional
    eval_proba = model.inplace_predict(X_eval_cu)

    MINIMUM_RECALL_TARGET = 0.98

    # Calculamos la curva ROC en las probabilidades CALIBRADAS
    fprs, tprs, thresholds = roc_curve(y_eval, eval_proba.get())

    # Buscamos todos los thresholds que satisfacen nuestra restriccion
    valid_indices = np.where(tprs >= MINIMUM_RECALL_TARGET)[0]

    if len(valid_indices) == 0:
        raise ValueError(f"Could not meet the recall target of {MINIMUM_RECALL_TARGET}. "
                         "The best achievable recall was {tprs.max():.4f}. "
                         "Consider lowering the target or improving the model.")

    # Escogemos el que tiene una menor tasa de falsos positivos
    best_index = valid_indices[fprs[valid_indices].argmin()]
    best_thr = thresholds[best_index]

    eval_fpr_at_best_thr = fprs[best_index]
    eval_tpr_at_best_thr = tprs[best_index]

    print(f"Constraint: Minimum Recall (TPR) >= {MINIMUM_RECALL_TARGET:.4f}")
    print(f"Chosen Threshold: {best_thr:.4f}")
    print(f"  > Achieved on Eval Set: TPR (Recall) = {eval_tpr_at_best_thr:.4f}, FPR = {eval_fpr_at_best_thr:.4f}")

    # IMPORTANCIA DE ATRIBUTOS
    model_feature_names = [f"f{i}" for i in range(len(orig_feature_names))]
    idx_to_name = {f"f{i}": orig_feature_names[i] for i in range(len(orig_feature_names))}

    imp_gain = model.get_score(importance_type="gain")
    imp_weight = model.get_score(importance_type="weight")
    imp_total_gain = model.get_score(importance_type="total_gain")
    imp_cover = model.get_score(importance_type="cover")
    imp_total_cover = model.get_score(importance_type="total_cover")

    rows = []
    for f in model_feature_names:
        rows.append({
            "feature_model": f,
            "feature_original": idx_to_name.get(f, f),
            "gain": float(imp_gain.get(f, 0.0)),
            "weight": float(imp_weight.get(f, 0.0)),  # split count
            "total_gain": float(imp_total_gain.get(f, 0.0)),
            "cover": float(imp_cover.get(f, 0.0)),
            "total_cover": float(imp_total_cover.get(f, 0.0)),
        })

    fi_df = pd.DataFrame(rows)

    fi_df["gain_norm"] = fi_df["gain"] / (fi_df["gain"].sum() or 1.0)
    fi_df["tgain_norm"] = fi_df["total_gain"] / (fi_df["total_gain"].sum() or 1.0)

    fi_df.sort_values("total_gain", ascending=False, inplace=True)


    TOP_N = 30
    top_df = fi_df.head(TOP_N).iloc[::-1]  # reverse for nicer horizontal bars
    plt.figure(figsize=(10, max(6, TOP_N * 0.3)))
    plt.barh(top_df["feature_original"], top_df["total_gain"])
    plt.xlabel("Total Gain")
    plt.title(f"Top {TOP_N} Feature Importance (Total Gain)")
    plt.tight_layout()
    fi_png_path = plots_dir / f"feature_importance_top{TOP_N}_binary.png"
    plt.savefig(fi_png_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {fi_png_path}")

    test_proba = model.inplace_predict(X_test_cu)
    test_pred  = (test_proba > best_thr).astype(int).get()

    acc = accuracy_score(y_test, test_pred)
    print(f"\nAccuracy (test): {acc:.4f}")
    print("Classification report (test):")
    print(classification_report(y_test, test_pred, target_names=["BENIGN","ATTACK"]))

    # Curves + metrics
    fpr, tpr, _ = roc_curve(y_test, test_proba.get())
    roc_auc = auc(fpr, tpr)
    pr, rc, _ = precision_recall_curve(y_test, test_proba.get())
    ap = average_precision_score(y_test, test_proba.get())
    print(f"ROC-AUC (test): {roc_auc:.4f}")
    print(f"PR-AUC / Average Precision (test): {ap:.4f}")

    # PLOTS
    # PR curve
    plt.figure(figsize=(8,6))
    plt.plot(rc, pr, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall (test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "pr_curve_test.png", dpi=200)
    plt.close()

    # ROC
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "roc_curve_test.png", dpi=200)
    plt.close()

    # Confusion matrix (raw counts)
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["BENIGN","ATTACK"], yticklabels=["BENIGN","ATTACK"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (test)")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(cm_dir / f"cm_{ts}.png", dpi=200)
    plt.close()

    # Optional: threshold analysis plot on test probs
    plot_threshold_analysis(y_test, test_proba.get(), save_path=plots_dir / "threshold_sweep_test.png")

    num_features = X_train.shape[1]
    onnx_path = mdl_dir / f"xgb_binary_{ts}.onnx"
    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, num_features]))]
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX to {onnx_path}")

    meta = {
        "timestamp": ts,
        "num_features": num_features,
        "features": [f"f{i}" for i in range(num_features)],
        "dropped_features": dropped,
        "final_params": final_params,
        "best_iteration": int(model.best_iteration or 0),
        "train_time_sec": round(train_time, 3),
        "eval_threshold": float(best_thr),
        "metrics_test": {
            "accuracy": float(acc),
            "roc_auc": float(roc_auc),
            "average_precision": float(ap),
        },
        "scale_pos_weight_train": float(scale_pos_weight),
    }
    with open(mdl_dir / f"model_meta_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")



def plot_threshold_analysis(y_true, y_proba, save_path=None):
    thresholds = np.linspace(0.01, 0.99, 50)
    prec, rec, f1, fpr = [], [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1.append(f1_score(y_true, y_pred))
        prec.append(precision); rec.append(recall)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)

    plt.figure(figsize=(9,6))
    plt.plot(thresholds, prec, label="Precision")
    plt.plot(thresholds, rec,  label="Recall")
    plt.plot(thresholds, f1,   label="F1")
    plt.plot(thresholds, fpr,  label="FPR")
    plt.axvline(0.5, ls="--", alpha=0.6, label="0.5")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold Sweep (test)")
    plt.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
