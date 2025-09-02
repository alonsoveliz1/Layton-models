import time
from pathlib import Path
from datetime import datetime
import json

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
from sklearn.metrics import precision_recall_fscore_support
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def train_test_eval_split(df: pd.DataFrame):
    """
    Features = all except 'Attack Category' (target) + 'Attack Name' (removed after split).
    Target    = 'Attack Category'.
    Stratify by Attack Name to keep subtype mix across splits.
    """

    # Eliminamos la Label a predecir
    X = df.drop(columns=["Attack Category"])
    y = df["Attack Category"]

    # Train 70 / Temp 30 (estratificamos por Nombre del Ataque)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=df["Attack Name"]
    )

    # Second split: Test (20), Eval (10) → i.e. 2/3 test, 1/3 eval of X_temp
    df_temp = df.loc[X_temp.index].copy()
    X_test, X_eval, y_test, y_eval = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=df_temp["Attack Name"]
    )

    # Guardamos el nombre del ataque y lo eliminamos
    attack_names_test = X_test["Attack Name"].copy()

    X_train = X_train.drop(columns=["Attack Name"])
    X_test  = X_test.drop(columns=["Attack Name"])
    X_eval  = X_eval.drop(columns=["Attack Name"])

    return X_train, X_test, X_eval, y_train, y_test, y_eval, attack_names_test


def drop_and_encode(X_train, X_test, X_eval):
    for frame in (X_train, X_test, X_eval):
        if "Service" in frame.columns:
            frame.drop(columns="Service", inplace=True)
    return X_train, X_test, X_eval


def remove_correlated_features(X: pd.DataFrame, y: pd.Series, threshold: float = 0.9):
    """
    Correlation pruning among numeric features:
      - Build abs corr matrix among X only
      - From each correlated group (> threshold), keep the feature more correlated with y
    """
    corr_matrix = X.corr(numeric_only=True).abs()

    label_corr = {}
    for col in X.columns:
        try:
            label_corr[col] = np.abs(pd.concat([X[col], y], axis=1).corr().iloc[0, -1])
        except Exception:
            label_corr[col] = 0.0

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = [r for r in upper.index if upper.loc[r, col] > threshold and r != col]
        if not correlated:
            continue
        correlated.append(col)
        keep = max(correlated, key=lambda c: label_corr.get(c, 0))
        to_drop.update([c for c in correlated if c != keep])

    return X.drop(columns=list(to_drop)), sorted(to_drop)


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)



def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    f1_per_class: np.ndarray,
    support: np.ndarray,
    macro_f1: float,
    acc: float,
    save_path: Path,
):
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums > 0)
    cm_norm[np.isnan(cm_norm)] = 0.0

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm_norm[i, j]
            annot[i, j] = f"{cm[i, j]}\n{pct:.0f}%"

    ytick = [f"{c}\n(n={int(support[k])}, F1={f1_per_class[k]:.2f})"
             for k, c in enumerate(class_names)]

    h = max(6, 0.8 * len(class_names))
    w = max(8, 0.8 * len(class_names))
    plt.figure(figsize=(w, h))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        xticklabels=class_names,
        yticklabels=ytick,
        cbar_kws={"label": "Row-normalized (recall)"}
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Multiclass Confusion Matrix (row-normalized)\n"
              f"Accuracy={acc:.3f} • Macro-F1={macro_f1:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def main():
    pd.set_option("display.max_columns", None)

    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "processed" / "CIC-BCCC-NRC-TabularIoT-2024-MOD" / "combinado_balanceado_ataques.csv"
    out_dir   = base_path / "output"
    mdl_dir   = out_dir / "models"
    plots_dir = out_dir / "plots"
    cm_dir    = out_dir / "confusion_matrix"
    ensure_dirs(out_dir, mdl_dir, plots_dir, cm_dir)

    df = pd.read_csv(data_path, low_memory=False)

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    X_train, X_test, X_eval, y_train_str, y_test_str, y_eval_str, attack_names_test = train_test_eval_split(df)

    le = LabelEncoder()
    le.fit(pd.concat([y_train_str, y_test_str, y_eval_str], axis=0))
    class_names = list(le.classes_)
    num_classes = len(class_names)
    y_train = pd.Series(le.transform(y_train_str), index=y_train_str.index)
    y_test  = pd.Series(le.transform(y_test_str),  index=y_test_str.index)
    y_eval  = pd.Series(le.transform(y_eval_str),  index=y_eval_str.index)

    X_train, X_test, X_eval = drop_and_encode(X_train, X_test, X_eval)
    X_train, dropped = remove_correlated_features(X_train, y_train, threshold=0.9)
    X_test  = X_test.drop(columns=dropped, errors="ignore")
    X_eval  = X_eval.drop(columns=dropped, errors="ignore")

    feature_map_path = mdl_dir / "feature_map_reduced_l2.txt"
    orig_feature_names = list(X_train.columns)
    with open(feature_map_path, "w") as f:
        for i, name in enumerate(X_train.columns):
            f.write(f"{i} {name}\n")

    X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_test.columns  = [f"f{i}" for i in range(X_test.shape[1])]
    X_eval.columns  = [f"f{i}" for i in range(X_eval.shape[1])]

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_eval  = xgb.DMatrix(X_eval,  label=y_eval)
    d_test  = xgb.DMatrix(X_test,  label=y_test)

    X_eval_cu = cp.asarray(X_eval.values)
    X_test_cu = cp.asarray(X_test.values)

    def objective(trial):
        params = {
            "objective": "multi:softprob",  # multiclass probabilities
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "device": "cuda",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 0.0, 5.0),
        }
        model = xgb.train(
            params,
            d_train,
            num_boost_round=2000,
            evals=[(d_eval, "validation")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        # Best score is mlogloss on eval (lower is better) → return negative so Optuna maximizes
        return -float(model.best_score)

    print("\n>>> Optuna: maximizing -mlogloss on eval (i.e., minimizing mlogloss)")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print("Best params:", best_params)

    final_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
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

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    eval_proba = model.inplace_predict(X_eval_cu)
    eval_pred_ids = cp.argmax(eval_proba, axis=1).get()
    eval_pred = le.inverse_transform(eval_pred_ids)

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
    fi_png_path = plots_dir / f"feature_importance_top{TOP_N}_multiclass.png"
    plt.savefig(fi_png_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {fi_png_path}")

    test_proba = model.inplace_predict(X_test_cu)
    test_pred_ids = cp.argmax(test_proba, axis=1).get()
    test_pred = le.inverse_transform(test_pred_ids)

    acc = accuracy_score(y_test_str, test_pred)
    macro_f1 = f1_score(y_test_str, test_pred, average="macro")
    print(f"\nAccuracy (test): {acc:.4f}")
    print(f"Macro F1  (test): {macro_f1:.4f}\n")
    print("Classification report (test):")
    print(classification_report(y_test_str, test_pred, labels=class_names))

    cm = confusion_matrix(y_test_str, test_pred, labels=class_names)
    prec, rec, f1_cls, support = precision_recall_fscore_support(
        y_test_str, test_pred, labels=class_names, zero_division=0
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        f1_per_class=f1_cls,
        support=support,
        macro_f1=macro_f1,
        acc=acc,
        save_path=cm_dir / f"cm_multiclass_with_counts_perc_{ts}.png",
    )

    top1_scores = test_proba.max(axis=1).get()
    top1_ids = test_pred_ids

    top3_ids = cp.argsort(test_proba, axis=1)[:, -3:][:, ::-1].get()
    top3_labels = [[class_names[j] for j in row] for row in top3_ids]
    top3_scores = [[float(test_proba[i, j]) for j in row] for i, row in enumerate(top3_ids)]

    pred_df = pd.DataFrame({
        "AttackName_Test": attack_names_test.values,
        "y_true": y_test_str.values,
        "y_pred": test_pred,
        "y_pred_score": top1_scores,
        "top3_labels": top3_labels,
        "top3_scores": top3_scores,
    })
    pred_df.to_csv(out_dir / f"test_predictions_{ts}.csv", index=False)

    num_features = X_train.shape[1]
    onnx_path = mdl_dir / f"xgb_multiclass_{ts}.onnx"
    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[("input", FloatTensorType([None, num_features]))]
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX to {onnx_path}")

    class_map = {
        "num_classes": num_classes,
        "id_to_label": {int(i): cls for i, cls in enumerate(class_names)},
        "label_to_id": {cls: int(i) for i, cls in enumerate(class_names)},
    }
    with open(mdl_dir / f"class_map_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)
    print(f"Saved class map to {mdl_dir / f'class_map_{ts}.json'}")

    meta = {
        "timestamp": ts,
        "num_features": int(num_features),
        "features": [f"f{i}" for i in range(num_features)],
        "dropped_features": list(dropped),
        "final_params": final_params,
        "best_iteration": int(model.best_iteration or 0),
        "train_time_sec": round(train_time, 3),
        "metrics_test": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        },
        "classes": class_names,
        "data_path": str(data_path),
    }
    with open(mdl_dir / f"model_meta_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
