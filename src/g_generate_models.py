import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def main():
    df = pd.read_csv("../data/processed/CIC-BCCC-NRC-TabularIoT-2024-MOD/combinado_balanceado.csv")

    # (1) Attributes that are not necessary for binary classifier
    df = df.drop(columns=['Attack Name','Attack Category','Timestamp'])

    # (2) Drop Label since it'll be our predicting value and timestamp
    X = df.drop(columns=['Label'])
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

    # (3) One hot encoding of categorical attributes (Service)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[categorical_features])

    X_train_ohe = ohe.transform(X_train[categorical_features])
    X_test_ohe = ohe.transform(X_test[categorical_features])
    X_eval_ohe = ohe.transform(X_eval[categorical_features])

    ohe_cols = ohe.get_feature_names_out(categorical_features)
    X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe_cols, index=X_train.index)
    X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe_cols, index=X_test.index)
    X_eval_ohe = pd.DataFrame(X_eval_ohe, columns=ohe_cols, index=X_eval.index)

    # (4) Dropping Service from the train/test/eval
    X_train_final = pd.concat([X_train.drop(columns=categorical_features), X_train_ohe], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=categorical_features), X_test_ohe], axis=1)
    X_eval_final = pd.concat([X_eval.drop(columns=categorical_features), X_eval_ohe], axis=1)


    """ CORRELATION BETWEEN ATTRIBUTES"""
    # Plot full correlation matrix
    corr_matrix = X_train_final.corr()
    sns.clustermap(corr_matrix, cmap='coolwarm', linewidths =0.5,  figsize=(36, 36))
    plt.show()

    # Print pairs which are grather than > 0.7 THRESHOLD
    threshold = 0.7
    high_corr_pairs = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    print("Highly correlated pairs (>0.7 or < -0.7):")
    for pair in high_corr_pairs:
        print(pair, corr_matrix.loc[pair[0], pair[1]])

    # Dimensionality Reduction (I only drop the attributes less related to the label
    X_train_reduced, dropped_features = remove_correlated_features(X_train_final, y_train, threshold)
    X_test_reduced = X_test_final.drop(columns = dropped_features)
    X_eval_reduced = X_eval_final.drop(columns=dropped_features)

    # Plot again the full correlated matrix with less attributes (41 less)
    corr_matrix = X_train_reduced.corr()
    sns.clustermap(corr_matrix, cmap='coolwarm', linewidths=0.5, figsize=(36, 36))
    plt.show()

    """FEATURE IMPORTANCE"""
    # We get again numerical & categorical features (since a lot of em have been dropped)
    numerical_features = X_train_reduced.select_dtypes(include=np.number).columns
    categorical_features = X_train_reduced.select_dtypes(exclude=np.number).columns

    X_train_scaled = robust_scaler_with_epsilon(X_train_reduced, numerical_features)

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

    plt.figure(figsize=(20, 12))
    plt.title("Feature Importances (RandomForest)")
    plt.bar([top_features[i] for i in range(len(top_features))], [top_importances[i] for i in range(len(top_importances))])
    plt.xticks(rotation=90)
    plt.show()

    """
    # Set up PyCaret with optimized parameters
    exp1 = setup(
        df,
        target='Label',
        session_id=42,
        normalize=True,
        use_gpu=True,
        verbose=True
    )
    
    print("Starting compare models with pycaret\n")
    # Specify LightGBM parameters when creating the model
    best_model = compare_models()

    # Tune the best model
    tuned_model = tune_model(best_model)

    # Evaluate the model
    evaluate_model(tuned_model)

    # Optional: Save the model
    save_model(tuned_model, 'FinalModel_22_03')
    """

    results = {}

    # XGBoost model
    try:
        import xgboost as xgb
        print("\nTRAINING XGBOOST\n")

        start_time = time.time()

        # Convert to DMatrix for faster processing
        dtrain = xgb.DMatrix(X_train_reduced, label=y_train)
        dtest = xgb.DMatrix(X_test_reduced, label=y_test)
        deval = xgb.DMatrix(X_eval_reduced, label=y_eval)

        params = {
            'objective': 'binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softmax',
            'eval_metric': 'logloss' if len(np.unique(y_train)) == 2 else 'mlogloss', # Log regression binary clasification, softmax multiclass
            'device': "cuda", # CUDA device selection
            'max_depth': 12, # Max deph of tree
            'eta': 0.05, # Learning rate
            'min_child_weight': 1, # Minimum sum of instance weight needed in a child
            'subsample': 0.9, # Ratio of training instances
            'colsample_bytree': 0.9,
            'max_bin': 256, # Bucket for continuous features
            'grow_policy': 'lossguide', # Best loss reduction
            'predictor': 'gpu_predictor', # Gpu prediction
            'sampling_method': 'gradient_based' # Sample training instances
        }

        # Add num_class for multiclass
        if len(np.unique(y_train)) > 2:
            params['num_class'] = len(np.unique(y_train))

        # Train with more boosting rounds
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dtest, 'test'), (deval, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=50
        )

        # Make predictions
        y_pred = model.predict(deval)
        if len(np.unique(y_train)) > 2:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_eval, y_pred)

        print(f"XGBoost training completed in {time.time() - start_time:.2f} seconds")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_eval, y_pred))

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

        # Free memory
        del model, dtrain, dtest, deval
        gc.collect()

    except Exception as e:
        print(f"Error with XGBoost: {e}")

    # LightGBM Model
    try:
        import lightgbm as lgb
        print("\nTRAINING LIGHTGM\n")

        start_time = time.time()

        # GPU parameters
        params = {
            'objective': 'binary' if len(np.unique(y_train)) == 2 else 'multiclass',
            'metric': 'binary_logloss' if len(np.unique(y_train)) == 2 else 'multi_logloss',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 15,
            'num_leaves': 255,
            'learning_rate': 0.05,
            'verbose': 1,
            'min_data_in_leaf': 1,
            'min_data_in_bin': 1,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'feature_fraction': 0.9,
            'max_bin': 255,
            'force_row_wise': True,
            'histogram_pool_size': -1
        }

        # Add num_class for multiclass
        if len(np.unique(y_train)) > 2:
            params['num_class'] = len(np.unique(y_train))

        # Create dataset
        train_data = lgb.Dataset(X_train_reduced, label=y_train)
        test_data = lgb.Dataset(X_test_reduced, label=y_test, reference=train_data)
        eval_data = lgb.Dataset(X_eval_reduced, label=y_eval, reference=train_data)

        # Train with more rounds
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, test_data, eval_data],
            valid_names=['train', 'test', 'eval'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
        )

        # Make predictions
        y_pred = model.predict(X_eval_reduced)
        if len(np.unique(y_train)) > 2:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_eval, y_pred)

        print(f"LightGBM training completed in {time.time() - start_time:.2f} seconds")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_eval, y_pred))

        results['lightgbm'] = {
            'accuracy': acc,
            'model': model,
            'training_time': time.time() - start_time
        }

        # Get feature importance
        if hasattr(model, 'feature_importance'):
            importance = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()

            print("\nTop 10 Feature Importance:")
            sorted_idx = np.argsort(importance)[::-1]
            for i in sorted_idx[:10]:
                print(f"{feature_names[i]}: {importance[i]}")

        # Free memory
        del model, train_data, test_data, eval_data
        gc.collect()

    except Exception as e:
        print(f"Error with LightGBM: {e}")

        # PyTorch Deep Neural Network with GPU
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        print("\nTRAINING PYTORCH DNN\n")

        start_time = time.time()

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_reduced.values if hasattr(X_train_reduced, 'values') else X_train_reduced).to(device)
        y_train_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test_reduced.values if hasattr(X_test_reduced, 'values') else X_test_reduced).to(device)
        y_test_tensor = torch.LongTensor(y_test.values if hasattr(y_test, 'values') else y_test).to(device)
        X_eval_tensor = torch.FloatTensor(X_eval_reduced.values if hasattr(X_eval, 'values') else X_eval_reduced).to(device)
        y_eval_tensor = torch.LongTensor(y_eval.values if hasattr(y_eval, 'values') else y_eval).to(device)

        # Create dataset and dataloader for batch processing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # Define neural network architecture
        input_size = X_train_reduced.shape[1]
        hidden_sizes = [512, 256, 128, 64]
        output_size = len(np.unique(y_train))

        # All-in-one neural network definition
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.3),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.3),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Dropout(0.3),

            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[3]),
            nn.Dropout(0.3),

            nn.Linear(hidden_sizes[3], output_size)
        ).to(device)

        # Define loss function and optimizer
        # Since dataset aint balanced I make it so errors of the malicious class are more penalized
        weights = torch.tensor([1.0, 9.0], dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # Training loop
        epochs = 100
        best_loss = float('inf')

        print("Starting PyTorch training loop...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)

                _, predicted = torch.max(test_outputs, 1)
                test_acc = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

            # Learning rate scheduling
            scheduler.step(test_loss)

            # Print only every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            # Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), "best_dnn_model.pt")

        # Load best model
        model.load_state_dict(torch.load("best_dnn_model.pt"))

        # Final evaluation
        model.eval()
        with torch.no_grad():
            eval_outputs = model(X_eval_tensor)
            _, y_pred = torch.max(eval_outputs, 1)
            y_pred = y_pred.cpu().numpy()

        # Calculate metrics
        acc = accuracy_score(y_eval, y_pred)

        print(f"PyTorch DNN training completed in {time.time() - start_time:.2f} seconds")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_eval, y_pred))

        results['pytorch'] = {
            'accuracy': acc,
            'model': model,
            'training_time': time.time() - start_time
        }

        # Free memory
        del model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_eval_tensor, y_eval_tensor
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error with PyTorch: {e}")

        # Compare all model results
    print("\n==== MODEL COMPARISON ====")
    for model_name, model_results in results.items():
        print(
            f"{model_name.upper()}: Accuracy = {model_results['accuracy']:.4f}, Training Time = {model_results['training_time']:.2f} seconds")

    # Return the best model based on accuracy
    if results:
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\nBest model: {best_model_name.upper()} with accuracy {results[best_model_name]['accuracy']:.4f}")
        return results[best_model_name]['model']
    else:
        print("No models were successfully trained.")
        return None

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



main()