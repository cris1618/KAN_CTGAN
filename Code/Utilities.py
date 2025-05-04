import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Import comparisons functions
def overall_similarity(real_df, synthetic_df, 
                       weight_mean=0.3, weight_median=0.2, weight_mode=0.1, weight_sd=0.1, weight_var=0.1,
                       weight_ks=0.2, weight_wasserstein=0.1):
    """
    Computes an overall similarity score between real and synthetic datasets based
    on normalized differences in mean, median (if numerical) and mode (if categorical) across columns.

    Parameters:
    - real_df (pd.DataFrame): Real dataset
    - synthetic_df (pd.DataFrame): Synthetic dataset
    - weight_mean (float, optional): Weight for mean difference. Defaults to 0.3.
    - weight_median (float, optional): Weight for median difference. Defaults to 0.3.
    - weight_mode (float, optional): Weight for mode difference. Defaults to 0.2.
    - weight_sd (float, optional): Weight for standard deviation difference. Defaults to 0.1
    - weight_var (float, optional): Weight for variance difference. Defaults to 0.1

    Returns:
    Returns a score between 0 and 100, where 100 indicates perfect similarity.
    """

    # Drop datetime columns
    real_df = real_df.select_dtypes(exclude=["datetime64"])
    synthetic_df = synthetic_df.select_dtypes(exclude=["datetime64"])

    scores = []
    common_cols = set(real_df.columns).intersection(set(synthetic_df.columns))

    for col in common_cols:
        # Check if numerical
        if pd.api.types.is_numeric_dtype(real_df[col]):
            # Classic statistics
            real_mean, syn_mean = real_df[col].mean(), synthetic_df[col].mean() 
            real_median, syn_median = real_df[col].median(), synthetic_df[col].median()
            real_sd, syn_sd = real_df[col].std(), synthetic_df[col].std()
            real_var, syn_var = real_df[col].var(), synthetic_df[col].var()
            
            # Avoid division by zero
            norm_mean = min(1, abs(real_mean - syn_mean) / (abs(real_mean) + 1e-6))
            norm_median = min(1, abs(real_median - syn_median) / (abs(real_median) + 1e-6))
            norm_sd = min(1, abs(real_sd - syn_sd) / (abs(real_sd) + 1e-6))
            norm_var = min(1, abs(real_var - syn_var) / (abs(real_var) + 1e-6))
      
            # Kolomogorov-Smirnov Test (Checks if distributions are similar)
            ks_stat, _ = ks_2samp(real_df[col].dropna(), synthetic_df[col].dropna())
        
            # Wasserstein Distance (Lower means closer distributions)
            wasserstein_dist = wasserstein_distance(real_df[col].dropna(), synthetic_df[col])
            norm_wasserstein = 1 / (1 + wasserstein_dist) # now between 0 and 1

            col_score = 1 - (weight_mean * norm_mean + 
                             weight_median * norm_median + 
                             weight_sd * norm_sd + 
                             weight_var * norm_var +
                             weight_ks * ks_stat + 
                             weight_wasserstein * (1 - norm_wasserstein))
        
        else:
            real_mode = real_df[col].mode()
            syn_mode = synthetic_df[col].mode()
            if not real_mode.empty and not syn_mode.empty:
                mode_score = 1.0 if real_mode.iloc[0] == syn_mode.iloc[0] else 0.0
            else:
                mode_score = 0.0
            col_score = mode_score* weight_mode
        
        scores.append(col_score)
    
    overall_score = np.mean(scores) * 100
    return round(overall_score, 2)

# Machine Learning efficacy comparison
def evaluate_all_models(X_real, y_real, synthetic_datasets, models, test_size=0.2, random_state=1618, repeats=10):
    """
    Evaluate all models on all synthetic datasets using repeated holdout for robust results.

    Parameters:
    - X_real (pd.DataFrame): Features from the real dataset
    - y_real (pd.Series): Target variable from the real dataset
    - synthetic_datasets (dict): Dictionary of synthetic datasets
    - models (dict): Dictionary of models to evaluate
    - test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
    - random_state (int, optional): Seed for random number generation. Defaults to 1618
    - repeats (int, optional): Number of random splits of the data for statistical significant results. Defaults to 1

    Returns:
    - real_metrics_df (pd.DataFrame): Evaluation metrics for each model on real data
    - overall_syn_metrics_df (pd.DataFrame): Average Evaluation metrics
    - detailed_syn_metrics (dict): Nested dictionary of metrics for each synthetic data generator 
    """

    # Scale the data
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)

    # Dictionaries to hold repeated metrics
    real_metrics_accum = {model_name: [] for model_name in models}
    detailed_syn_metrics_accum = {method: {model_name: [] for model_name in models} for method in synthetic_datasets}

    # Repeated holdouts
    for i in range(repeats):
        current_seed = random_state + i # change random state everytime to evaluate different part of the dataframe

        # Split real data
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real_scaled, y_real, test_size=test_size, random_state=current_seed
        )

        # Evaluate all models on real data
        for model_name, model in models.items():
            model_clone = clone(model)
            model_clone.fit(X_train_real, y_train_real)
            y_pred_real = model_clone.predict(X_test_real)

            real_metrics_accum[model_name].append({
                "MAE": mean_absolute_error(y_test_real, y_pred_real),
                "MSE": mean_squared_error(y_test_real, y_pred_real),
                "R2": r2_score(y_test_real, y_pred_real)
            })
        
        # Evaluate synthetic data 
        for method, (X_syn, y_syn) in synthetic_datasets.items():
            X_syn_scaled = scaler.transform(X_syn)

            # Split the syn data (To ensure same proportions)
            X_train_syn, _ , y_train_syn, _ = train_test_split(
                X_syn_scaled, y_syn, test_size=test_size, random_state=current_seed
            )

            for model_name, model in models.items():
                model_clone = clone(model)
                model_clone.fit(X_train_syn, y_train_syn)
                y_pred_syn = model_clone.predict(X_test_real) # Test on REAL data
                
                # Test everything on REAL data
                detailed_syn_metrics_accum[method][model_name].append({
                    "MAE": mean_absolute_error(y_test_real, y_pred_syn),
                    "MSE": mean_squared_error(y_test_real, y_pred_syn),
                    "R2": r2_score(y_test_real, y_pred_syn)
                })
    
    # Avarege all the results
    real_metrics = {
        model_name: {
            metric: np.mean([res[metric] for res in results])
            for metric in ["MAE", "MSE", "R2"]
        }
        for model_name, results in real_metrics_accum.items()
    }

    real_metrics_df = pd.DataFrame(real_metrics).T

    # Same for synthetic data
    detailed_syn_metrics = {
        method: {
            model_name: {
                metric: np.mean([res[metric] for res in results])
                for metric in ["MAE", "MSE", "R2"]
            }
            for model_name, results in model_dict.items()
        }
        for method, model_dict in detailed_syn_metrics_accum.items()
    }

    # Compute overall avareges for synthetic data
    overall_syn_metrics = {
        method: {
            "MAE_avg": np.mean([metrics["MAE"] for metrics in model_dict.values()]),
            "MSE_avg": np.mean([metrics["MSE"] for metrics in model_dict.values()]),
            "R2_avg": np.mean([metrics["R2"] for metrics in model_dict.values()]),
        }
        for method, model_dict in detailed_syn_metrics.items()
    }

    overall_syn_metrics_df = pd.DataFrame(overall_syn_metrics).T # Transpose to have metrics as columns

    return real_metrics_df, overall_syn_metrics_df, detailed_syn_metrics