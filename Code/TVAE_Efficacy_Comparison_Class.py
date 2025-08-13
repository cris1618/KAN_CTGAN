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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from Utilities import overall_similarity, evaluate_all_models_classification

"""# Import the data 
real_df = pd.read_csv("TestDatasets/adult.csv") # Binary Classification
# real_df = pd.read_csv("TestDatasets/covtype.csv") # TARGET COLUMN: Cover_Type # Multiclass 7 classes
# real_df = pd.read_csv("TestDatasets/alarm.csv") # TARGET COLUMN: AlarmSeverityName # Multiclass 3 classes
# real_df = pd.read_csv("TestDatasets/credit.csv") # TARGET COLUMN: loan_status # Binary Classification
synthetic_df_STANDARD_TVAE = pd.read_csv("TestDatasets/AdultSynthetic/synthetic_df_Adult_STANDARD_TVAE.csv")
synthetic_df_KAN_TVAE = pd.read_csv("TestDatasets/AdultSynthetic/synthetic_df_Adult_KAN_TVAE.csv")
synthetic_df_hybrid_KAN_TVAE = pd.read_csv("TestDatasets/AdultSynthetic/synthetic_df_Adult_Hybrid_KAN_TVAE.csv")

# Split the real dataset in two random subsets (TO TEST THE FUNCTION)
real_data_part_1, real_data_part_2 = train_test_split(real_df, test_size=0.5, random_state=1618)

# Evaluate the two parts on the statistical function
sim_score_test = overall_similarity(real_data_part_1, real_data_part_2)
print(f"Similarity score: {sim_score_test}")

sim_score_STANDARD_TVAE = overall_similarity(real_df, synthetic_df_STANDARD_TVAE)
print("Similarity between real data and synthetic data with Standard TVAE: ", sim_score_STANDARD_TVAE)

sim_score_KAN_TVAE = overall_similarity(real_df, synthetic_df_KAN_TVAE)
print("Similarity between real data and synthetic data with KAN TVAE: ", sim_score_KAN_TVAE)

sim_score_Hybrid_KAN_TVAE = overall_similarity(real_df, synthetic_df_hybrid_KAN_TVAE)
print("Similarity between real data and synthetic data with Hybrid KAN TVAE: ", sim_score_Hybrid_KAN_TVAE)

# Evaluate the ML efficacy
# Divide all dataframes in training and targets
X_real = real_df.drop(["income"], axis=1)
y_real = real_df["income"]

X_STANDARD_TVAE = synthetic_df_STANDARD_TVAE.drop(["income"], axis=1)
y_STANDARD_TVAE = synthetic_df_STANDARD_TVAE["income"]

X_KAN_TVAE = synthetic_df_KAN_TVAE.drop(["income"], axis=1)
y_KAN_TVAE = synthetic_df_KAN_TVAE["income"]

X_Hybrid_KAN_TVAE = synthetic_df_hybrid_KAN_TVAE.drop(["income"], axis=1)
y_Hybrid_KAN_TVAE = synthetic_df_hybrid_KAN_TVAE["income"]

# Create a dictionary for the synthetic data and one for the ML models that will be used
synthetic_datasets = {
    "STANDARD TVAE": (X_STANDARD_TVAE, y_STANDARD_TVAE),
    "KAN TVAE": (X_KAN_TVAE, y_KAN_TVAE),
    "Hybrid KAN TVAE": (X_Hybrid_KAN_TVAE, y_Hybrid_KAN_TVAE)
}

models = {
    "Logistic":    LogisticRegression(max_iter=1000, random_state=1618),
    "RF":          RandomForestClassifier(n_estimators=240, max_depth=40, max_features="sqrt", random_state=1618),
    "XGB":         XGBClassifier(colsample_bytree=0.8, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=1.0, random_state=1618),
    "SVC":         SVC(C=8, gamma=1, kernel="rbf", probability=True, random_state=1618),
}

# Create fake dictionary
real_data_dict = {
    "Real_Data": (X_real, y_real)
}

# Evaluate function
# THIS IS TO PROVE THAT IF THE TRAINING DATA ARE VERY SIMILAR (BOTH REAL) THEN THE PERFORMANCE WILL BE ALMOST IDENTICAL
real_metric_1, real_metric_2, _ = evaluate_all_models(X_real, y_real, real_data_dict, models, test_size=0.2, random_state=1618, repeats=5)
real_metric_2.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/TEST_EQUAL_TO_REAL.csv", index=False)
print(real_metric_1.mean()[["MAE", "MSE", "R2"]])
print(real_metric_2.head())

# Create the metrics datasets
real_metrics_df, overall_syn_metrics_df_TVAE, detailed_syn_metrics_TVAE = evaluate_all_models_classification(X_real, y_real, synthetic_datasets, models, test_size=0.2, random_state=1618, repeats=10)

real_metrics_df.to_csv("TestDatasets/AdultSynthetic/SyntheticPerformance/real_metrics_adult.csv", index=False)
overall_syn_metrics_df_TVAE.to_csv("TestDatasets/AdultSynthetic/SyntheticPerformance/overall_syn_metrics_adult_TVAE.csv", index=False)
print(real_metrics_df.head())
print(overall_syn_metrics_df_TVAE.head())
print(detailed_syn_metrics_TVAE)"""

# Import the datasets with performances
real_metrics_df = pd.read_csv("TestDatasets/AlarmSynthetic/SyntheticPerformanceFromCluster/real_metrics_TVAE.csv")
overall_syn_metrics_df_TVAE = pd.read_csv("TestDatasets/AlarmSynthetic/SyntheticPerformanceFromCluster/overall_syn_metrics_TVAE.csv")
#TEST = pd.read_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/TEST_EQUAL_TO_REAL.csv")

# Create diff metrics to store the differences in performance from the original data
# Compute the differences
diff = overall_syn_metrics_df_TVAE.copy()
for metric in diff.columns:
    diff[f"Delta_{metric}"] = abs(real_metrics_df.at[0, metric] - diff[metric])

# Compute per-metric score = 1 - (delta/real)
for metric in diff.columns:
    if metric.startswith("Delta_"):
        base = real_metrics_df.at[0, metric.replace("Delta_", "")]
        diff[f"{metric.replace('Delta_', '')} Score"] = 1 - (diff[metric] / base)

# Overall score
score_cols = [c for c in diff.columns if c.endswith("Score")]
diff["Overall_Score"] = diff[score_cols].mean(axis=1)
print(diff["Overall_Score"])

model_names = ["Standard TVAE", "KAN TVAE", "Hybrid TVAE"]
diff.index = model_names

# Visualize the rank
fig, ax = plt.subplots(1,1,figsize=(12,6))
bars = ax.bar(diff.index, diff["Overall_Score"], color="green", edgecolor="black")

for bar in bars:
    height = bar.get_height()
    if height >= 0:
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f"{height:.2f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height - 0.02,
                f"{height:.2f}",
                ha="center", va="top",
                fontsize=10, fontweight="bold")

ax.set_title("Overall Performance of Synthetic Data Generators")
ax.set_ylabel("Overall Score (Better if closer to 1)")
ax.set_xlabel("Synthetic Data Generator")
ax.set_xticklabels(diff.index, rotation=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


