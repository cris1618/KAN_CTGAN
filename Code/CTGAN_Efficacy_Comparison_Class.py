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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from Utilities import overall_similarity, evaluate_all_models_classification

"""# Import the data 
real_df = pd.read_csv("TestDatasets/adut.csv")
# real_df = pd.read_csv("TestDatasets/covtype.csv") # TARGET COLUMN: Cover_Type
# real_df = pd.read_csv("TestDatasets/alarm.csv") # TARGET COLUMN: AlarmSeverityName
# real_df = pd.read_csv("TestDatasets/credit.csv") # TARGET COLUMN: loan_status
synthetic_df_STANDARD_CTGAN = pd.read_csv("TestDatasets/AdultSynthetic/")
synthetic_df_KAN_CTGAN = pd.read_csv("TestDatasets/AdultSynthetic/")
synthetc_df_HYBRID_CTGAN = pd.read_csv("TestDatasets/AdultSynthetic/")

# Split the real dataset in two random subsets (TO TEST THE FUNCTION)
real_data_part_1, real_data_part_2 = train_test_split(real_df, test_size=0.5, random_state=1618)

# Evaluate the two parts on the statistical function
sim_score_test = overall_similarity(real_data_part_1, real_data_part_2)
print(f"Similarity score: {sim_score_test}")

sim_score_STANDARD_CTGAN = overall_similarity(real_df, synthetic_df_STANDARD_CTGAN)
print("Similarity between real data and synthetic data with Standard CTGAN: ", sim_score_STANDARD_CTGAN)

sim_score_KAN_CTGAN = overall_similarity(real_df, synthetic_df_KAN_CTGAN)
print("Similarity between real data and synthetic data with KAN CTGAN: ", sim_score_KAN_CTGAN)

sim_score_HYBRID_CTGAN = overall_similarity(real_df, synthetc_df_HYBRID_CTGAN)
print("Similarity between real data and synthetic data with HYBRID KAN CTGAN: ", sim_score_HYBRID_CTGAN)

# Evaluate the ML efficacy
# Divide all dataframes in training and targets
X_real = real_df.drop(["Appliances"], axis=1)
y_real = real_df["Appliances"]

X_STANDARD_CTGAN = synthetic_df_STANDARD_CTGAN.drop(["Appliances"], axis=1)
y_STANDARD_CTGAN = synthetic_df_STANDARD_CTGAN["Appliances"]

X_KAN_CTGAN = synthetic_df_KAN_CTGAN.drop(["Appliances"], axis=1)
y_KAN_CTGAN = synthetic_df_KAN_CTGAN["Appliances"]

X_HYBRID_KAN_CTGAN = synthetc_df_HYBRID_CTGAN.drop(["Appliances"], axis=1)
y_HYBRID_KAN_CTGAN = synthetc_df_HYBRID_CTGAN["Appliances"]

# Create a dictionary for the synthetic data and one for the ML models that will be used
synthetic_datasets = {
    "STANDARD CTGAN": (X_STANDARD_CTGAN, y_STANDARD_CTGAN),
    "KAN CTGAN": (X_KAN_CTGAN, y_KAN_CTGAN),
    "HYBRID KAN CTGAN": (X_HYBRID_KAN_CTGAN, y_HYBRID_KAN_CTGAN)
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
}"""

"""# Evaluate function
# THIS IS TO PROVE THAT IF THE TRAINING DATA ARE VERY SIMILAR (BOTH REAL) THEN THE PERFORMANCE WILL BE ALMOST IDENTICAL
real_metric_1, real_metric_2, _ = evaluate_all_models(X_real, y_real, real_data_dict, models, test_size=0.2, random_state=1618, repeats=5)
real_metric_2.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/TEST_EQUAL_TO_REAL.csv", index=False)
print(real_metric_1.mean()[["MAE", "MSE", "R2"]])
print(real_metric_2.head())"""

"""# Create the metrics datasets
print("Start Evaluation")
real_metrics_df, overall_syn_metrics_df, detailed_syn_metrics = evaluate_all_models_classification(X_real, y_real, synthetic_datasets, models, test_size=0.2, random_state=1618, repeats=10)

real_metrics_df.to_csv("TestDatasets/AdultSynthetic/SyntheticPerformance/real_metrics.csv", index=False)
overall_syn_metrics_df.to_csv("TestDatasets/AdultSynthetic/SyntheticPerformance/overall_syn_metrics.csv", index=False)
print(real_metrics_df.head())
print(overall_syn_metrics_df.head())
print(detailed_syn_metrics)"""


# Import the datasets with performances (Classification)
real_metrics_df_class = pd.read_csv("TestDatasets/AlarmSynthetic/SyntheticPerformanceFromCluster/real_metrics.csv")
overall_syn_metrics_df_class = pd.read_csv("TestDatasets/AlarmSynthetic/SyntheticPerformanceFromCluster/overall_syn_metrics.csv")
#TEST = pd.read_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/SyntheticPerformanceFromCluster/") 

# Create diff metrics to store the differences in performance from the original data
# Compute the differences
diff = overall_syn_metrics_df_class.copy()
for metric in diff.columns:
    diff[f"Delta_{metric}"] = abs(real_metrics_df_class.at[0, metric] - diff[metric])

# Compute per-metric score = 1 - (delta/real)
for metric in diff.columns:
    if metric.startswith("Delta_"):
        base = real_metrics_df_class.at[0, metric.replace("Delta_", "")]
        diff[f"{metric.replace('Delta_', '')} Score"] = 1 - (diff[metric] / base)

# Overall score
score_cols = [c for c in diff.columns if c.endswith("Score")]
diff["Overall_Score"] = diff[score_cols].mean(axis=1)

model_names = ["Standard CTGAN", "KAN CTGAN", "Hybrid KAN CTGAN", "DISC KAN CTGAN", "GEN KAN CTGAN"]
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


