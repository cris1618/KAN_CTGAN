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
from Utilities import overall_similarity, evaluate_all_models

"""# Import the data 
real_df = pd.read_csv("TestDatasets/energydata_complete.csv")
# real_df = pd.read_csv("TestDatasets/news.csv") # TARGET COLUMN: shares
synthetic_df_STANDARD_TVAE = pd.read_csv("TestDatasets/EnergySynthetic/synthetic_df_STANDARD_TVAE.csv")
synthetic_df_KAN_TVAE = pd.read_csv("TestDatasets/EnergySynthetic/synthetic_df_KAN_TVAE.csv")
synthetic_df_hybrid_KAN_TVAE = pd.read_csv("TestDatasets/EnergySynthetic/synthetic_df_Hybrid_KAN_TVAE.csv")

# Evaluate the Predictive Efficacy
real_df = real_df.drop("date", axis=1)
synthetic_df_STANDARD_TVAE = synthetic_df_STANDARD_TVAE.drop("date", axis=1)
synthetic_df_KAN_TVAE = synthetic_df_KAN_TVAE.drop("date", axis=1)
synthetic_df_hybrid_KAN_TVAE = synthetic_df_hybrid_KAN_TVAE.drop("date", axis=1)

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
X_real = real_df.drop(["Appliances"], axis=1)
y_real = real_df["Appliances"]

X_STANDARD_TVAE = synthetic_df_STANDARD_TVAE.drop(["Appliances"], axis=1)
y_STANDARD_TVAE = synthetic_df_STANDARD_TVAE["Appliances"]

X_KAN_TVAE = synthetic_df_KAN_TVAE.drop(["Appliances"], axis=1)
y_KAN_TVAE = synthetic_df_KAN_TVAE["Appliances"]

X_Hybrid_KAN_TVAE = synthetic_df_hybrid_KAN_TVAE.drop(["Appliances"], axis=1)
y_Hybrid_KAN_TVAE = synthetic_df_hybrid_KAN_TVAE["Appliances"]

# Create a dictionary for the synthetic data and one for the ML models that will be used
synthetic_datasets = {
    "STANDARD TVAE": (X_STANDARD_TVAE, y_STANDARD_TVAE),
    "KAN TVAE": (X_KAN_TVAE, y_KAN_TVAE),
    "Hybrid KAN TVAE": (X_Hybrid_KAN_TVAE, y_Hybrid_KAN_TVAE)
}

models = {
    "XGB": XGBRegressor(colsample_bytree = 0.8, 
                     gamma = 0, learning_rate = 0.1, 
                     max_depth = 5, 
                     n_estimators = 100, 
                     subsample = 1.0, 
                     random_state=1618),
    "RF": RandomForestRegressor(max_depth=40, max_features="sqrt", n_estimators=240),
    "SVR": SVR(C=8, gamma=1, kernel="rbf"),
    "Linear": LinearRegression()
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
real_metrics_df, overall_syn_metrics_df_TVAE, detailed_syn_metrics_TVAE = evaluate_all_models(X_real, y_real, synthetic_datasets, models, test_size=0.2, random_state=1618, repeats=10)

real_metrics_df.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/real_metrics.csv", index=False)
overall_syn_metrics_df_TVAE.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/overall_syn_metrics_TVAE.csv", index=False)
print(real_metrics_df.head())
print(overall_syn_metrics_df_TVAE.head())
print(detailed_syn_metrics_TVAE)"""

# Import the datasets with performances
real_metrics_df = pd.read_csv("TestDatasets/BikeSynthetic/SyntheticPerformanceFromCluster/real_metrics_TVAE.csv")
overall_syn_metrics_df_TVAE = pd.read_csv("TestDatasets/BikeSynthetic/SyntheticPerformanceFromCluster/overall_syn_metrics_TVAE.csv")


# Create diff metrics to store the differences in performance from the original data
# Compute the differences
diff_metrics = overall_syn_metrics_df_TVAE.copy()

# Absolute difference
for metric in ["MAE", "MSE"]:
    diff_metrics[f"Delta_{metric}"] = abs(real_metrics_df.mean()[metric] - overall_syn_metrics_df_TVAE[f"{metric}_avg"])

# Calculate RMSE 
overall_syn_metrics_df_TVAE["RMSE_avg"] = np.sqrt(overall_syn_metrics_df_TVAE["MSE_avg"])
real_rmse = np.sqrt(real_metrics_df.mean()["MSE"])

# Compute the difference is RMSE
diff_metrics["Delta_RMSE"] = abs(real_rmse - overall_syn_metrics_df_TVAE["RMSE_avg"])

# Absolute difference
for metric in ["MAE"]:
    diff_metrics[f"Delta_{metric}"] = abs(real_metrics_df.mean()[metric] - overall_syn_metrics_df_TVAE[f"{metric}_avg"])

# Normalize the scores
real_mae = real_metrics_df.mean()["MAE"]

# Range: (-inf, 1], where 1 means perfect fit with real outputs
diff_metrics["MAE_Score"] = 1 - (diff_metrics["Delta_MAE"] / real_mae)
diff_metrics["RMSE_Score"] = 1 - (diff_metrics["Delta_RMSE"] / real_rmse)

"""# Clamp at zero to keep in range [0,1]
for col in ["MAE_Score","RMSE_Score","R2_Score"]:
    diff_metrics[col] = diff_metrics[col].clip(lower=0)"""


model_names = ["Standard TVAE", "KAN TVAE", "Hybrid KAN TVAE"]
diff_metrics.index = model_names

# Creating a overall score (since now MAE_Score, MSE_Score and R2_Score are in the same range (-inf, 1])
diff_metrics["Overall_Score"] = (diff_metrics[["MAE_Score", "RMSE_Score"]].mean(axis=1)) 
diff_metrics = diff_metrics.sort_values(by="Overall_Score", ascending=False)
print(diff_metrics[["MAE_Score", "RMSE_Score"]])
print(diff_metrics["Overall_Score"])


# Visualize the rank
fig, ax = plt.subplots(1,1,figsize=(12,6))
bars = ax.bar(diff_metrics.index, diff_metrics["Overall_Score"], color="green", edgecolor="black")

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
ax.set_xticklabels(diff_metrics.index, rotation=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


