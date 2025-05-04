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

# Import the data 
real_df = pd.read_csv("TestDatasets/energydata_complete.csv")
synthetic_df_STANDARD_CTGAN = pd.read_csv("TestDatasets/EnergySynthetic/synthetic_df_STANDARD_CTGAN.csv")
synthetic_df_KAN_CTGAN = pd.read_csv("TestDatasets/EnergySynthetic/synthetic_df_KAN_CTGAN.csv")

# Evaluate the Predictive Efficacy
real_df = real_df.drop("date", axis=1)
synthetic_df_STANDARD_CTGAN = synthetic_df_STANDARD_CTGAN.drop("date", axis=1)
synthetic_df_KAN_CTGAN = synthetic_df_KAN_CTGAN.drop("date", axis=1)

# Split the real dataset in two random subsets (TO TEST THE FUNCTION)
real_data_part_1, real_data_part_2 = train_test_split(real_df, test_size=0.5, random_state=1618)

# Evaluate the two parts on the statistical function
sim_score_test = overall_similarity(real_data_part_1, real_data_part_2)
print(f"Similarity score: {sim_score_test}")

sim_score_STANDARD_CTGAN = overall_similarity(real_df, synthetic_df_STANDARD_CTGAN)
print("Similarity between real data and synthetic data with Standard CTGAN: ", sim_score_STANDARD_CTGAN)

sim_score_KAN_CTGAN = overall_similarity(real_df, synthetic_df_KAN_CTGAN)
print("Similarity between real data and synthetic data with KAN CTGAN: ", sim_score_KAN_CTGAN)

# Evaluate the ML efficacy
# Divide all dataframes in training and targets
X_real = real_df.drop(["Appliances"], axis=1)
y_real = real_df["Appliances"]

X_STANDARD_CTGAN = synthetic_df_STANDARD_CTGAN.drop(["Appliances"], axis=1)
y_STANDARD_CTGAN = synthetic_df_STANDARD_CTGAN["Appliances"]

X_KAN_CTGAN = synthetic_df_KAN_CTGAN.drop(["Appliances"], axis=1)
y_KAN_CTGAN = synthetic_df_KAN_CTGAN["Appliances"]

# Create a dictionary for the synthetic data and one for the ML models that will be used
synthetic_datasets = {
    "STANDARD CTGAN": (X_STANDARD_CTGAN, y_STANDARD_CTGAN),
    "KAN CTGAN": (X_KAN_CTGAN, y_KAN_CTGAN),
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
}

"""# Evaluate function
# THIS IS TO PROVE THAT IF THE TRAINING DATA ARE VERY SIMILAR (BOTH REAL) THEN THE PERFORMANCE WILL BE ALMOST IDENTICAL
real_metric_1, real_metric_2, _ = evaluate_all_models(X_real, y_real, real_data_dict, models, test_size=0.2, random_state=1618, repeats=5)
real_metric_2.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/TEST_EQUAL_TO_REAL.csv", index=False)
print(real_metric_1.mean()[["MAE", "MSE", "R2"]])
print(real_metric_2.head())"""

"""# Create the metrics datasets
real_metrics_df, overall_syn_metrics_df, detailed_syn_metrics = evaluate_all_models(X_real, y_real, synthetic_datasets, models, test_size=0.2, random_state=1618, repeats=10)

real_metrics_df.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/real_metrics.csv", index=False)
overall_syn_metrics_df.to_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/overall_syn_metrics.csv", index=False)
print(real_metrics_df.head())
print(overall_syn_metrics_df.head())
print(detailed_syn_metrics)"""

# Import the datasets with performances
real_metrics_df = pd.read_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/real_metrics.csv")
overall_syn_metrics_df = pd.read_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/overall_syn_metrics.csv")
TEST = pd.read_csv("TestDatasets/EnergySynthetic/SyntheticPerformance/TEST_EQUAL_TO_REAL.csv")

# Create diff metrics to store the differences in performance from the original data
# Compute the differences
diff_metrics = overall_syn_metrics_df.copy()
diff_metrics_TEST = TEST.copy()

# Absolute difference
for metric in ["MAE", "MSE", "R2"]:
    diff_metrics[f"Delta_{metric}"] = abs(real_metrics_df.mean()[metric] - overall_syn_metrics_df[f"{metric}_avg"])

# TEST
for metric in ["MAE", "MSE", "R2"]:
    diff_metrics_TEST[f"Delta_TEST_{metric}"] = abs(real_metrics_df.mean()[metric] - TEST[f"{metric}_avg"])

# Percentages, relative difference
for metric in ["MAE", "MSE"]:
    diff_metrics[f"Real_Delta_{metric} (%)"] = (diff_metrics[f"Delta_{metric}"] / (real_metrics_df.mean()[metric])) * 100

# TEST 
for metric in ["MAE", "MSE"]:
    diff_metrics_TEST[f"Real_Delta_TEST_{metric} (%)"] = (diff_metrics_TEST[f"Delta_TEST_{metric}"] / (real_metrics_df.mean()[metric])) * 100

# For R2 the difference is simplier since it's already a percentage
diff_metrics["Real_Delta_R2 (%)"] = (diff_metrics["Delta_R2"] / (real_metrics_df.mean()["R2"])) * 100 # CHECK HERE

# TEST
diff_metrics_TEST["Real_Delta_TEST_R2 (%)"] = (diff_metrics_TEST["Delta_TEST_R2"] / (real_metrics_df.mean()["R2"])) * 100 

model_names = ["Standard CTGAN", "KAN CTGAN"]
diff_metrics.index = model_names

print(diff_metrics)

# Calculate the scores
diff_metrics["MAE_Score"] = 1 - (diff_metrics["Delta_MAE"] / real_metrics_df.mean()["MAE"]) 
diff_metrics["MSE_Score"] = 1 - (diff_metrics["Delta_MSE"] / real_metrics_df.mean()["MSE"])
diff_metrics["R2_Score"] = 1 - (diff_metrics["Delta_R2"] / real_metrics_df.mean()["R2"])

# TEST
diff_metrics_TEST["MAE_Score"] = 1 - (diff_metrics_TEST["Delta_TEST_MAE"] / real_metrics_df.mean()["MAE"])
diff_metrics_TEST["MSE_Score"] = 1 - (diff_metrics_TEST["Delta_TEST_MSE"] / real_metrics_df.mean()["MSE"])
diff_metrics_TEST["R2_Score"] = 1 - (diff_metrics_TEST["Delta_TEST_R2"] / real_metrics_df.mean()["R2"])
print(diff_metrics_TEST)


# Creating a overall score (since now MAE_Score, MSE_Score and R2_Score are in the same range (-inf, 1])
diff_metrics["Overall_Score"] = (diff_metrics[["MAE_Score", "MSE_Score", "R2_Score"]].mean(axis=1))
diff_metrics = diff_metrics.sort_values(by="Overall_Score", ascending=False)
print(diff_metrics[["MAE_Score", "MSE_Score", "R2_Score"]])
print(diff_metrics["Overall_Score"])

# TEST 
diff_metrics_TEST["Overall_Score"] = (diff_metrics_TEST[["MAE_Score", "MSE_Score", "R2_Score"]].mean(axis=1))
diff_metrics_TEST = diff_metrics_TEST.sort_values(by="Overall_Score")
print(diff_metrics_TEST["Overall_Score"])

# Add the test to diff_metrics
test_overall_score = diff_metrics_TEST["Overall_Score"].iloc[0] 

# Append a new row to diff_metrics with the label "TEST" and the overall score value.
new_row = pd.DataFrame({"Overall_Score": [test_overall_score]}, index=["TEST (Models trained with real data)"])
diff_metrics = pd.concat([diff_metrics, new_row])

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


