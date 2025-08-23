import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots

def gen_fake_data(N, M):
    """
    Generate N samples, M binary features, and binary labels.Labels are random just for runtime testing.
    """
    p=0.6
    X =pd.DataFrame(np.random.randint(0, 2, size=(N, M)),
                     columns=[f"x{i}" for i in range(M)])
    y =pd.Series(np.random.randint(0, 2, size=N))
    noise= np.random.rand(N)
    X['x0'] =np.where(noise<p,y,1-y)
    return X,y

def measure_time(N_values, M_values):
    results = []

    for criterion in ["information_gain", "gini_index"]:
        print(f"############ Criterion: {criterion} ###############\n")
        for N in N_values:
            for M in M_values:
                X, y = gen_fake_data(N, M)
                fit_times = []
                pred_times = []

                for _ in range(num_average_time):
                    clf = DecisionTree(criterion=criterion)

                    # Measure training time
                    start_fit = time.time()
                    clf.fit(X, y)
                    end_fit = time.time()
                    fit_times.append(end_fit - start_fit)

                    # Measure prediction time
                    X_test, _ = gen_fake_data(N, M)
                    start_pred = time.time()
                    clf.predict(X_test)
                    end_pred = time.time()
                    pred_times.append(end_pred - start_pred)

                # store averages
                results.append({
                    "criterion": criterion,
                    "N": N,
                    "M": M,
                    "fit_time_mean": np.mean(fit_times),
                    "fit_time_std": np.std(fit_times),
                    "pred_time_mean": np.mean(pred_times),
                    "pred_time_std": np.std(pred_times)
                })

    return pd.DataFrame(results)

def plot_results(df):
    for criterion in df["criterion"].unique():
        subdf = df[df["criterion"] == criterion]
        
        plt.figure(figsize=(12,5))
        for M in subdf["M"].unique():
            df_m = subdf[subdf["M"] == M]
            plt.errorbar(df_m["N"], df_m["fit_time_mean"], 
                         yerr=df_m["fit_time_std"], label=f"Fit M={M}")
        plt.title(f"Training Time vs N ({criterion})")
        plt.xlabel("N (samples)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12,5))
        for M in subdf["M"].unique():
            df_m = subdf[subdf["M"] == M]
            plt.errorbar(df_m["N"], df_m["pred_time_mean"], 
                         yerr=df_m["pred_time_std"], label=f"Predict M={M}")
        plt.title(f"Prediction Time vs N ({criterion})")
        plt.xlabel("N (samples)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.show()


N_values = [50, 100, 200, 300, 400] 
M_values = [5, 10, 20]

df_results = measure_time(N_values, M_values)
print("Experiments done, plotting the results./n")
plot_results(df_results)

