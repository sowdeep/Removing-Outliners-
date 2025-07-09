import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
import os
from pathlib import Path

# --- Configuration for Output Files ---
# Set the desired base output directory as requested
base_output_dir = Path(r"C:\Users\aaa\Desktop\ekxin")

# Define the subdirectories for removed and curved points
removed_points_dir = base_output_dir / "Removed_Points"
curved_points_dir = base_output_dir / "Curved_Points"

# Ensure both output directories exist
try:
    removed_points_dir.mkdir(parents=True, exist_ok=True)
    curved_points_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directories ensured:\n  {removed_points_dir}\n  {curved_points_dir}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not create output directories. "
          f"Please check permissions for '{base_output_dir}'. Error: {e}")
    exit() # Exit the script if directories can't be created.


# Define the R-squared targets you want to track
TARGET_R2_VALUES = [0.9, 0.85, 0.8, 0.75, 0.7]
TARGET_R2_VALUES.sort(reverse=True) # Sort descending for logical processing and summary

# Define the MINIMUM number of points that must remain for a result to be valid
# This addresses "don't remove all sets of data" and "maximize the data values"
MIN_REMAINING_POINTS_FOR_VALID_RESULT = 30 # As discussed, aiming to keep at least 30 points

# Data provided by the user
data = """
783	3.204698446
1352	2.57133835
1292	2.287811317
1879	2.473894717
197	1.896698444
1455	3.955961316
632	1.97231132
149	2.151841649
159	2.511606793
184	2.443047077
1180	2.615231653
280	2.956030312
1017	1.483334751
2727	1.961600587
1090	2.174503503
2113	1.897060837
2017	1.236840558
953	2.587933553
243	1.767691526
129	1.588525915
590	1.853895599
166	2.041286602
161	1.826000114
1568	2.375872124
1314	2.310926431
835	1.54763614
1807	3.535095239
2363	1.9835602
2989	2.06353285
1822	2.146780432
2422	2.716423919
2993	3.193869316
2098	1.164750317
2288	3.234169768
1913	3.485530492
685	2.234030006
683	2.067957081
725	2.383432193
1557	1.354677227
895	1.417910535
1412	2.8861567
2741	0.661995483
964	3.621339134
1161	2.885106844
3465	0.429059647
1642	3.369892523
900	3.454242951
1722	3.52309143
2872	3.21936939
2742	4.684651995
673	2.86888841
494	2.184881237
180	2.714316933
237	3.528376449
112	2.170743582
167	2.719482148
106	1.642143084
1353	2.819885417
95	1.728789839
110	2.203263645
871	3.881529703
1881	2.909354169610627
3068	6.35309397329915
2490	2.3414096667249615
3671	0.9256740320814937
3705	0.4501285867196866
3620	0.3834204679939064
3610	0.7293781274805543
3200	0.8581777086561275
1970	265.8179299781152
2330	4.406171678550766
3886	1.108898646287201
1801	2.357295015535291
3650	2.229815095368024
820	3.938070945391346
724	2.71401453943162
617	2.575471645118872
1617	6.990123579120465
1738	9.281793575369415
2680	2.4658394346627905
3556	1.1774948379443797
1960	5.676805211213144
1996	7.923714024798616
3300	1.6124247662495856
4100	59.49080958700559
189	2.636678992816091
177	2.6228040034417925
310	2.7558161560015915
137	2.1554289383729532
87	1.803436727454001
152	2.0742237994574775
1535	2.1063726918156016
518	4.515816944369367
630	3.5355414711241298
909	2.351991936542037
1940	3.5825783335298054
1877	4.159605949273412
1536	2.6851732245698066
1417	2.7516584658801073
271	2.0705327582609256
76	1.6363195349309945
127	2.13765010156672
90	1.7800211824813903
2064	4.473038499010169
2383	3.2042908412524707
2039	6.215371825234093
2642	4.037744099
1945	3.599341892
1731	3.212411893
1529	1.617339132
63	1.695182239
1612	2.332542527
68	2.034115191
1614	3.444965911
71	2.286084302
1494	6.94859593
1277	2.301910414
1720	1.746878863
1457	1.240122411
146	3.519908789
310	2.482605523
1595	1.936101642
119	2.161264288
122	1.972952009
1654	2.158340634
530	2.443856712
168	2.190713746

"""

# Parse the data
lines = data.strip().split('\n')
parsed_data = []
for line in lines:
    x_str, y_str = line.split('\t')
    parsed_data.append({'X': float(x_str), 'Y': float(y_str)})

original_df = pd.DataFrame(parsed_data)
# Add a unique original index to help track points after removal
original_df['original_idx'] = original_df.index


# Define functions for different regression types
def linear_func(x, a, b):
    return a * x + b

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def logarithmic_func(x, a, b):
    return a * np.log(x) + b

def power_func(x, a, b):
    return a * np.power(x, b)

# Function to evaluate R-squared for all models on a given DataFrame
def evaluate_models(df_current):
    r2_scores = {}
    predictions = {}
    coefs = {}

    if len(df_current) < 2: # Need at least 2 points for most regressions
        return {}, {}, {}

    # Linear Regression
    try:
        lin_reg = LinearRegression()
        lin_reg.fit(df_current[['X']], df_current['Y'])
        y_pred_linear = lin_reg.predict(df_current[['X']])
        r2_scores['Linear'] = r2_score(df_current['Y'], y_pred_linear)
        predictions['Linear'] = y_pred_linear
        coefs['Linear'] = {'a': lin_reg.coef_[0], 'b': lin_reg.intercept_}
    except Exception:
        r2_scores['Linear'] = -np.inf

    # Polynomial Regression (Order 2, 3, 4)
    for order in range(2, 5):
        try:
            poly_features = PolynomialFeatures(degree=order)
            X_poly = poly_features.fit_transform(df_current[['X']])
            poly_reg = LinearRegression()
            poly_reg.fit(X_poly, df_current['Y'])
            y_pred_poly = poly_reg.predict(X_poly)
            r2_scores[f'Polynomial (Order {order})'] = r2_score(df_current['Y'], y_pred_poly)
            predictions[f'Polynomial (Order {order})'] = y_pred_poly
            coefs[f'Polynomial (Order {order})'] = {'coefs': poly_reg.coef_, 'intercept': poly_reg.intercept_}
        except Exception:
            r2_scores[f'Polynomial (Order {order})'] = -np.inf

    # Exponential Regression
    df_exp = df_current[df_current['Y'] > 0]
    if len(df_exp) >= 2:
        try:
            popt_exp, pcov_exp = curve_fit(exponential_func, df_exp['X'], df_exp['Y'], p0=[1, 0.001], maxfev=5000)
            y_pred_exp = exponential_func(df_current['X'], *popt_exp)
            r2_scores['Exponential'] = r2_score(df_current['Y'], y_pred_exp)
            predictions['Exponential'] = y_pred_exp
            coefs['Exponential'] = {'a': popt_exp[0], 'b': popt_exp[1]}
        except Exception:
            r2_scores['Exponential'] = -np.inf
    else:
        r2_scores['Exponential'] = -np.inf

    # Logarithmic Regression
    df_log = df_current[df_current['X'] > 0]
    if len(df_log) >= 2:
        try:
            popt_log, pcov_log = curve_fit(logarithmic_func, df_log['X'], df_log['Y'], p0=[1, 1], maxfev=5000)
            y_pred_log = logarithmic_func(df_current['X'], *popt_log)
            r2_scores['Logarithmic'] = r2_score(df_current['Y'], y_pred_log)
            predictions['Logarithmic'] = y_pred_log
            coefs['Logarithmic'] = {'a': popt_log[0], 'b': popt_log[1]}
        except Exception:
            r2_scores['Logarithmic'] = -np.inf
    else:
        r2_scores['Logarithmic'] = -np.inf

    # Power Regression
    df_power = df_current[(df_current['X'] > 0) & (df_current['Y'] > 0)]
    if len(df_power) >= 2:
        try:
            popt_power, pcov_power = curve_fit(power_func, df_power['X'], df_power['Y'], p0=[1, 0.1], maxfev=5000)
            y_pred_power = power_func(df_current['X'], *popt_power)
            r2_scores['Power'] = r2_score(df_current['Y'], y_pred_power)
            predictions['Power'] = y_pred_power
            coefs['Power'] = {'a': popt_power[0], 'b': popt_power[1]}
        except Exception:
            r2_scores['Power'] = -np.inf
    else:
        r2_scores['Power'] = -np.inf

    return r2_scores, predictions, coefs

# Dictionary to store the best results for each target R-squared
# Key: target_r2 (e.g., 0.9), Value: {'r2': float, 'model': str, 'removed_count': int, 'removed_points': list, 'remaining_points_df': DataFrame, 'coefficients': dict}
results_by_target_r2 = {target: {'r2': -1.0, 'model': None, 'removed_count': -1, 'removed_points': [], 'remaining_points_df': None, 'coefficients': None} for target in TARGET_R2_VALUES}

current_df = original_df.copy()
initial_num_points = len(original_df)
# Set a generous upper limit for total removals to ensure we can hit high R2s if possible
# Max iteration count is initial_num_points - MIN_REMAINING_POINTS_FOR_VALID_RESULT
max_iterations = initial_num_points - MIN_REMAINING_POINTS_FOR_VALID_RESULT
if max_iterations < 0:
    max_iterations = 0 # Ensures loop doesn't run if dataset is already too small

print(f"Starting with {initial_num_points} points.")
print(f"Searching for R-squared >= {TARGET_R2_VALUES} while keeping at least {MIN_REMAINING_POINTS_FOR_VALID_RESULT} points...")

# List to keep track of points removed in the current sequence for the iteration
cumulative_points_removed_for_iter = []

# Iterate, removing one point at a time, until minimum remaining points or max iterations
# i represents the number of points removed so far
for i in range(max_iterations + 1): # +1 to include the initial state (0 removals)
    if len(current_df) < MIN_REMAINING_POINTS_FOR_VALID_RESULT:
        print(f"Stopped because only {len(current_df)} points remain, which is below the minimum of {MIN_REMAINING_POINTS_FOR_VALID_RESULT} for a valid result.")
        break
    if len(current_df) < 5 and i > 0: # Still ensure enough for basic regression models
        print(f"Stopped because only {len(current_df)} points remain, too few for robust regression models.")
        break


    r2_scores, predictions, current_coefs = evaluate_models(current_df)

    # Find the best model for this iteration (current R-squared with 'i' points removed)
    current_iteration_best_model = None
    current_iteration_best_r2 = -np.inf
    for model, r2 in r2_scores.items():
        if r2 > current_iteration_best_r2:
            current_iteration_best_r2 = r2
            current_iteration_best_model = model
    
    if current_iteration_best_model is None or current_iteration_best_r2 == -np.inf:
        print(f"No valid regression could be performed with {len(current_df)} points at step {i}.")
        break

    # --- Check against all target R-squared values ---
    for target_r2 in TARGET_R2_VALUES:
        # Only record if the target is met AND we have enough remaining points
        if current_iteration_best_r2 >= target_r2 and len(current_df) >= MIN_REMAINING_POINTS_FOR_VALID_RESULT:
            # IMPORTANT: Update only if this is the FIRST time we hit this target, OR if we hit it with FEWER removals
            if results_by_target_r2[target_r2]['removed_count'] == -1 or i < results_by_target_r2[target_r2]['removed_count']:
                results_by_target_r2[target_r2]['r2'] = current_iteration_best_r2
                results_by_target_r2[target_r2]['model'] = current_iteration_best_model
                results_by_target_r2[target_r2]['removed_count'] = i
                results_by_target_r2[target_r2]['removed_points'] = list(cumulative_points_removed_for_iter)
                results_by_target_r2[target_r2]['remaining_points_df'] = current_df.copy()
                results_by_target_r2[target_r2]['coefficients'] = current_coefs.get(current_iteration_best_model)
                print(f"  Target R-squared {target_r2} MET/IMPROVED with {i} points removed ({len(current_df)} remaining). Best R2: {current_iteration_best_r2:.4f} ({current_iteration_best_model})")


    # --- Prepare for the next iteration (remove the worst point) ---
    # If this is the very last allowed iteration, or we've hit max_iterations, don't remove more points
    if i == max_iterations:
        print(f"Maximum iterations ({max_iterations} points removed) reached for data retention constraint.")
        break

    # Identify the point with the largest residual for the current best model for removal in next iteration
    if current_iteration_best_model in predictions:
        y_pred = predictions[current_iteration_best_model]
        residuals_aligned = pd.Series(y_pred, index=current_df.index)
        residuals = np.abs(current_df['Y'] - residuals_aligned)

        if not residuals.empty:
            largest_residual_index_in_current_df = residuals.idxmax()
            
            original_point_idx = current_df.loc[largest_residual_index_in_current_df, 'original_idx']
            
            cumulative_points_removed_for_iter.append(
                original_df[original_df['original_idx'] == original_point_idx].iloc[0].drop('original_idx').to_dict()
            )
            
            current_df = current_df.drop(largest_residual_index_in_current_df).reset_index(drop=True)
        else:
            print(f"Could not calculate residuals for removal step {i+1}. Empty residuals for {current_iteration_best_model}.")
            break
    else:
        print(f"No valid prediction for best model '{current_iteration_best_model}' at step {i+1}. Cannot identify outlier for removal.")
        break

# --- Final Output and CSV Saving for each Achieved R-squared Target ---
print("\n--- Summary of Achieved R-squared Targets (keeping at least 30 points) ---")
overall_highest_r2_found = -1.0
overall_highest_r2_details = {} # Details of the single highest R2 found under constraints

for target_r2 in sorted(TARGET_R2_VALUES, reverse=True):
    result = results_by_target_r2[target_r2]
    if result['r2'] >= target_r2: # If the target was met
        print(f"\nTarget R-squared {target_r2} was ACHIEVED:")
        print(f"  Best R-squared: {result['r2']:.4f}")
        print(f"  Model: {result['model']}")
        print(f"  Points removed: {result['removed_count']}")
        print(f"  Points remaining: {len(result['remaining_points_df'])}")

        # Save removed points CSV
        try:
            removed_df = pd.DataFrame(result['removed_points'])
            if not removed_df.empty or result['removed_count'] == 0:
                r2_filename_part = str(target_r2).replace(".", "_")
                removed_file_path = removed_points_dir / f'removed_r2_{r2_filename_part}.csv'
                removed_df.to_csv(removed_file_path, index=False)
                print(f"  Removed points saved to '{removed_file_path}'.")
            else:
                print(f"  No points were specifically removed to achieve R2 {target_r2} (empty removed list).")
        except Exception as e:
            print(f"  Error saving removed points for R2 {target_r2}: {e}")

        # Save remaining points for curve CSV
        if result['remaining_points_df'] is not None and not result['remaining_points_df'].empty:
            try:
                points_for_curve_df = result['remaining_points_df'].drop(columns=['original_idx'], errors='ignore')
                r2_filename_part = str(target_r2).replace(".", "_")
                curve_file_path = curved_points_dir / f'curve_r2_{r2_filename_part}.csv'
                points_for_curve_df.to_csv(curve_file_path, index=False)
                print(f"  Points for curve saved to '{curve_file_path}'.")
            except Exception as e:
                print(f"  Error saving points for curve for R2 {target_r2}: {e}")
        
        # Track the highest R2 found overall under the constraints
        if result['r2'] > overall_highest_r2_found:
            overall_highest_r2_found = result['r2']
            overall_highest_r2_details = {
                'model': result['model'],
                'removed_count': result['removed_count'],
                'points_remaining': len(result['remaining_points_df']),
                'target_achieved': target_r2
            }
    else:
        print(f"\nTarget R-squared {target_r2} was NOT achieved (or not under the {MIN_REMAINING_POINTS_FOR_VALID_RESULT} points remaining constraint).")
        print(f"  Highest R-squared reached for this target was: {result['r2']:.4f}")

# Final summary of the single highest R-squared achieved under all constraints
print("\n--- Overall Best Single Curve Outcome (Highest R-squared under constraints) ---")
if overall_highest_r2_found > -1:
    print(f"Overall highest R-squared found: {overall_highest_r2_found:.4f}")
    print(f"Achieved with Model: {overall_highest_r2_details.get('model', 'N/A')}")
    print(f"Points removed: {overall_highest_r2_details.get('removed_count', 'N/A')}")
    print(f"Points remaining: {overall_highest_r2_details.get('points_remaining', 'N/A')}")
    if 'target_achieved' in overall_highest_r2_details:
        print(f"This met the specific target R2 of: {overall_highest_r2_details['target_achieved']}")
    else:
        print(f"This was the highest R2, but did not meet any of the specified targets ({TARGET_R2_VALUES}).")
else:
    print(f"No valid R-squared (above -1.0) could be determined under the constraints (at least {MIN_REMAINING_POINTS_FOR_VALID_RESULT} points remaining).")