# Removing-Outliners-
✅
The script's output directory is C:\Users\aaa\Desktop\ekxin.
It creates Removed_Points and Curved_Points subdirectories there.
Input data is a hardcoded string of tab-separated X and Y values.
It evaluates linear, polynomial (orders 2-4), exponential, logarithmic, and power regressions.
The script removes one outlier at a time, based on the largest residual from the best-fit model.
Its goal is to meet R-squared targets (0.9, 0.85, 0.8, 0.75, 0.7).
A minimum of 30 data points must remain for a valid result.
For each achieved target, CSVs of removed and remaining points are saved.
Finally, a summary of target achievements and the overall best curve is printed.
