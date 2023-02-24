# ======================================================================== #
# Benjamin Ward
# Julio Lemmus
#
# CIS 492 - Machine Learning
# Dr. Qin Lin
# 12 December, 2022
# 
# PCA Dimension Reduction + K Means Clustering for College Type Classification
#
# 777 rows of 19 feaures (-1 because labeled): 
#   Name, Type, Apps, Accept, Enroll, Top10perc, Top25perc, F.Undergrad, P.undergrad, Outstate,
#   Room.Board, Books, Personal, PhD, Terminal, S.F.Ratio, perc.alumni, Expend, Grad.Rate
# 
#     Training set: 85% (660 rows)
# Cross Validation: 15% (117 rows)
#      Testing set: None because it'd be the same as cross validation evaluation.
# ======================================================================== #

from os import getcwd

import numpy as np
import pandas as pd
import seaborn as sns


# =========================== Function Definitions ======================= #
def debugprint(text: str):
    """
    Prints text if the debug variable is set to True.
    """
    if(DO_DEBUG):
        print(text)

def calc_precision(true_pos: int, false_pos: int):
    """
    Returns the precision given the number of true positives and false positives.
    """

    precision = true_pos / (true_pos + false_pos)
    return precision

def calc_recall(true_pos: int, false_neg: int):
    """
    Returns the recall given the number of true positives and false negatives.
    """

    recall = true_pos / (true_pos + false_neg)
    return recall

def calc_f1_score(precision: float, recall: float):
    """
    Returns the F1 score (float) given a precision and recall.
    """

    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score
# ======================================================================== #





# =========================== Initial Variables ========================== #
DO_DEBUG = False

basepath = getcwd() + '\\'
path_data = basepath + 'College_Data.csv'
path_export = basepath + 'Exported DataFrames\\'

df_raw = pd.read_csv(path_data)
# ======================================================================== #





# ========================= PCA Dimension Reduction ====================== #
# Find the means of each feature in the raw data
feature_means = []
for col in list(df_raw.columns):
    if col == 'Unnamed: 0' or col == 'Private': continue
    col_mean = df_raw[col].astype(float).mean()
    feature_means.append(col_mean)

# Remove the name of the university and its type from the dataframe
df_colleges = df_raw[['Unnamed: 0']]
df_all_truths = df_raw[['Private']]
df_raw = df_raw[df_raw.columns.difference(['Unnamed: 0', 'Private'])]

# Center the data points by subtracting each column by the mean for that column
mtx_raw = df_raw.astype(float).to_numpy()
mtx_centered = np.subtract(mtx_raw, feature_means)

# Tranpose so we have a shape of eigen vecs that map to our features
covariance = np.cov(mtx_centered.T)

# Calculate the eigen values and vectors of the covariance
eig_vals, eig_vecs = np.linalg.eig(covariance)

# Each eigenval contributes to the total percentage of data captured
# Create a dataframe of the iterative sum of total % of data captured
running_sum = 0.0
percentage_list = [0]
cumulative_list = [0]
eigval_sum = sum(eig_vals)

for v in eig_vals:
    percentage = (v/sum(eig_vals)) *100     # Find the percentage
    percentage_list.append(percentage)      # Append the percentage to a discrete % list
    running_sum += percentage               # Add the current percentage to the running sum   
    cumulative_list.append(running_sum)     # Append the running cumulative % to a list 

# Export the percentages to a CSV for review
if(DO_DEBUG):
    df_percent = pd.DataFrame(percentage_list)
    df_percent.to_csv(path_export + 'df_percent.csv')
    df_cumulative = pd.DataFrame(cumulative_list)
    df_cumulative.to_csv(path_export + 'df_cumulative.csv')

print('Cumulative Percentage of Data Variance Captured in the the first 6 features:\n', cumulative_list[7], '%\n')
print(f'Discrete Percentages of Data Variance Captured in the the first 6 features:')
for i in range(0,7):
    print(f'{i}) ', percentage_list[i])
print()

# Use the first 6 eigenvectors to reduce the data's dimensions
mtx_projected = np.matmul(mtx_centered,eig_vecs[0:6].T)

# Seperate the Training and Cross Validation data
mtx_training = mtx_projected[:660]
mtx_cv = mtx_projected[661:]
# ======================================================================== #





# ============================ K-Means Clustering ======================== #
# Create two centroids, randomaly initialized
centroid1 = np.random.rand(1,6)*10
centroid2 = np.random.rand(1,6)*10

# Store the initial centroid locations
initc1 = centroid1
initc2 = centroid2

df_centroid1 = []
df_centroid2 = []

# Create an array that represents the prediction for each college, initialized to -1
# 1 = closest to centroid 1
# 2 = closest to centroid 2
predictions = list(np.zeros(660)-1)

# Repeat until convergence:
CONVERGED = False
count = 0
while(not CONVERGED):
    # Assign each point to their closest centroid cluster
    for row in range(0, len(mtx_training)):
        d_cent1 = np.linalg.norm(mtx_training[row] - centroid1)
        d_cent2 = np.linalg.norm(mtx_training[row] - centroid2)
        
        predictions[row] = 1 if d_cent1 < d_cent2 else 2

    # Move the clusters centroids by summing all the data points associated with each cluster
    # and then divide that number by the number of data points associated with that cluster by...
    
    # 1. Get the indexes of each data point and its associated cluster
    c1_indexes = []
    c2_indexes = []
    for row in range(0, len(predictions)):
        if predictions[row] == 1: c1_indexes.append(row)
        else: c2_indexes.append(row)

    # 2. Count the number of datapoints associated with each cluster
    count_c1 = len(c1_indexes)
    count_c2 = len(c2_indexes)
    debugprint(f'\tNumber elements in Centroid 1: {count_c1}')
    debugprint(f'\tNumber elements in Centroid 2: {count_c2}')

    # 3. Sum each cluster's assoicated data points
    sum_c1 = np.zeros((1,6))
    sum_c2 = np.zeros((1,6))

    for dataindex in c1_indexes:
        sum_c1 += mtx_training[dataindex]

    for dataindex in c2_indexes:
        sum_c2 += mtx_training[dataindex]

    # 4. Calculate the new cluster by taking the mean of all its associated points
    new_c1 = sum_c1 / count_c1
    new_c2 = sum_c2 / count_c2

    # Debug prints
    debugprint(f'Old Centroid 1: {centroid1}')
    debugprint(f'New Centroid 1: {new_c1}\n')
    debugprint(f'Old Centroid 2: {centroid2}')
    debugprint(f'New Centroid 2: {new_c2}')

    # Append the new centroid measurements to their associated pandas dataframe
    df_centroid1.append(new_c1)
    df_centroid2.append(new_c2)

    count+=1
    print(f'Iteration {count} Completed\n')

    # Check if the centroids have converged
    if (np.linalg.norm(centroid1 - new_c1) <= 0.000001) and (np.linalg.norm(centroid2 - new_c2) <= 0.000001):
        print('The Centroids Have Converged')
        CONVERGED = True

    centroid1 = new_c1
    centroid2 = new_c2

df_centroid1 = pd.DataFrame([df_centroid1]).T
df_centroid2 = pd.DataFrame([df_centroid2]).T

if(DO_DEBUG):
    df_centroid1.to_csv(path_export + 'df_centroid1.csv')
    df_centroid2.to_csv(path_export + 'df_centroid2.csv')
# ======================================================================== #





# ========================= Performance Evaluation ======================= #
# Step 1. For each row of data, assign data to a cluster and make prediction
# Public == 1
# Private == 2

cv_predictions = list(np.zeros(117)-1)
for row in range(0, len(mtx_cv)):
    d_cent1 = np.linalg.norm(mtx_cv[row] - centroid1)
    d_cent2 = np.linalg.norm(mtx_cv[row] - centroid2)
    cv_predictions[row] = 1 if d_cent1 < d_cent2 else 2

# Step 2. Get the indexes of each data point and its associated cluster (optional)
#cv_c1_indexes = []
#cv_c2_indexes = []
#for row in range(0, len(cv_predictions)):
#    if cv_predictions[row] == 1: cv_c1_indexes.append(row)
#    else: cv_c2_indexes.append(row)

# Step 3. Map the ground truths words to numbers (data 660:777)
cv_truths = df_all_truths[660:]
cv_truths = cv_truths['Private'].map({'Yes': 2, 'No': 1})
cv_truths = cv_truths.to_list()

# Step 4. Evalute clusters' True and False public and private classifications
# Yes = Private = 2
# No = Public = 1
actual_private = cv_truths.count(2)
actual_public = cv_truths.count(1)

predicted_private = cv_predictions.count(2)
predicted_public = cv_predictions.count(1)

true_privates = 0
false_privates = 0

true_publics = 0
false_publics = 0

# Count the number of matching and mismatching classifications
for i in range(0, len(cv_predictions)):
    if cv_predictions[i] == 2:                          # If private was predicted:
        if cv_truths[i] == 2:  true_privates += 1       # If GT is private, increment true privates
        else:                   false_privates += 1     # If GT is public, increment false privates

    if cv_predictions[i] == 1:                          # If public was predicted:
        if cv_truths[i] == 1:  true_publics += 1        # If GT is public, increment true publics
        else:                   false_publics += 1      # If GT is public, increment false privates

# Expand on performance metrics
# The rand index == correct cluster assignments / total # of assignments
precision = calc_precision(true_privates, false_privates)
recall = calc_recall(true_privates, false_publics)
f1 = calc_f1_score(precision, recall)
rand_index = (true_privates + true_publics) / len(cv_truths)





print(f'\n\n====================================================================')
print('Clustering Cross Validation Results                                      ')
print('====================================================================     ')
print(f'        Predicted Private Schools: {predicted_private}                  ')
print(f'           Actual Private Schools: {actual_private}                   \n')
print(f'        Predicted Public Schools: {predicted_public}                    ')
print(f'           Actual Public Schools: {actual_public}                     \n')
print(f'  True Private School Predictions: {true_privates}                      ')
print(f' False Private School Predictions: {false_privates}                 \n\n')
print(f'  True Public School Predictions: {true_publics}                        ')
print(f' False Public School Predictions: {false_publics}                   \n\n')
print(f'                                                                        ')
print(f'     Rand Index: {rand_index}                                           ')
print(f'      Precision: {precision}                                            ')
print(f'         Recall: {recall}                                               ')
print(f'      F-1 Score: {f1}                                           \n\n\n\n')
print(f'Centroid 1 Original Location: {initc1}                                  ')
print(f'Centroid 1 Original Location: {initc2}                                  ')
print(f'====================================================================\n\n')

print('\n====================================================================')
print('Script Finished Executing Successfully!')
print('====================================================================\n')

# ======================================================================== #