# ======================================================================== #
# Benjamin Ward
#
# CIS 492 - Machine Learning
# Dr. Qin Lin
# 25 November, 2022
# 
# PCA Dimension Reduction
#
# 1567 rows of 590 feaures: 
# You can find the dataset and more information at http://archive.ics.uci.edu/ml/machine-learning-databases/secom/.
#
# Task : Implement PCA for dimension reduction. 
# If we want to capture 99% of variance, how many principle components we need? Complete the table [Principle component number, % variance, % cumulative variance].
#
# Hint on data preprocessing: there are lots of NAN in the dataset. 
# For each of them, replace it using its corresponding feature’s mean value (mean calculated using non-NAN data).
# ======================================================================== #

from os import getcwd

import numpy as np
import pandas as pd

DO_DEBUG = True

def debugprint(text: str):
    """
    Prints a debugging output if the DO_DEBUG variable is set to True.
    """
    if(DO_DEBUG):
        print(text)

basepath = getcwd() + '\\'
path_data = basepath + 'data\\secom.txt'

# Read in the data from the text file
record_count = 0
raw_data = []
with open(path_data) as f:
    for l in f.readlines():
        record_count += 1
        line = l.replace('\n', '')
        raw_data.append(line.split(' '))

debugprint(record_count)

raw_data = np.array(raw_data)
df = pd.DataFrame(raw_data)



# Replace the NaN numbers (noise) with the feature mean for that column
feature_means = []

for col in range(0, len(list(df.columns))):
    col_mean = df[col].astype(float).mean()
    feature_means.append(col_mean)

for col in range(0, len(list(df.columns))):
    df[col] = df[col].replace('NaN', feature_means[col])



# Center the data points by subtracting each column by the mean for that column
data = df.astype(float).to_numpy()
centered =  np.subtract(data, feature_means)

# [Debugging] - DF of centered data
#pd.DataFrame(centered)

# Tranpose so we have a shape of eigen vecs that map to our features
covariance = np.cov(centered.T)
#covariance.shape

# Calculate the eigen values and vectors of the covariance
vals, vecs = np.linalg.eig(covariance)
#print(vals)
#print(vecs)



# Each eigenval contributes to the total percentage of data captured
# Create a dataframe of the iterative sum of total % of data captured
running_sum = 0.0
percentage_list = [0]
cumulative_list = [0]
eigval_sum = sum(vals)

for v in vals:
    percentage = (v/eigval_sum) *100        # Find the percentage
    percentage_list.append(percentage)      # Append the percentage to a discrete % list
    running_sum += percentage               
    cumulative_list.append(running_sum)     # Append the running cumulative % to a list 
    


# Export the percentages to a CSV for review
if(DO_DEBUG):
    df_percent = pd.DataFrame(percentage_list)
    df_percent.to_csv('df_percent.csv')

    df_cumulative = pd.DataFrame(cumulative_list)
    df_cumulative.to_csv('df_cumulative.csv')

    # print(percentage_list)
    # print(cumulative_list)

    

print(f'Cumulative Percentage of Data Captured in the the first 17 rows: ', float(sum(percentage_list[0:18])))
print(f'Discrete Percentages of Data Captured in the the first 17 rows:')
for i in range(0,18):
    print(float(percentage_list[i]))

# Project the original data (centered) onto the new basis
# projection = np.matmul(centered, vecs[:][:17].T)
