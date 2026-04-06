import numpy as np
from scipy.stats import ttest_rel

naiveBayesF1Scores = np.load("f1_scores_NB.npy").tolist()
logisticRegressionF1Scores = np.load("f1_scores_LR.npy").tolist()

meanF1NB = np.mean(naiveBayesF1Scores)
meanF1LR = np.mean(logisticRegressionF1Scores)

t_statistic, p_value = ttest_rel(naiveBayesF1Scores, logisticRegressionF1Scores)
print("t-statistic:", t_statistic)
print("p-value:", p_value)