import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Get current working directory
cwd = os.getcwd()
# Get players csv data
players = pd.read_csv(cwd + '\data' + '\\players.csv')

# Load processed dataset created by data_processing.py file
processed_data = pd.read_csv('processed_data')

# Separate data into train/test sets
# Train dataset weeks 1-8 of 2022 NFL season
X_train = processed_data[processed_data.gameId < 2022110300].loc[:, ['dist_to_bc', 'dist_to_bc_avg', 'dist_rank', 'defender_in_front', 'sideline_dist', 'endzone_dist', 'rel_angle', 'rel_speed', 'is_dlineman', 'is_linebacker', 'is_secondary', 'is_pass', 'is_rush', 'is_bc_wr', 'is_bc_te', 'is_bc_rb', 'is_bc_qb']]
y_train = processed_data[processed_data.gameId < 2022110300].loc[:, ['tackle_participant']]

# Test dataset week 9 of 2022 NFL season
X_test = processed_data[processed_data.gameId >= 2022110300].loc[:, ['dist_to_bc', 'dist_to_bc_avg', 'dist_rank', 'defender_in_front', 'sideline_dist', 'endzone_dist', 'rel_angle', 'rel_speed', 'is_dlineman', 'is_linebacker', 'is_secondary', 'is_pass', 'is_rush', 'is_bc_wr', 'is_bc_te', 'is_bc_rb', 'is_bc_qb']]
y_test = processed_data[processed_data.gameId >= 2022110300].loc[:, ['tackle_participant']]
y_test.reset_index(drop=True, inplace=True)

# Save test set player IDs
y_test_ids = processed_data[processed_data.gameId >= 2022110300].loc[:, ['nflId']]
y_test_ids.reset_index(drop=True, inplace=True)

# Build basic model using training data
gb_clf = GradientBoostingClassifier()
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'n_estimators': [20, 50, 100, 200, 500],
    'max_depth': [1, 2, 3, 4, 5]}
tuned_clf = RandomizedSearchCV(gb_clf, param_grid, random_state=0, n_jobs=-1, scoring='neg_brier_score')
# Best parameters: lr = 0.1, n_est = 200, max_depth = 3
tuned_clf.fit(X_train, y_train.values.ravel())

# Plot feature importance scores
importances = tuned_clf.best_estimator_.feature_importances_
indices = np.argsort(importances)
fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(X_train.columns)[indices])
plt.title('Classifier Feature Importance Scores')
plt.show()

# Calculate tackle probabilities using gradient boosted classifier
y_preds = pd.DataFrame(tuned_clf.predict_proba(X_test)).rename(columns={0:'pred_no', 1:'pred_yes'})
week_9_preds = pd.concat([y_test_ids, y_test, y_preds], axis=1)
week_9_tackles = week_9_preds.groupby('nflId').sum(['tackle_participant', 'pred_yes'])
week_9_tackles['performance_diff'] = week_9_tackles.tackle_participant - week_9_tackles.pred_yes
# Add player names to final dataframe
week_9_final = week_9_tackles.merge(players, left_on='nflId', right_on='nflId')


# Create week 9 tackling analysis plot
mpl.rcParams['figure.dpi']= 150
# Plot data points for week 9 defenders
plt.scatter(week_9_final.pred_yes, week_9_final.tackle_participant, s=15)
text = week_9_final.nflId.unique()
# Annotate high performing player names
for i in range(len(text)):
    if (week_9_final.performance_diff[i] >= 4.5) & (week_9_final.displayName[i] != 'Richie Grant'):
        plt.annotate(week_9_final.displayName[i], (week_9_final.pred_yes[i], week_9_final.tackle_participant[i] + 0.25), size=5.5)
#Find line of best fit
a, b = np.polyfit(week_9_final.pred_yes, week_9_final.tackle_participant, 1)
plt.plot(week_9_final.pred_yes, a*week_9_final.pred_yes+b, color='gray', linestyle='--', linewidth=2)  
plt.title('NFL Week 9')
plt.xlabel('Expected Tackles')
plt.ylabel('Tackles')
plt.show()