import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Load processed dataset created by data_processing.py file
processed_data = pd.read_csv('data_features')
processed_data.set_index('nflId', inplace=True)

# Separate data into train/test sets
# Train dataset weeks 1-8 of 2022 NFL season
X_train = processed_data[processed_data.gameId < 2022110300].loc[:, ['dist_to_bc', 'dist_to_bc_avg', 'dist_rank', 'defender_in_front', 'sideline_dist', 'endzone_dist', 'rel_vec', 'is_dlineman', 'is_linebacker', 'is_secondary', 'is_pass', 'is_rush', 'is_bc_wr', 'is_bc_te', 'is_bc_rb', 'is_bc_qb']]
y_train = processed_data[processed_data.gameId < 2022110300].loc[:, ['tackle_participant']]

# Test dataset week 9 of 2022 NFL season
X_test = processed_data[processed_data.gameId >= 2022110300].loc[:, ['dist_to_bc', 'dist_to_bc_avg', 'dist_rank', 'defender_in_front', 'sideline_dist', 'endzone_dist', 'rel_vec', 'is_dlineman', 'is_linebacker', 'is_secondary', 'is_pass', 'is_rush', 'is_bc_wr', 'is_bc_te', 'is_bc_rb', 'is_bc_qb']]
y_test = processed_data[processed_data.gameId >= 2022110300].loc[:, ['tackle_participant']]


# Build basic model using training data
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train.values.ravel())
print(clf.score(X_test, y_test))

y_preds = pd.DataFrame(clf.predict_proba(X_test)).rename(columns={0:'pred_no', 1:'pred_yes'})

week_9_preds = pd.concat([y_test.reset_index(), y_preds], axis=1)

week_9_tackles = week_9_preds.groupby('nflId').sum(['tackle_participant', 'pred_yes'])

week_9_tackles['performance_diff'] = week_9_tackles.tackle_participant - week_9_tackles.pred_yes

cwd = os.getcwd()
players = pd.read_csv(cwd + '\data' + '\\players.csv')
week_9_final = week_9_tackles.merge(players, left_on='nflId', right_on='nflId')

mpl.rcParams['figure.dpi']= 150

plt.scatter(week_9_final.pred_yes, week_9_final.tackle_participant)
text = week_9_final.nflId.unique()

for i in range(len(text)):
    if week_9_final.performance_diff[i] >= 5:
        plt.annotate(week_9_final.displayName[i], (week_9_final.pred_yes[i], week_9_final.tackle_participant[i] + 0.25), size=6)

#find line of best fit
a, b = np.polyfit(week_9_final.pred_yes, week_9_final.tackle_participant, 1)
plt.plot(week_9_final.pred_yes, a*week_9_final.pred_yes+b, color='black', linestyle='--', linewidth=2)  
plt.xlabel('Expected Tackles')
plt.ylabel('Tackles')