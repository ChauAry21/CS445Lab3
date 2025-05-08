from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
 
car_evaluation = fetch_ucirepo(id=19) 

# combine features + target for visualization
df = pd.concat([car_evaluation.data.features, car_evaluation.data.targets], axis=1)
print(df.head())

# encode categorical variables
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# create scatter plot
scatter = ax.scatter(
    df_encoded['buying'], 
    df_encoded['maint'], 
    df_encoded['safety'],
    c=df_encoded['class'], 
    cmap='viridis',
    s=50,
    alpha=0.6
)

# label
ax.set_xlabel('buying price (encoded)')
ax.set_ylabel('maintenance price (encoded)')
ax.set_zlabel('safety (encoded)')
ax.set_title('3d scatter of car features vs. acceptability')

# colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('car acceptability (encoded)')

plt.tight_layout()
plt.savefig('CarEvaluation_3DScatterFeatures.png')
plt.show()

# create pivot table of counts for three features
pivot_table = df.groupby(['buying', 'maint', 'safety']).size().reset_index(name='count')

# encode categorical variables for plotting
le = LabelEncoder()
pivot_table['buying_encoded'] = le.fit_transform(pivot_table['buying'])
pivot_table['maint_encoded'] = le.fit_transform(pivot_table['maint'])
pivot_table['safety_encoded'] = le.fit_transform(pivot_table['safety'])

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# create 3d bar chart
dx = dy = dz = 0.5
ax.bar3d(
    pivot_table['buying_encoded'], 
    pivot_table['maint_encoded'], 
    np.zeros(len(pivot_table)),
    dx, dy, pivot_table['count'], 
    color='skyblue',
    alpha=0.8
)

# add labels
ax.set_xlabel('buying price (encoded)')
ax.set_ylabel('maintenance price (encoded)')
ax.set_zlabel('count of combinations')
ax.set_title('3d bar chart of car feature combinations')

# set custom tick labels
ax.set_xticks(sorted(pivot_table['buying_encoded'].unique()))
ax.set_xticklabels(sorted(df['buying'].unique()))

ax.set_yticks(sorted(pivot_table['maint_encoded'].unique()))
ax.set_yticklabels(sorted(df['maint'].unique()))

plt.tight_layout()
plt.savefig('CarEvaluation_3DBarChart.png')
plt.show()

# calculate acceptance rate by feature combinations
acceptance = df.groupby(['buying', 'maint', 'safety'])['class'].apply(
    lambda x: (x.isin(['acc', 'good', 'v-good'])).mean()
).reset_index(name='acceptance_rate')

# pivot for surface plot
pivot_acceptance = acceptance.pivot_table(
    index='buying', 
    columns='maint', 
    values='acceptance_rate', 
    aggfunc='mean'
)

# encode buying + maint for numerical plotting
buying_levels = sorted(df['buying'].unique())
maint_levels = sorted(df['maint'].unique())
buying_codes = {level: i for i, level in enumerate(buying_levels)}
maint_codes = {level: i for i, level in enumerate(maint_levels)}

X = np.array([buying_codes[b] for b in pivot_acceptance.index])
Y = np.array([maint_codes[m] for m in pivot_acceptance.columns])
X, Y = np.meshgrid(X, Y)
Z = pivot_acceptance.values.T

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# create surface plot
surf = ax.plot_surface(
    X, Y, Z, 
    cmap='coolwarm',
    edgecolor='none',
    alpha=0.8
)

# add labels
ax.set_xlabel('buying price')
ax.set_ylabel('maintenance price')
ax.set_zlabel('acceptance rate')
ax.set_title('3d surface plot of car acceptance rates by price factors')

# set custom tick labels
ax.set_xticks(range(len(buying_levels)))
ax.set_xticklabels(buying_levels)

ax.set_yticks(range(len(maint_levels)))
ax.set_yticklabels(maint_levels)

# cdd colorbar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig('CarEvaluation_3DSurfaceAcceptance.png')
plt.show()

#comments:
#scatter plot show how safety interacts with price factors to influence car acceptability
#bar chart shows the distribution of features in the dataset
#surface plot shows how price factors combine to affect acceptance rates