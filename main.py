# set up
import pandas as pd
import numpy as np

# set seed
np.random.seed(seed=1234)

#load data
# Read from CSV to Pandas DataFrame
url ="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/titanic.csv"
df = pd.read_csv(url, header=0)
df.head()

# Exploratory data analysis
import matplotlib.pyplot as plt

# We can aslo extract some standard details by using the described functions
df.describe()

# Correlation matrix
plt.matshow(df.corr())
continous_features =df.describe().columns
plt.xticks((range(len(continous_features))), continous_features, rotation="45")
plt.yticks((range(len(continous_features))), continous_features, rotation="45")
plt.colorbar()
plt.show()

# Histograms
df["sex"].hist()

# Unique values
df["sex"].unique()

# Filtering

# Selecting data by feature
df["ticket"].head()

# Filtering
df[df["sex"]=="female"].head()

# sorting
df.sort_values("age", ascending=False).head()

# grouping
survived_group = df.groupby("survived")
survived_group.mean()

