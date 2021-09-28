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

# indexing
# Selecting row 0
df.iloc[0, :]
#We can use iloc to get rows or columns at particular positions in the dataframe
# Selecting a specific value
df.iloc[0, 1]

######Preprocessing#####

# Rows with at least one NaN value
df[pd.isnull(df).any(axis=1)].head()
# Drop rows with Nan values
df = df.dropna()
df = df.reset_index()
df.head()
# Dropping multiple columns
df = df.drop(["name","cabin","ticket"], axis=1)
df.head()
# Map feature values
df["sex"] = df["sex"].map({"female":0, "male":1}).astype(int)
df["embarked"] = df["embarked"].dropna().map({"S":0, "C":1, "Q":2}).astype(int)
df.head()
# Feature engineering
def get_family_size(sibsp, parch):
    family_size = sibsp + parch
    return family_size

"""
Once we define the function, we can use lambda
to apply that function on each row
(using the numbers of siblings and parents in each row to
determine the family size for each row).
"""
df["family_size"] = df[["sibsp", "parch"]].apply(lambda x: get_family_size(x["sibsp"], x["parch"]), axis=1)
df.head()

# Reorganize headers
df = df[["pclass", "sex", "age", "sibsp", "parch", "family_size", "fare", "embarked", "survived"]]
df.head()

# save_data
df.to_csv("processed_titanic.csv", index= True)