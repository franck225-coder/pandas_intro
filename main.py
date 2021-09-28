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
