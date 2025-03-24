import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Introduce some missing values for demonstration
np.random.seed(42)
for col in df.columns[:-1]:  # Exclude target column
    df.loc[np.random.choice(df.index, size=5, replace=False), col] = np.nan

# Handling missing values (Replace NaN with column mean)
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Normalize numerical columns
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Exploratory Data Analysis (EDA)
summary = df.describe().T  # Transpose for better readability

# Additional statistics
summary['median'] = df.median()
summary['std_dev'] = df.std()

# Data visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :-1])
plt.title("Boxplot of Iris Dataset Features")
plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=45)
plt.show()

sns.pairplot(df, hue='target', diag_kind='kde')
plt.show()

# Display summary report
print("Summary Report:")
print(summary)
