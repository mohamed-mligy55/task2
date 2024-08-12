import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
# For demonstration, we'll use an open dataset from sklearn. 
# You would replace this with your local dataset, e.g., `pd.read_csv('path/to/dataset.csv')`
data = fetch_openml("titanic", version=1, as_frame=True)
df = data.frame

# Checking the first few rows of the dataset
print(df.head())
# Identifying categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(exclude=['object']).columns

# Imputing missing values
# For numerical columns, we'll use the mean; for categorical, we'll use the most frequent value
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

imputer = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numerical_cols),
        ('cat', categorical_imputer, categorical_cols)
    ]
)

df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(df_imputed.isnull().sum())  # Checking if there are any missing values left
# Applying OneHotEncoder for categorical data
encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ], remainder='passthrough'  # Keep the rest of the columns unchanged
)

df_encoded = pd.DataFrame(encoder.fit_transform(df_imputed))
print(df_encoded.head())
# Applying Standardization (z-score normalization)
scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
print(df_scaled.head())
# Applying PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance

df_reduced = pd.DataFrame(pca.fit_transform(df_scaled))
print(df_reduced.head())
# Full pipeline
pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('encoder', encoder),
    ('scaler', scaler),
    ('pca', pca)
])

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['survived']), df['survived'], test_size=0.2, random_state=42)

# Applying the pipeline to the training data
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)

print(f"Shape of Training Data after preprocessing: {X_train_preprocessed.shape}")
print(f"Shape of Testing Data after preprocessing: {X_test_preprocessed.shape}")
