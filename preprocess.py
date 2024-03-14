import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, FunctionTransformer
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(dataframe, target_column_name, columns_to_drop = None):
    """
    Preprocesses the input dataset by cleaning data, handling missing values,
    encoding categorical variables, and scaling numerical variables.

    Parameters:
    - dataframe: pandas DataFrame containing the dataset to preprocess.
    - target_column_name: string, name of the target column.
    - columns_to_drop: list of strings, names of columns to be dropped.

    Returns:
    - X_processed: The preprocessed features as a DataFrame or numpy array (depending on the processing).
    - y_processed: The preprocessed target variable.

    """
    # Remove duplicate rows, keeping the first occurrence
    dataframe = dataframe.drop_duplicates() 

    # Separate features and target if needed (assume target is the last column)
    columns_to_drop.append(target_column_name)
    features = dataframe.drop(columns = columns_to_drop)
    target = dataframe[[target_column_name]]

    # Define which columns are numeric and which are categorical
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object', 'category']).columns
    
    # preprocess target column include handle missing values
    target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='No Disorder')),
        ('flat' , FunctionTransformer(np.ravel))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    x_preprocessed = preprocessor.fit_transform(features)

    y_preprocessed  = target_transformer.fit_transform(target)
    le = LabelEncoder()
    y_preprocessed = le.fit_transform(y_preprocessed)

    #over sampling to balance data
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(x_preprocessed, y_preprocessed)


    # Return preprocessed features and target
    return X_sm, y_sm

