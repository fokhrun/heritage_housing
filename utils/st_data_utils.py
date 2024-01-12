"""Download data and other resources."""

# pylint: disable=R0914

import os
import dotenv
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Load environment variables
dotenv.load_dotenv()


def get_path(filename):
    """
    Get the path of a file in the data folder.

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
        str
    """
    return os.path.join(os.getenv("STREAMLIT_DATA_PATH"), filename)


@st.cache_data
def get_correlated_variables():
    """
    Get the correlated variables.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("CORRELATED_VARIABLE_FILES")))


@st.cache_data
def get_na_data():
    """
    Get the missing values data.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("NA_STATS_HOUSING_RECORDS_FILENAME")))


@st.cache_data
def get_training_variable_info():
    """
    Get the training variable information.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("VARIABLE_FILES")))


@st.cache_data
def get_training_data():
    """
    Get the training data.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("HOUSING_RECORDS_FILENAME")))


def view_training_data(data):
    """
    View the training data.

    Parameters
    ----------
    data : pandas.DataFrame
        Training data.

    Returns
    -------
    pandas.DataFrame
        styled for display
    """
    variables = get_training_variable_info()
    categorical_features = variables[
        variables["featureType"] == "categorical"
        ]["featureName"].values

    formatter = {}

    for var in data.columns:
        if var not in categorical_features:
            formatter[var] = "{:.0f}"
        else:
            if data[var].dtype == "float64":
                formatter[var] = "{:.0f}"

    return data.style.format(formatter)


@st.cache_data
def get_prediction_data():
    """
    Get the prediction data.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("PREDICTION_SUBSET_FILENAME")))


@st.cache_data
def get_prediction_features():
    """
    Get the prediction features.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("PREDICTION_FEATURES_FILENAME")))


@st.cache_data
def get_optimisation_performance():
    """
    Get the optimisation performance.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("OPTIMISATION_PERFORMANCE_FILENAME")))


@st.cache_data
def get_optimisation_feature_importance():
    """
    Get the optimisation feature importance.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("OPTIMISATION_FEATURES_IMPORTANCE_FILENAME")))


@st.cache_data
def get_optimisation_parameters():
    """
    Get the optimisation parameters.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("OPT_PARAMETERS_FILENAME")))


@st.cache_data
def get_model_performance():
    """
    Get the model performance.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("MODEL_PERFORMANCE_FILENAME")))


@st.cache_data
def get_model_feature_importance():
    """
    Get the model feature importance.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("MODEL_FEATURES_IMPORTANCE_FILENAME")))


@st.cache_data
def get_model_parameters():
    """
    Get the model parameters.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(get_path(os.getenv("MODEL_PARAMETERS_FILENAME")))


@st.cache_resource
def get_learning_curve_path():
    """
    Get the learning curve path.

    Returns
    -------
    str
    """
    return get_path(os.getenv("LEARNING_CURVE_FILENAME"))


@st.cache_resource
def get_prediction_correlation():
    """
    Get the prediction correlation.

    Returns
    -------
    str
    """
    return get_path(os.getenv("PREDICTION_CORRELATION_FILENAME"))


@st.cache_resource
def get_estimator():
    """
    Get the estimator.

    Returns
    -------
    sklearn.base.BaseEstimator
    """
    return joblib.load(get_path(os.getenv("HOUSING_ESTIMATOR_NAME")))


@st.cache_data
def get_prediction_feature_info(prediction_choices, feature_info):
    """
    Get the prediction feature information.

    Parameters
    ----------
    prediction_choices : pandas.DataFrame
    feature_info : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    sample = get_prediction_features()
    prediction_features = pd.DataFrame(np.zeros((1, sample.shape[1])), columns=sample.columns)
    prediction_subset_df = pd.DataFrame(prediction_choices, index=[0])

    # categorical features
    categorical_feature_names = feature_info.columns[
        feature_info.loc["featureType"] == "categorical"
    ]
    inherited_houses_categorical = prediction_subset_df[categorical_feature_names]
    inherited_houses_categorical = inherited_houses_categorical.astype({
        var: "object" for var in categorical_feature_names
    })
    assert inherited_houses_categorical.shape == (1, len(categorical_feature_names))

    inherited_categorical = pd.get_dummies(inherited_houses_categorical)\
        .groupby(level=0)\
        .agg(["sum", "mean"])
    inherited_categorical.columns = ["_".join(_) for _ in inherited_categorical.columns]
    inherited_categorical.reset_index()
    for col in inherited_categorical.columns:
        prediction_features.loc[0, col] = inherited_categorical.loc[0, col]

    # numerical features
    numerical_feature_names = feature_info.columns[feature_info.loc["featureType"] == "numerical"]
    inherited_houses_numerical = prediction_subset_df[numerical_feature_names]
    inherited_houses_numerical = inherited_houses_numerical.astype({
        var: "float" for var in numerical_feature_names
    })
    assert inherited_houses_numerical.shape == (1, len(numerical_feature_names))

    inherited_numerical = inherited_houses_numerical\
        .groupby(level=0)\
        .agg(["count", "mean", "max", "min", "sum"])
    inherited_numerical.columns = ["_".join(_) for _ in inherited_numerical.columns]
    for col in inherited_numerical.columns:
        prediction_features.loc[0, col] = inherited_numerical.loc[0, col]

    # temporal features
    temporal_feature_names = feature_info.columns[feature_info.loc["featureType"] == "temporal"]
    inherit_houses_t = prediction_subset_df[temporal_feature_names]
    assert inherit_houses_t.shape == (1, len(temporal_feature_names))

    latest_year = int(feature_info.loc["featureValues", "YearBuilt"].split("-")[-1])

    prediction_features["NumYearsSinceBuilt"] = latest_year - inherit_houses_t["YearBuilt"]
    prediction_features["NumYearsSinceRemodelled"] = latest_year - inherit_houses_t["YearRemodAdd"]

    estimator = get_estimator()

    prediction_features.loc[0, "SalePrice"] = estimator.predict(prediction_features)

    return prediction_features


@st.cache_data
def get_correlated_info():
    """
    Get the correlated information.

    Returns
    -------
        dict, pandas.DataFrame
    """
    correlated = get_correlated_variables()
    feature_configuration = {}
    correlated_info = pd.DataFrame(columns=correlated["featureName"])

    for idx in correlated.index:
        feature_name = correlated.loc[idx, "featureName"]
        feature_values = correlated.loc[idx, "featureValues"]

        if ":" in feature_values:  # ensure showing the values separated by a newline
            feature_configuration[feature_name] = [_.strip() for _ in feature_values.split(",")]
            feature_values = "\n".join([f"{_}\n" for _ in feature_configuration[feature_name]])
        else:
            feature_configuration[feature_name] = [_.strip() for _ in feature_values.split("-")]

        correlated_info.loc["featureValues", feature_name] = feature_values
        correlated_info.loc["featureDescription", feature_name] = correlated.loc[
            idx, "featureDescription"
        ]
        correlated_info.loc["featureType", feature_name] = correlated.loc[idx, "featureType"]

    return feature_configuration, correlated_info
