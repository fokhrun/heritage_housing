
import os
import dotenv
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Load environment variables
dotenv.load_dotenv()


@st.cache_data
def get_correlated_variables():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("CORRELATED_VARIABLE_FILES")
        )
    )


@st.cache_data
def get_na_data():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("NA_STATS_HOUSING_RECORDS_FILENAME")
        )
    )


@st.cache_data
def get_training_data():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("HOUSING_RECORDS_FILENAME")
        )
    )


@st.cache_data
def get_training_variable_info():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("VARIABLE_FILES")
        )
    )


@st.cache_data
def get_prediction_data():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("PREDICTION_SUBSET_FILENAME")
        )
    )


@st.cache_data
def get_prediction_features():

    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("PREDICTION_FEATURES_FILENAME")
        )
    )


@st.cache_resource
def get_estimator():
    return joblib.load(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("HOUSING_ESTIMATOR_NAME")
        )
    )


@st.cache_data
def get_prediction_features_info(prediction_choices, feature_info):

    sample = get_prediction_features()
    prediction_features = pd.DataFrame(np.zeros((1, sample.shape[1])), columns=sample.columns)
    prediction_subset_df = pd.DataFrame(prediction_choices, index=[0])

    # categorical features
    categorical_feature_names = feature_info.columns[feature_info.loc["featureType"] == "categorical"]
    inherited_houses_categorical = prediction_subset_df[categorical_feature_names]
    inherited_houses_categorical = inherited_houses_categorical.astype({
        var: "object" for var in categorical_feature_names
    })
    assert inherited_houses_categorical.shape == (1, len(categorical_feature_names))

    inherited_categorical = pd.get_dummies(inherited_houses_categorical).groupby(level=0).agg(["sum", "mean"])
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
    inherited_houses_temporal = prediction_subset_df[temporal_feature_names]
    assert inherited_houses_temporal.shape == (1, len(temporal_feature_names))

    latest_years_built = int(feature_info.loc["featureValues", "YearBuilt"].split("-")[-1])

    prediction_features["NumYearsSinceBuilt"] = latest_years_built - inherited_houses_temporal["YearBuilt"]
    prediction_features["NumYearsSinceRemodelled"] = latest_years_built - inherited_houses_temporal["YearRemodAdd"]

    estimator = get_estimator()

    prediction_features.loc[0, "SalePrice"] = estimator.predict(prediction_features)

    return prediction_features


@st.cache_data
def get_correlated_info():
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
        correlated_info.loc["featureDescription", feature_name] = correlated.loc[idx, "featureDescription"]
        correlated_info.loc["featureType", feature_name] = correlated.loc[idx, "featureType"]

    return feature_configuration, correlated_info
