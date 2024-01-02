
import os
import dotenv
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Load environment variables
dotenv.load_dotenv()


title = "Housing Price Prediction"


st.set_page_config(page_title=title, page_icon="📊")
st.markdown(f"# {title}")
st.sidebar.header(title)


def get_correlated_variables():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"), 
            os.getenv("CORRELATED_VARIABLE_FILES")
        )
    )


def get_prediction_data():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("PREDICTION_SUBSET_FILENAME")
        )
    )

def get_prediction_features(prediction_choices, feature_info):

    sample = pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("PREDICTION_FEATURES_FILENAME")
        )
    )

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

    estimator = joblib.load(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"), 
            os.getenv("HOUSING_ESTIMATOR_NAME")
        )
    )

    prediction_features.loc[0, "SalePrice"] = estimator.predict(prediction_features)

    return prediction_features


correlated = get_correlated_variables()
correlated_flipped = pd.DataFrame(columns=correlated["featureName"])

feature_configuration = {}

for idx in correlated.index:
    feature_name = correlated.loc[idx, "featureName"]
    feature_values = correlated.loc[idx, "featureValues"]

    if ":" in feature_values:  # ensure showing the values separated by a newline
        feature_configuration[feature_name] = [_.strip() for _ in feature_values.split(",")]
        feature_values = "\n".join([f"{_}\n" for _ in feature_configuration[feature_name]])
    else:
        feature_configuration[feature_name] = [_.strip() for _ in feature_values.split("-")]
    correlated_flipped.loc["featureValues", feature_name] = feature_values    
    correlated_flipped.loc["featureDescription", feature_name] = correlated.loc[idx, "featureDescription"]
    correlated_flipped.loc["featureType", feature_name] = correlated.loc[idx, "featureType"]


prediction_results = get_prediction_data()
prediction_results_wo_target = prediction_results.columns.drop("SalePrice")
target_column = "SalePrice"
prediction_results_sorted = pd.concat([
        prediction_results[target_column],
        prediction_results[prediction_results_wo_target]],
    axis=1
)


choices = prediction_results_sorted.iloc[-1].to_dict()

for var in correlated_flipped.columns:

        if correlated_flipped.loc["featureType", var] == "categorical":

            options = []
            captions = []
            for _ in feature_configuration[var]:
                option, caption = _.split(":")
                options.append(option)
                captions.append(caption)

            choices[var] = st.radio(
                label=correlated_flipped.loc["featureDescription", var],
                options=options,
                captions=captions,
                index=options.index(str(choices[var])),
                horizontal=True
            )

        else:
            choices[var] = st.slider(
                correlated_flipped.loc["featureDescription", var],
                min_value=int(feature_configuration[var][0]),
                max_value=int(feature_configuration[var][1]),
                value=int(choices[var])
            )


def index_formatter(idx):
    return f"House {idx + 1}"


next_index = index_formatter(prediction_results_sorted.index[-1] + 1)
prediction_results_sorted.index = [index_formatter(idx) for idx in prediction_results_sorted.index]

if st.button("Predict"):
    prediction = get_prediction_features(prediction_choices=choices, feature_info=correlated_flipped)
    choices[target_column] = prediction.loc[0, target_column]
    prediction_results_sorted.loc[next_index, :] = choices


for var in correlated_flipped.columns:
    if var == target_column:
        prediction_results_sorted[var] = prediction_results_sorted[var].apply(lambda _: f"${_:,.0f}")
    try:
        prediction_results_sorted[var] = prediction_results_sorted[var].apply(int)
    except ValueError:
        prediction_results_sorted[var] = prediction_results_sorted[var].apply(str)
    
st.write(
    prediction_results_sorted\
        .style\
        .set_properties(**{"background-color": "blue"}, subset=[target_column])
)
