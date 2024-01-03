
import pandas as pd
import streamlit as st
from utils.st_data_utils import get_correlated_info, get_prediction_data, get_prediction_features_info
from utils.st_parameters import target_column, page_icon

title = "Housing Price Prediction"


st.set_page_config(page_title=title, page_icon=page_icon)
st.markdown(f"# {title}")
st.sidebar.header(title)

feature_configuration, correlated = get_correlated_info()
prediction_results = get_prediction_data()

prediction_results_wo_target = prediction_results.columns.drop("SalePrice")
prediction_results_sorted = pd.concat([
    prediction_results[target_column],
    prediction_results[prediction_results_wo_target]],
    axis=1
)


choices = prediction_results_sorted.iloc[-1].to_dict()

for var in correlated.columns:

    if correlated.loc["featureType", var] == "categorical":

        options = []
        captions = []
        for _ in feature_configuration[var]:
            option, caption = _.split(":")
            options.append(option)
            captions.append(caption)

        choices[var] = st.radio(
            label=correlated.loc["featureDescription", var],
            options=options,
            captions=captions,
            index=options.index(str(choices[var])),
            horizontal=True
        )

    else:
        choices[var] = st.slider(
            correlated.loc["featureDescription", var],
            min_value=int(feature_configuration[var][0]),
            max_value=int(feature_configuration[var][1]),
            value=int(choices[var])
        )


def index_formatter(idx):
    return f"House {idx + 1}"


next_index = index_formatter(prediction_results_sorted.index[-1] + 1)
prediction_results_sorted.index = [index_formatter(idx) for idx in prediction_results_sorted.index]

if st.button("Predict"):
    prediction = get_prediction_features_info(prediction_choices=choices, feature_info=correlated)
    choices[target_column] = prediction.loc[0, target_column]
    prediction_results_sorted.loc[next_index, :] = choices

for var in correlated.columns:
    if var == target_column:
        prediction_results_sorted[var] = prediction_results_sorted[var].apply(lambda _: f"${_:,.0f}")
    try:
        prediction_results_sorted[var] = prediction_results_sorted[var].apply(int)
    except ValueError:
        prediction_results_sorted[var] = prediction_results_sorted[var].apply(str)

st.write(
    prediction_results_sorted.style.set_properties(
        **{"background-color": "blue"},
        subset=[target_column]
    )
)

with st.sidebar:
    if st.button("Reset"):
        st.empty()
