import streamlit as st 
from utils.st_data_utils import (
    get_optimisation_parameters,
    get_optimisation_performance,
    get_optimisation_feature_importance,
    get_model_parameters,
    get_model_performance,
    get_model_feature_importance,
    get_learning_curve_path
)
from utils.st_parameters import page_icon


page_title = "Training Performance Analysis"

st.set_page_config(page_title=page_title, page_icon=page_icon)
st.markdown(f"# {page_title}")


st.dataframe(get_optimisation_parameters())
st.dataframe(get_optimisation_performance())
st.dataframe(get_optimisation_feature_importance())
st.dataframe(get_model_parameters())
st.dataframe(get_model_performance())
st.dataframe(get_model_feature_importance())
st.image(get_learning_curve_path())
