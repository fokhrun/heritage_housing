
import numpy as np
import pandas as pd
import streamlit as st
from utils.st_data_utils import (
    get_model_parameters,
    get_model_performance,
    get_learning_curve_path,
    get_optimisation_parameters,
    get_optimisation_performance
)
from utils.st_parameters import page_icon


page_title = "Training Performance"

st.set_page_config(page_title=page_title, page_icon=page_icon)
st.markdown(f"# {page_title}")

training_tab, optimisation_tab = st.tabs(["Model Training", "Hyperparameter Tuning"])
with training_tab:
    with st.expander("Show Model Performance"):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(get_model_performance(), hide_index=True)
        with col2:
            st.dataframe(get_model_parameters(), hide_index=True)

    with st.expander("Model Learning Curve"):
        st.image(get_learning_curve_path())

with optimisation_tab:

    with st.expander("Optimisation Performance"):
        st.dataframe(
            get_optimisation_performance().style.format({
                "mse": "{:.2f}",
                "r2": "{:.2f}",
            }),
            hide_index=True
        )

    with st.expander("Optimisation Parameters"):
        optimisation_parameters = get_optimisation_parameters()

        best_value = np.max(optimisation_parameters["rmse"])
        best_value_parameters = optimisation_parameters[optimisation_parameters["rmse"] == best_value]

        st.dataframe(optimisation_parameters.style.format({
                "learning_rate": "{:.2f}",
                "max_depth": "{:.0f}",
                "n_estimators": "{:.0f}",
                "rmse": "{:.2f}",
                "min_child_weight": "{:.0f}",
            }).set_properties(
                **{'background-color': 'grey'},
                subset=pd.IndexSlice[best_value_parameters.index, best_value_parameters.columns]
            ),
            hide_index=True
        )
        st.write("- The marked parameters provide the best value and, thus, chosen for model training. ")
