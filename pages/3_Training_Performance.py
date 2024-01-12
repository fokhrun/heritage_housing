"""Training Performance page shown when the user enters the application """

# pylint: disable=C0103,E0401

import numpy as np
import pandas as pd
import streamlit as st
from utils.st_data_utils import (
    get_correlated_variables,
    get_model_parameters,
    get_model_performance,
    get_learning_curve_path,
    get_optimisation_parameters,
    get_optimisation_performance,
    get_prediction_correlation
)
from utils.st_parameters import page_icon


page_title = "Training Performance"

st.set_page_config(page_title=page_title, page_icon=page_icon)
st.markdown(f"# {page_title}")

hypothesis_tab, training_tab, optimisation_tab = st.tabs([
    "Hypothesis", "Model Training", "Hyperparameter Tuning"
])

correlated = get_correlated_variables()

with hypothesis_tab:

    st.markdown("""
        The factors that effect housing prices are:

        - house size:
            - hypothesis: larger the property, higher the price
            - columns: `1stFlrSF`, `TotalBsmtSF`, `GarageArea`, `GrLivArea`
        - house condition:
            - hypothesis: better the condition, higher the price
            - columns: `KitchenQual`, `OverallCond`, `OverallQual`
        - house age: newer the house, higher the price
            - hypothesis: newer the house, higher the price
            - columns: `YearBuilt`, `YearRemodAdd`
    """)

    st.markdown("""
        Our hypothesis should be validated by the following observations:

        - Both the actual and predicted (on data unused in training) Sale Price should be 
        very strongly correlated
        - The predicted (on data unused in training) Sale Price should generally increase 
        with the increase in the house size, condition, and age. It should show correlation 
        to the columns mentioned above similarly to the actual sale price.
    """)

    st.markdown("""
        Note: Location desirability and number of rooms also have similar effect,
        but the dataset did not those.
    """)

with training_tab:
    with st.expander("Model Performance"):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(get_model_performance().style.format({
                "mse": "{:.2f}",
                "r2": "{:.2f}",
            }), hide_index=True)
        with col2:
            st.dataframe(get_model_parameters(), hide_index=True)

    with st.expander("Model Learning Curve"):
        st.image(get_learning_curve_path())

    with st.expander("Hypothesis validation"):
        st.image(get_prediction_correlation())
        st.markdown("""
            The above correlation plot validates the following:
            - the predicted sale price is very strongly correlated to the actual sale price
            - the predicted sale price typical increases with the increase in the house size, 
            condition, and age
        """)

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
        best_value_parameters = optimisation_parameters[
            optimisation_parameters["rmse"] == best_value
        ]

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
        st.write("""- The marked parameters provide the best value and, thus,
                 chosen for model training.""")
