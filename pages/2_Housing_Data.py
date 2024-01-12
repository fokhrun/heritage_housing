"""Streamlit page for housing data analysis"""

# pylint: disable=C0103,E0401

import streamlit as st
from utils.st_data_utils import (
    get_correlated_variables, get_na_data, get_training_data, get_training_variable_info,
    view_training_data
)
from utils.st_insight_utils import plot_correlated_features, plot_data_distribution
from utils.st_parameters import page_icon, plot_columns, separator, target_column


page_title = "Housing Data Analysis"

st.set_page_config(page_title=page_title, page_icon=page_icon)
st.markdown(f"# {page_title}")


# load data
variable_info = get_training_variable_info()
na_data = get_na_data()
training_data = get_training_data()
correlated = get_correlated_variables()


training_data_tab, correlation_tab = st.tabs([
    "Basic Data Analysis", f"Correlation To {target_column}"
])


with training_data_tab:
    # work with subset of the training data
    container = st.container()
    current_page = container.number_input(
        "Column Set",
        min_value=1,
        max_value=int(len(training_data.columns) / separator),
        value="min",
        step=1,
    )
    display_columns = training_data.columns[
        (current_page-1) * separator: current_page * separator
    ]
    display_dataframe = training_data[display_columns]

    container.markdown("#### Data Snapshot")

    container.dataframe(
        view_training_data(display_dataframe.head(5)),
        hide_index=True
    )

    with st.expander("Data Distribution"):
        st.pyplot(
            plot_data_distribution(
                plot_columns=plot_columns,
                variable_info=variable_info,
                data=display_dataframe
            )
        )

    with st.expander("Data Description"):
        data_description = display_dataframe.dropna().describe()
        st.dataframe(data_description.style.format("{:.2f}"))

    with st.expander("Missing Values"):
        na_data_display = na_data[na_data["column"].isin(display_columns)].reset_index()
        st.dataframe(na_data_display[["column", "percentage"]], hide_index=True)

with correlation_tab:
    # load data
    high_correlated_features = correlated["featureName"].tolist() + [target_column]
    low_correlated_features = training_data.columns.difference(set(high_correlated_features))

    high_correlated_fig = plot_correlated_features(
        plot_columns=plot_columns,
        chosen_variables=high_correlated_features,
        data=training_data,
        variable_info=variable_info,
        target_column=target_column
    )

    st.markdown("### Highly Correlated Features")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(correlated[["featureName", "featureValues"]], hide_index=True)

    with col2:
        st.write("""
            These variables are considered to be highly correlated to sale price.
            and are used in the model training and prediction. The selection is based
            on the following criterias:
        """)

        st.write("- Numerical variables: moderate and above correlation coefficient score")
        st.write("- Temporal variables: moderate and above correlation coefficient score")
        st.write("- Categorical variables: if the median value per category increases"
                 "with the sale price")

    with st.expander(f"High Correlated Features vs {target_column}"):
        st.pyplot(high_correlated_fig)

    with st.expander(f"Low Correlated Features vs {target_column}"):
        low_correlated_fig = plot_correlated_features(
            plot_columns=plot_columns,
            chosen_variables=low_correlated_features,
            data=training_data,
            variable_info=variable_info,
            target_column=target_column
        )
        st.pyplot(low_correlated_fig)
