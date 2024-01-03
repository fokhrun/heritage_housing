import streamlit as st
from utils.st_data_utils import get_correlated_variables, get_training_data, get_training_variable_info
from utils.st_insight_utils import plot_correlated_features
from utils.st_parameters import page_icon, plot_columns, target_column


page_title = "Sale Price Correlation"

st.set_page_config(page_title=page_title, page_icon=page_icon)
st.markdown(f"# {page_title}")

# load data
variable_info = get_training_variable_info()
training_data = get_training_data()
correlated = get_correlated_variables()

high_correlated_features = correlated["featureName"].tolist()
low_correlated_features = training_data.columns.difference(set(high_correlated_features))


high_correlated_fig = plot_correlated_features(
    plot_columns=plot_columns,
    variables=high_correlated_features,
    data=training_data,
    variable_info=variable_info,
    target_column=target_column
)


show_high_correlated = st.checkbox("Highly Correlated Features", value=True)
show_low_correlated = st.checkbox("Low Correlated Features")


if show_high_correlated:
    st.markdown("### Highly correlated Features")
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

        st.write("- Categorical features: if the median value per category increases with the sale price")

    st.markdown("##### Correlation Analysis")
    st.pyplot(high_correlated_fig)


if show_low_correlated:
    low_correlated_fig = plot_correlated_features(
        plot_columns=plot_columns,
        variables=low_correlated_features,
        data=training_data,
        variable_info=variable_info,
        target_column=target_column
    )

    st.markdown("### Low correlated Features")
    st.markdown("##### Correlation Analysis")
    st.pyplot(low_correlated_fig)
