import streamlit as st
from utils.st_data_utils import get_correlated_variables, get_training_data, get_training_variables
from utils.st_insight_utils import plot_correlated_features


st.set_page_config(page_title="SalePrice Correlation", page_icon="📈")


target_column = "SalePrice"
variables = get_training_variables()
categorical_variables = variables[variables["featureType"] == "categorical"]["featureName"].tolist()  
numerical_variables = variables[variables["featureType"] == "numerical"]["featureName"].tolist()
temporal_variables = variables[variables["featureType"] == "temporal"]["featureName"].tolist()


training_data = get_training_data()
correlated = get_correlated_variables()

high_correlated_features = correlated["featureName"].tolist()
low_correlated_features = training_data.columns.difference(set(high_correlated_features))

# plotting variables
plot_columns = 2
high_correlated_plot_rows = int(len(high_correlated_features) / 2)
low_correlated_plot_rows = int(len(low_correlated_features) / 2)


high_correlated_fig = plot_correlated_features(
    plot_rows=high_correlated_plot_rows,
    plot_columns=plot_columns,
    features=high_correlated_features,
    data=training_data,
    variables=variables,
    categorical_variables=categorical_variables,
    target_column=target_column
)


with st.sidebar:
    show_high_correlated = st.checkbox("Highly Correlated Features", value=True)
    show_low_correlated = st.checkbox("Low Correlated Features")


st.markdown("# Correlation to Sale Price")

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
        plot_rows=low_correlated_plot_rows,
        plot_columns=plot_columns,
        features=low_correlated_features,
        data=training_data,
        variables=variables,
        categorical_variables=categorical_variables,
        target_column=target_column
    )

    st.markdown("### Low correlated Features")
    st.markdown("##### Correlation Analysis")
    st.pyplot(low_correlated_fig)
