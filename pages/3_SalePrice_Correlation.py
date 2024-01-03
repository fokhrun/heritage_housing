import os
import dotenv
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns


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
def get_training_variables():
    return pd.read_csv(
        os.path.join(
            os.getenv("STREAMLIT_DATA_PATH"),
            os.getenv("VARIABLE_FILES")
        )
    )


def get_correlation(data, categorical_variables, dependent_variable, independent_variable):

    if independent_variable in categorical_variables:
        corr_value = 0.0
        corr_description = "categorical:not applicable"
    else:
        corr = pd.concat(
            [data[dependent_variable], data[independent_variable]], 
            axis=1
        ).corr()
        corr_value = corr[independent_variable].iloc[0]
        if corr.isnull().sum().sum() > 0:
            corr_description = "no correlation"
        elif independent_variable == dependent_variable:
            corr_description = "not applicable"
            corr_value = 1.0
        else:
            if corr_value < 0.3:
                corr_description = "very weak correlation"
            elif 0.3 <= corr_value < 0.5:
                corr_description = "weak correlation"
            elif 0.5 <= corr_value < 0.7:
                corr_description = "moderate correlation"
            else:
                corr_description = "strong correlation"

    return corr_value, corr_description


def plot_correlated_features(plot_rows, plot_columns, features, data, variables, categorical_variables, target_column):

    plot_height = 2.5 * plot_rows
    plot_width = 6 * plot_columns

    fig, axs = plt.subplots(plot_rows, plot_columns, figsize=(plot_width, plot_height))

    for idx, var in enumerate(features):
        ax = axs[int(idx / 2)][idx % 2]
        corr_value, corr_description = get_correlation(
            data=data,
            categorical_variables=categorical_variables,
            dependent_variable=target_column,
            independent_variable=var
        )
        var_description = variables[variables["featureName"] == var]["featureDescription"].values[0]
        X_train = data[var]
        y_train = data[target_column]

        if var in categorical_variables:
            title = var_description
            sns.boxplot(ax=ax, x=X_train, y=y_train)
        else:
            title = f"{var_description} \n {corr_description}: {corr_value:.2f}"
            ax.scatter(X_train, y_train)

        ax.set_title(title)
        ax.set_ylabel(target_column)
        ax.set_xlabel(var)

    plt.tight_layout()
    return fig


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
