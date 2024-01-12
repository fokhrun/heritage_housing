"""Utility functions for insight generation"""

# pylint: disable=R0914,C0103

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_correlation(data, categorical_variables, dependent_variable, independent_variable):
    """ Get the correlation between the dependent and independent variables.

    Parameters
    ----------
    data : pandas.DataFrame
    categorical_variables : list
    dependent_variable : str
    independent_variable : str

    Returns
    -------
    corr_value : float
    """
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


def plot_correlated_features(plot_columns, chosen_variables, data, variable_info, target_column):
    """Plot the correlated features.

    Parameters
    ----------
    plot_columns : list
    chosen_variables : list
    data : pandas.DataFrame
    variable_info : pandas.DataFrame
    target_column : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    categorical_variables = variable_info[
        variable_info["featureType"] == "categorical"
    ]["featureName"].tolist()

    plot_rows = int(len(chosen_variables) / 2)
    plot_height = 2.5 * plot_rows
    plot_width = 12

    fig, axs = plt.subplots(plot_rows, plot_columns, figsize=(plot_width, plot_height))

    for idx, var in enumerate(chosen_variables):
        ax = axs[int(idx / 2)][idx % 2]
        corr_value, corr_description = get_correlation(
            data=data,
            categorical_variables=categorical_variables,
            dependent_variable=target_column,
            independent_variable=var
        )
        var_description = variable_info[
            variable_info["featureName"] == var
        ]["featureDescription"].values[0]
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


def plot_data_distribution(plot_columns, variable_info, data):
    """
    Plot the data distribution.

    Parameters
    ----------
    plot_columns : list
    variable_info : pandas.DataFrame
    data : pandas.DataFrame

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    categorical_variables = variable_info[
        variable_info["featureType"] == "categorical"
    ]["featureName"]

    plot_rows = int(len(data.columns) / plot_columns)
    plot_width = 12
    plot_height = 2.5 * plot_rows
    fig, axs = plt.subplots(plot_rows, plot_columns, figsize=(plot_width, plot_height))

    for idx, var in enumerate(data.columns):

        ax = axs[int(idx / plot_columns)][idx % plot_columns]

        if var in categorical_variables:
            sns.countplot(data=data, x=var, ax=ax)
        else:
            sns.histplot(data=data, x=var, ax=ax)

        ax.set_title(
            f"{variable_info[variable_info['featureName'] == var]['featureDescription'].values[0]}"
        )
        ax.set_xlabel(var)

    plt.tight_layout()

    return fig
