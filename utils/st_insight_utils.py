import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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
