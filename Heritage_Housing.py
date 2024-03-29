"""Heritage Housing application."""

# pylint: disable=C0103,E0401

import streamlit as st
from utils.st_parameters import page_icon


page_title = "Heritage Housing"

st.set_page_config(page_title=page_title, page_icon=page_icon)

st.markdown(f"## {page_title}")


st.markdown("""
#### Business Requirements

The dashboard application that allows her to maximize sale price of her inherited houses
as well as any other houses.

The dashboard should includes the following features:
- prediction of house sale prices from her 4 inherited houses, and any other house in Ames,
Iowa. The prediction should be generated by a reliable ML system.
- data visualizations of training data that shows correlations of the properties of houses
against the sale price. It is sufficient to use conventional data analysis techniques.
- A way to explain the predictions made by the estimator. The estimator should demonstrate
an R2 score of at least 0.75 on the training and the test data.
- Hypothesis used in the project and how it has been validated

#### Project Dataset

The dataset used in this project is a curated dataset of publicly available Ames Housing dataset.
The dataset contains 24 explanatory variables describing key attributes of residential homes and
their sale prices in Ames, Iowa. It contains unique records of 1430 houses.
The dataset can be downloaded from
[here](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data).

- For additional information, please visit and read the
Project [README](https://github.com/fokhrun/heritage_housing/blob/documentation/README.md) file.
""")
