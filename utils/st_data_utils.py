
import os
import dotenv
import pandas as pd
import streamlit as st


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
