import streamlit as st
from utils.st_data_utils import get_na_data, get_training_data, get_training_variable_info
from utils.st_insight_utils import plot_data_distribution
from utils.st_parameters import separator, page_icon, plot_columns


page_title = "Housing Data Analysis"

st.set_page_config(page_title=page_title, page_icon=page_icon)
st.markdown(f"# {page_title}")

# load data
variables = get_training_variable_info()
na_data = get_na_data()
training_data = get_training_data()

show_data_distribution = st.checkbox("Show Data Distribution")
show_data_description = st.checkbox("Show Data Description")

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
container.dataframe(data=display_dataframe.head(5))

if show_data_distribution:
    st.markdown("#### Data Distribution")
    st.pyplot(
        plot_data_distribution(
            plot_columns=plot_columns,
            variable_info=variables,
            data=display_dataframe
        )
    )

if show_data_description:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Data Description")
        data_description = display_dataframe.dropna().describe()
        st.dataframe(data=data_description)

    with col2:
        st.markdown("#### Missing Values")
        na_data_display = na_data[na_data["column"].isin(display_columns)].reset_index()
        st.dataframe(na_data_display[["column", "count", "percentage"]])
