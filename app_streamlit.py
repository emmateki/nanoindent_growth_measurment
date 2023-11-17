import streamlit as st
import main 

st.set_page_config(page_title="Grid growth Desktop App", layout="wide")


def main_section():

    st.title("Grid Analyzer")
    selected_folder = st.text_input(
        "Choose a folder with data:", key="-IN2-")
    input_version = st.selectbox(
        "Version", ['Small', 'Big'], index=0, key="version")

    if input_version == 'Small':
        default_x1_thr = 40
        default_x_thr = 4
        default_n_rows = 7
        input_n_parts = 2
    elif input_version == 'Big':
        default_x1_thr = 40
        default_x_thr = 4
        default_n_rows = 11
        input_n_parts = 3

    with st.expander("Advanced Settings", expanded=False):
        input_x1_thr = st.number_input(
            "X1 Threshold", min_value=1, value=default_x1_thr, key="x1_thr")
        input_x_thr = st.number_input(
            "X Threshold", min_value=1, value=default_x_thr, key="x_thr")
        input_n_rows = st.number_input(
            "Number of Rows", min_value=1, max_value=30, value=default_n_rows, key="n_rows")
        input_n_parts = st.number_input(
            "Number of Parts", value=input_n_parts, key="n_parts", disabled=True)

    start_button = st.button("Start",type="primary")

    if start_button:
        if selected_folder:
            try:
                config = {
                    'x1_treshold': input_x1_thr,
                    'x_treshold': input_x_thr,
                    'row_in_part': input_n_rows,
                    'parts': input_n_parts,

                    'version': 'S' if input_version == 'Small' else 'M',
                }

                main.main(selected_folder, config)

            except Exception as e:
                st.error(f"Error processing data: {e}")
        else:
            st.error("Please select data folder.")


def manual_section():
    st.title("User Manual")
    st.write(
        """

        #### Main Section:

        The **Grid Analyzer** application allows you to analyze grid data from pictures and track their growth. For more information about the code and compatible data, visit the [Github repository](https://github.com/emmateki/nanoindent_growth_measurment).

        Follow these steps to get started:

        1. **Choose a Folder with Data:**
           - Enter the path to the folder containing your data in the provided text input field. Ideally, the folder should contain subfolders with two .png pictures each, representing the data.

        2. **Select Version:**
            - Choose between 'Small' and 'Big' versions using the dropdown menu. This determines the settings for the analysis.

        3. **Advanced Settings:**
            - Expand the settings to fine-tune the analysis parameters. For basic usage, there's no need to modify these parameters. Here are the adjustable settings:
                - **X1 Threshold:** Filters points based on the x-coordinate deviation from the least square method.
                - **X Threshold:** Filters points based on the x-coordinate deviation from the median.
                - **Number of Rows:** Determine the number of rows in individual parts.
                - **Number of Parts:** Displays the number of parts based on the selected version. This setting is disabled as it is predefined based on the chosen version.

        4. **Start:**
            - Click the "Start" button to initiate the analysis with the provided configurations.

        5. **Output:**
            - Your OUT folder will be created in the parent folder of the data folder. In the OUT folder, you will find three subfolders:
                - **PICTURES:** Contains your original data pictures with the red detected grid. These pictures should be used for reviewing the quality of the result.
                - **RESULTS:** Contains .csv files with data. Additional information about the data can be found [here](https://github.com/emmateki/nanoindent_growth_measurment).
                - **ERROR:** Contains a .log file where you can find errors that occurred during the run.

        ---
        """
    )
selection = st.sidebar.selectbox(
    "Navigation", ["Main", "User Manual"])

if selection == "Main":
    main_section()
elif selection == "User Manual":
    manual_section()
