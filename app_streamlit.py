import streamlit as st
import main
import config as cfg

st.set_page_config(page_title="Grid growth Desktop App", layout="wide")


def main_section():
    st.title("Grid Analyzer")
    selected_folder = st.text_input("Choose a folder with data:", key="-IN2-")
    input_version = st.selectbox("Version", ["Small", "Big"], index=0, key="version")

    if input_version == "Small":
        default_x1_thr = 40
        default_x_thr = 4
        default_n_rows = 7
        input_n_parts = 2
    elif input_version == "Big":
        default_x1_thr = 40
        default_x_thr = 4
        default_n_rows = 11
        input_n_parts = 3

    with st.expander("Advanced Settings", expanded=False):
        input_x1_thr = st.number_input(
            "X1 Threshold", min_value=1, value=default_x1_thr, key="x1_thr"
        )
        input_x_thr = st.number_input(
            "X Threshold", min_value=1, value=default_x_thr, key="x_thr"
        )
        input_n_rows = st.number_input(
            "Number of Rows",
            min_value=1,
            max_value=30,
            value=default_n_rows,
            key="n_rows",
        )
        input_n_parts = st.number_input(
            "Number of Parts", value=input_n_parts, key="n_parts", disabled=True
        )

    start_button = st.button("Start", type="primary")

    if start_button:
        if selected_folder:
            try:
                user_config = {
                    "x1_thr": input_x1_thr,
                    "x_threshold": input_x_thr,
                    "row_in_part": input_n_rows,
                    "parts": input_n_parts,
                    "version": "S" if input_version == "Small" else "M",
                }
                default_config = cfg.get_default_config()
                config = default_config | user_config
                main.main(selected_folder, config)

            except Exception as e:
                st.error(f"Error processing data: {e}")
        else:
            st.error("Please select data folder.")


def manual_section():
    st.title("User Manual")
    try:
        file = open("manual.md", "r")
        markdown_text = file.read()
        st.write(markdown_text)
        file.close()
    except FileNotFoundError:
        st.error("Manual file not found.")


selection = st.sidebar.selectbox("Navigation", ["Main", "User Manual"])

if selection == "Main":
    main_section()
elif selection == "User Manual":
    manual_section()
