import streamlit as st
import subprocess

st.set_page_config(page_title="Grid growth Desktop App", layout="wide")

st.title("Grid Analyzer")

selected_folder = st.sidebar.text_input(
    "Choose a folder with data:", key="-IN2-")

version = st.sidebar.selectbox("Version", ['Small', 'Big'], key="-VERSION-")
start_button = st.sidebar.button("Start")

if start_button:
    if selected_folder:
        script_path = 'main.py'
        if version == 'Small':
            command = f'python "{script_path}" "{selected_folder}" --x1-thr 40 --x-threshold 4 --n-rows 7 --n-parts 2 --version S'
        elif version == 'Big':
            command = f'python "{script_path}" "{selected_folder}" --x1-thr 40 --x-threshold 4 --n-rows 11 --n-parts 3 --version M'
        else:
            st.error("Choose version of the program.")

        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Error executing the program: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("Help"):
    st.sidebar.markdown("## Help Window")
    st.sidebar.markdown("This is the help window content.")

st.write("Main content goes here...")
