# nanoindent_growth_measurements

This Python project is dedicated to image processing, specifically analyzing pairs of images—usually "before" and "after" images—to quantify changes in a grid-like structure. It calculates the elongation of grid elements between two images and assesses the differences in width.
Here si example of data then detected points in small version of the app and then detected all the points in big version.
![intro](https://github.com/emmateki/nanoindent_growth_measurment/assets/116107969/6fb1c6e8-26ad-450a-becc-a26fc8696ffc)

## Table of Contents

- [nanoindent\_growth\_measurements](#nanoindent_growth_measurements)
  - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Run the UI](#run-the-ui)
    - [Usage](#usage)

### Installation

Begin by installing [conda](https://docs.conda.io/en/latest/miniconda.html) as a virtual environment manager. Then, create and set up the environment using the provided YAML file:

```sh
conda env create -f environment.yml
conda activate growth_measurment
```

### Run the UI 

**Linux**

```shell
streamlit run app_streamlit.py
```

**Windows**


to be added....


### Usage

Using this app is straightforward. Start by opening the Streamlit application. Select the data folder and choose the required version. Click on 'Start' to initiate the process. Once completed, an 'OUT' folder will be generated in the parent directory of the data folder.

The 'OUT' folder contains:
- Pictures illustrating the 'before' and 'after' states
- Results in CSV format

In case of any errors during processing, a `.log` file will be saved in the 'OUT' folder.

The app also includes a user manual for more detailed information on the available options.

For further details on the code and utilization of this project, visit the GitHub Wiki page for this repository.
