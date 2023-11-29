# nanoindent_growth_measurements

This Python project is dedicated to image processing, specifically analyzing pairs of images—usually "before" and "after" images—to quantify changes in a grid-like structure. It calculates the elongation of grid elements between two images and assesses the differences in width. It has two versions: 
* Big version - detect all points in three columns 
* Small version - detect first and last seven rows in three columns

Here si example of data with detected points in small version of the app and all the points in big version. \
![intro](https://github.com/emmateki/nanoindent_growth_measurment/assets/116107969/6fb1c6e8-26ad-450a-becc-a26fc8696ffc)

## Table of Contents

- [nanoindent\_growth\_measurements](#nanoindent_growth_measurements)
  - [Table of Contents](#table-of-contents)
    - [About](#about)
    - [Installation](#installation)
    - [Run the UI](#run-the-ui)
    - [Usage](#usage)

### About
This program operates as follows:  \
In the **small version**, it identifies the first and last points in the second column. Utilizing the average distance constant, it manually computes a grid encompassing the first and last seven rows. The program searches around these points within the manual grid. When a point is located, it is stored in an array. Empty spaces in the grid are determined through linear dependency across columns and the y-median within rows. 

The **big version** functions similarly, initially identifying the first, middle, and last points within an eleven-row span. It then searches for actual points around the manual grid, subsequently filling in missing points within empty regions using calculations involving medians and linear dependencies. This approach enables the detection of the entire grid. 

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

### Usage

Using this app is straightforward. Start by opening the Streamlit application. Select the data folder and choose the required version. Click on 'Start' to initiate the process. Once completed, an 'OUT' folder will be generated in the parent directory of the data folder.

The 'OUT' folder contains:
- Pictures illustrating the 'before' and 'after' states
- Results in CSV format

In case of any errors during processing, a `.log` file will be saved in the 'ERROR' folder.

The app also includes a user manual for more detailed information on the available options.

For further details on the code and utilization of this project, visit the GitHub Wiki page for this repository.
