
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

