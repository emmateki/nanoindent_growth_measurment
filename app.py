import PySimpleGUI as sg
import subprocess
import threading


sg.theme('DarkAmber')
selected_folder=None

layout = [
    [
        sg.Column(
            layout=[
                [sg.Text("Grid Analyzer", size=(20, 2), font=("Helvetica", 25), justification='center')],
                [sg.Text("Choose a folder with data: ", size=(15, 2)), sg.Input(key="-IN2-", change_submits=True), sg.FolderBrowse(key="-IN-"), sg.Button("Submit")],
                [sg.Text("Version", size=(10, 1)), sg.DropDown(['Small', 'Big'], size=(12, 1), key="-VERSION-"),
                 
                 sg.Button("Start", size=(20, 2)), sg.Text("", key='-LOADING-', size=(20, 1), visible=False)],
            ],
            element_justification="c"
        )
    ]
]
window = sg.Window("Vertex Desktop App", layout, finalize=True)

def run_script(selected_folder):
    if values['-VERSION-']=='Small':
        script_path = 'Smaller_version/main.py'

    elif values['-VERSION-']=='Big':
        script_path = 'Middle_version/main.py'

    else:
        sg.popup_error(f"Choose version of program.")


    command = f'python "{script_path}" "{selected_folder}"'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        sg.popup_error(f"Error executing the program: {e}")
    window['-LOADING-'].update("End of program", visible=True)

    
help_layout = [
    [sg.Text("Help Window", font=("fixed", 50))],
    [sg.Text("This is the help window content.")],
]

help_window = sg.Window("Help", help_layout, finalize=True)
help_window.hide()

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == "Submit":
        
        selected_folder = values["-IN-"]

    if event == "Start":
        if selected_folder is not None:
            window['-LOADING-'].update("Loading", visible=True)
            window.finalize()

            script_thread = threading.Thread(target=run_script, args=(selected_folder,))
            script_thread.start()
        else:
            sg.popup("Please select a folder before clicking Start.")
        

    if event == '-HELP-':

        help_window.un_hide()

window.close()
help_window.close()