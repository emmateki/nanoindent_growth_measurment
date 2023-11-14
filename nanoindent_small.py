import pathlib
import indentation_reader as indentation_reader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import visualization as visualization
import image_processing as ip
import argparse
import os
import numpy as np


def create_folder_if_not_exists(folder_path):
    if not folder_path.exists():
        folder_path.mkdir()


def main(data_root):
    data_root_path = pathlib.Path(data_root)
    df = indentation_reader.read_data(data_root_path)
    out_folder = data_root_path.parent / "OUT"
    create_folder_if_not_exists(out_folder)
    folder_names = [
        folder.name for folder in data_root_path.iterdir() if folder.is_dir()]

    for j in range(len(df)):
        if not df.empty:
            row = df.iloc[j]
            folder_name = folder_names[j]

            if folder_name :

                processing(row.img_before, folder_name,out_folder)
                processing(row.img_after, folder_name,out_folder)
            else:
                error_message = "Folder is empty."
                log_error(folder_name, error_message)



def save_pics(grid_final, img, folder_name, data_root_path):
    """
    Save processed images to a specified folder.

    Args:
        grid_final: Processed image data.
        img: Original image data.
        folder_name: Name of the folder.
        data_root_path: Path to the root data directory.
    """
    pictures_folder = data_root_path / "PICTURES"
    os.makedirs(pictures_folder, exist_ok=True)

    os.chdir(pictures_folder)
    plt.ioff()
    visualization.draw_grid(img, grid_final)

    if os.path.exists(f"{folder_name}.png"):
        plt.savefig(f"{folder_name}_after.png", format="png")
    else:
        plt.savefig(f"{folder_name}.png", format="png")

    plt.close()


def log_error(folder_name, error_message):
    """
    Log an error message to a file.

    Args:
        folder_name: Name of the folder where the error occurred.
        error_message: Error message to log.
    """
    error_folder = "ERROR"
    os.makedirs(error_folder, exist_ok=True)
    error_file = os.path.join(error_folder, f"{folder_name}_error.txt")
    with open(error_file, "a") as f:
        f.write(error_message + "\n")


def processing(img, folder_name,out_folder,minimum_detected_points=0.75):
    """
    Process image data and save results.

    Args:
        img: Original image data.
        folder_name: Name of the folder where the image is located.
        data_root_path: Path to the root data directory.

    """
    N_ROWS = 9
    PARTS = 2

    try:
        # coordinates x1,x2,y1,y2 of the region where the first point is located
        filtered_centers1_original = ip.find_origin_start(
            img, 450, 720, 500, 820)
        filtered_centers_last_original = ip.find_origin_last(
            img, 400, 700, 28320, 28650)
        filtered_centers_m_original = [0, 0]
        grid_manual = ip.manual_grid(
            filtered_centers_m_original, filtered_centers1_original, filtered_centers_last_original, N_ROWS, PARTS)

        grid_w_real_points = ip.create_new_grid(img, grid_manual)

        rearranged_grid = ip.rearrange_grid(grid_w_real_points)

        # Remove points based on least square fit and calculate linear dependency, equations, and coefficients
        grid_remove_points = ip.process_grid(
            rearranged_grid, X_THRESHOLD=4, X1_THRESHOLD=40, N_ROWS=9, PARTS=2)

        nan_count = np.isnan(grid_remove_points[:, :, 1]).sum()
        n_rows, n_columns, _ = grid_remove_points.shape
        if nan_count >= ((n_rows * n_columns) / minimum_detected_points):
            error_message = "Not enough points detected."
            log_error(folder_name, error_message)
        else:
            average_distance = ip.calculate_average_vertical_distance(
                grid_remove_points, N_ROWS=9)

            grid_final = ip.add_points_parts(
                average_distance, grid_remove_points, N_ROWS, PARTS)
            save_pics(grid_final, img, folder_name, out_folder)
            # Write the results to CSV files
            ip.calculate_distance_and_save_small(
                grid_final, folder_name, out_folder)
    except Exception as e:
        error_message = f"Error processing {folder_name}: {str(e)}"
        log_error(folder_name, error_message)
        # print(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data in a specified directory")
    parser.add_argument("data_root", type=str,
                        help="Path to the data root directory")

    args = parser.parse_args()
    main(args.data_root)
