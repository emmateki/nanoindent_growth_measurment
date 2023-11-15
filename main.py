import pathlib
import indentation_reader as indentation_reader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import visualization as visualization
import image_processing as ip
import argparse
import os
import numpy as np
import logging
import traceback

def main(data_root, config):
    data_root_path = pathlib.Path(data_root)
    out_folder = data_root_path.parent / "OUT"
    os.makedirs(out_folder, exist_ok=True)
    df = indentation_reader.read_data(data_root_path,out_folder)
    folder_names = [
        folder.name for folder in data_root_path.iterdir() if folder.is_dir()]

    for j in range(len(df)):
        if not df.empty:
            row = df.iloc[j]
            folder_name = folder_names[j]

            if folder_name:

                processing(row.img_before, folder_name, out_folder, config)
                processing(row.img_after, folder_name, out_folder, config)
            else:
                error_message = "Folder is empty."
                folder_name = "empty"
                log_error(folder_name, error_message, out_folder)


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


def configure_logger(folder_name, out_folder):
    log_folder = out_folder/"ERROR"
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, f"{folder_name}_error.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def log_error(folder_name, error_message, out_folder):
    """
    Log an error message using the configured logger.

    Args:
        folder_name: Name of the folder where the error occurred.
        error_message: Error message to log.
    """
    configure_logger(folder_name, out_folder)
    logging.error(error_message)


def processing(img, folder_name, out_folder, config, minimum_detected_points=0.75):
    """
    Process image data and save results.

    Args:
        img: Original image data.
        folder_name: Name of the folder where the image is located.
        data_root_path: Path to the root data directory.

    """

    try:
        N_ROWS = config['N_ROWS']
        PARTS = config['PARTS']
        VERSION = config['VERSION']
        
        # coordinates x1,x2,y1,y2 of the region where the first point is located
        filtered_centers1_original = ip.find_origin_start(
            img, 450, 720, 500, 820)
        filtered_centers_last_original = ip.find_origin_last(
            img, 400, 700, 28320, 28650)
        if VERSION == 'M':
            filtered_centers_m_original = ip.find_origin_middle(
                img, 450, 720, 14500, 14800)
        elif  VERSION == 'S':   
            filtered_centers_m_original = [0, 0]
        grid_manual = ip.manual_grid(
            filtered_centers_m_original, filtered_centers1_original, filtered_centers_last_original, N_ROWS, PARTS)

        grid_w_real_points = ip.create_new_grid(img, grid_manual)

        rearranged_grid = ip.rearrange_grid(grid_w_real_points)

        # Remove points based on least square fit and calculate linear dependency, equations, and coefficients
        grid_remove_points = ip.process_grid(
            rearranged_grid,
            X_THRESHOLD=config['X_THRESHOLD'],
            X1_THRESHOLD=config['X1_THRESHOLD'],
            N_ROWS=config['N_ROWS'],
            PARTS=config['PARTS']
        )  # X_THRESHOLD = 4, X1_THRESHOLD = 40, constants set based on observation for better effectivity

        nan_count = np.isnan(grid_remove_points[:, :, 1]).sum()
        n_rows, n_columns, _ = grid_remove_points.shape
        if nan_count >= ((n_rows * n_columns) / minimum_detected_points):
            error_message = "Not enough points detected."
            log_error(folder_name, error_message, out_folder)
        else:
            average_distance = ip.calculate_average_vertical_distance(
                grid_remove_points, N_ROWS)

            grid_final = ip.add_points_parts(
                average_distance, grid_remove_points, N_ROWS, PARTS)
            
            if VERSION == 'S':
                save_pics(grid_final, img, folder_name, out_folder)
                # Write the results to CSV files
                ip.calculate_distance_and_save_small(
                    grid_final, folder_name, out_folder)
            elif VERSION == 'M':
                grid_bigger_empty = ip.empty_grid(grid_final, N_ROWS)

                grid_bigger_empty = ip.add_points_full_grid(
                    average_distance, grid_bigger_empty, N_ROWS)

                grid_bigger_full = ip.create_new_grid(img, grid_bigger_empty)

                grid_final_final = ip.add_points_full_grid(
                    average_distance, grid_bigger_full, N_ROWS)

                save_pics(grid_final_final, img, folder_name, out_folder)

                # Write the results to CSV files
                ip.calculate_distance_and_save_big(
                    grid_final_final, folder_name, out_folder)
            else:
                error_message = "Wrong version."
                folder_name = "Version"
                log_error(folder_name, error_message, out_folder)

    except Exception as e:
        error_message = f"Error processing {folder_name}: {str(e)}"
        traceback.print_exc()  
        log_error(folder_name, error_message, out_folder)
        # print(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data in a specified directory")
    parser.add_argument("data_root", type=str,
                        help="Path to the data root directory")
    parser.add_argument("--x1-thr", dest='x1_thr', type=int,
                        default=40, help="X1_THRESHOLD value")
    parser.add_argument("--x-threshold", dest='x_threshold',
                        type=int, default=4, help="X_THRESHOLD value")
    parser.add_argument("--n-rows", dest='n_rows', type=int,
                        default=11, help="N_ROWS value")
    parser.add_argument("--n-parts", dest='n_parts',
                        type=int, default=3, help="PARTS value")
    parser.add_argument("--version", dest='version',
                        type=str, default='M',choices=['M', 'S'], help="M=middle, S=Small")

    args = parser.parse_args()
    config = {
        'X1_THRESHOLD': args.x1_thr,
        'X_THRESHOLD': args.x_threshold,
        'N_ROWS': args.n_rows,
        'PARTS': args.n_parts,
        'VERSION': args.version,
    }

    main(args.data_root, config)