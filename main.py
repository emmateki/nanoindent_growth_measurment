import pathlib
import indentation_reader as indentation_reader
import image_processing as ip
import argparse
import os
import numpy as np
import traceback
from shared import log_error
from shared import configure_logger

def main(data_root, config, out_folder=None):
    data_root_path = pathlib.Path(data_root)
    if out_folder is None:
        out_folder = data_root_path.parent / "OUT"
    else:
        out_folder = pathlib.Path(out_folder)
    
    os.makedirs(out_folder, exist_ok=True)
    configure_logger(out_folder)
    df = indentation_reader.read_data(data_root_path)
    folder_names = [
        folder.name for folder in data_root_path.iterdir() if folder.is_dir()]

    for j in range(len(df)):
        if not df.empty:
            row = df.iloc[j]
            folder_name = folder_names[j]

            if folder_name:
                
                processing(row.img_before, folder_name, out_folder, config,is_after=False)
                processing(row.img_after, folder_name, out_folder, config,is_after= True)
            else:
                error_message = "Folder is empty."
                folder_name = "empty"
                log_error(folder_name, error_message)


def processing(img, folder_name, out_folder, config, is_after,minimum_detected_points=0.75):
    """
    Process image data and save results.

    Args:
        img: Original image data.
        folder_name: Name of the folder where the image is located.
        data_root_path: Path to the root data directory.

    """

    try:
        row_in_part = config['row_in_part']
        parts = config['parts']
        version = config['version']

        # coordinates x1,x2,y1,y2 of the region where the first point is located
        filtered_centers1_original = ip.find_origin_start(
            img, 450, 720, 500, 820)
        filtered_centers_last_original = ip.find_origin_last(
            img, 400, 700, 28320, 28650)
        if version == 'M':
            filtered_centers_m_original = ip.find_origin_middle(
                img, 450, 720, 14500, 14800)
        elif version == 'S':
            filtered_centers_m_original = [0, 0]
        grid_manual = ip.manual_grid(
            filtered_centers_m_original, filtered_centers1_original, filtered_centers_last_original, row_in_part, parts)

        grid_w_real_points = ip.create_new_grid(img, grid_manual)

        rearranged_grid = ip.rearrange_grid(grid_w_real_points)

        # Remove points based on least square fit and calculate linear dependency, equations, and coefficients
        grid_remove_points = ip.process_grid(
            rearranged_grid,
            x_treshold=config['x_treshold'],
            x1_treshold=config['x1_treshold'],
            row_in_part=config['row_in_part'],
            parts=config['parts']

        )

        nan_count = np.isnan(grid_remove_points[:, :, 1]).sum()
        n_rows, n_columns, _ = grid_remove_points.shape
        if nan_count >= ((n_rows * n_columns) / minimum_detected_points):
            error_message = "Not enough points detected."
            log_error(folder_name, error_message)
        else:
            average_distance = ip.calculate_average_vertical_distance(
                grid_remove_points, row_in_part)
            
            grid_final = ip.add_points_parts(
                average_distance, grid_remove_points, row_in_part, parts)

            if version == 'S':
                ip.save_pics(grid_final, img, folder_name, out_folder,is_after)
                # Write the results to CSV files
                ip.calculate_distance_and_save_small(
                    grid_final, folder_name, out_folder)
                
            elif version == 'M':
                grid_bigger_empty = ip.empty_grid(grid_final, row_in_part)

                grid_bigger_empty = ip.add_points_full_grid(
                    average_distance, grid_bigger_empty, row_in_part)

                grid_bigger_full = ip.create_new_grid(img, grid_bigger_empty)

                grid_final_final = ip.add_points_full_grid(
                    average_distance, grid_bigger_full, row_in_part)

                ip.save_pics(grid_final_final, img, folder_name, out_folder,is_after)

                # Write the results to CSV files
                ip.calculate_distance_and_save_big(
                    grid_final_final, folder_name, out_folder)
            else:
                error_message = "Wrong version."
                folder_name = "Version"
                log_error(folder_name, error_message)

    except Exception as e:
        error_message = f"Error processing {folder_name}: {str(e)}"
        traceback.print_exc()
        log_error(folder_name, error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data in a specified directory")
    parser.add_argument("data_root", type=str,
                        help="Path to the data root directory")
    parser.add_argument("--x1-thr", dest='x1_thr', type=int,
                        default=40, help="filter points based on the x-coordinate deviation from the least square method")
    parser.add_argument("--x-threshold", dest='x_threshold',
                        type=int, default=4, help="filter points based on the x-coordinate deviation from the median")
    parser.add_argument("--row-in-part", dest='row_in_part', type=int,
                        default=11, help="number of rows in one part")
    parser.add_argument("--n-parts", dest='n_parts',
                        type=int, default=3, help="number of parts that the grid will be split")
    parser.add_argument("--version", dest='version',
                        type=str, default='M', choices=['M', 'S'], help="M=middle, S=Small")

    args = parser.parse_args()
    config = {
        'x1_treshold': args.x1_thr,
        'x_treshold': args.x_threshold,
        'row_in_part': args.row_in_part,
        'parts': args.n_parts,
        'version': args.version,
    }

    main(args.data_root, config)
