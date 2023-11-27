import pathlib
import indentation_reader as indentation_reader
import image_processing as ip
import argparse
import os
import numpy as np
import traceback
import logging
import config as cfg


def main(data_root, config, out_folder=None):
    data_root_path = pathlib.Path(data_root)
    if not any(data_root_path.iterdir()):
        raise Exception("Data root path is empty.")

    if out_folder is None:
        out_folder = data_root_path.parent / "OUT"
    else:
        out_folder = pathlib.Path(out_folder)

    os.makedirs(out_folder, exist_ok=True)
    log_file_path = out_folder / "ERROR" / "error.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING,
        filename=log_file_path,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    df = None
    try:
        df = indentation_reader.read_data(data_root_path)
    except Exception as e:
        logging.error(f"{e}")
        raise Exception(f"{e}")

    for index, row in df.iterrows():
        folder_name = row.folder_name
        if folder_name:
            processing(row.img_before, folder_name, out_folder, config, is_after=False)
            processing(row.img_after, folder_name, out_folder, config, is_after=True)
        else:
            logging.error(f"[{folder_name}] - Folder is empty.")


def processing(
    img, folder_name, out_folder, config, is_after, minimum_detected_points=0.75
):
    """
    Process image data and save results.

    Args:
        img: Original image data.
        folder_name: Name of the folder where the image is located.
        data_root_path: Path to the root data directory.

    """

    try:
        row_in_part = config["row_in_part"]
        parts = config["parts"]
        version = config["version"]

        filtered_centers1_original = ip.find_origin_start(
            img,
            config["filter_x1"],
            config["filter_x2"],
            config["filter_y1"],
            config["filter_y2"],
        )
        filtered_centers_last_original = ip.find_origin_last(
            img,
            config["last_x1"],
            config["last_x2"],
            config["last_y1"],
            config["last_y2"],
        )

        # coordinates x1,x2,y1,y2 of the region where the first point is located
        if version == "M":
            filtered_centers_m_original = ip.find_origin_middle(
                img,
                config["middle_x1"],
                config["middle_x2"],
                config["middle_y1"],
                config["middle_y2"],
            )
        elif version == "S":
            filtered_centers_m_original = [0, 0]

        grid_manual = ip.manual_grid(
            filtered_centers_m_original,
            filtered_centers1_original,
            filtered_centers_last_original,
            row_in_part,
            parts,
        )

        grid_w_real_points = ip.create_new_grid(img, grid_manual)

        rearranged_grid = ip.rearrange_grid(grid_w_real_points)

        # Remove points based on least square fit and calculate linear dependency, equations, and coefficients
        grid_remove_points = ip.process_grid(
            rearranged_grid,
            x_treshold=config["x_threshold"],
            x1_treshold=config["x1_thr"],
            row_in_part=config["row_in_part"],
            parts=config["parts"],
        )

        nan_count = np.isnan(grid_remove_points[:, :, 1]).sum()
        n_rows, n_columns, _ = grid_remove_points.shape
        if nan_count >= ((n_rows * n_columns) / minimum_detected_points):
            logging.error(
                f"Error processing {folder_name}: Not enough points detected."
            )
            raise Exception(
                f"Error processing {folder_name}: Not enough points detected."
            )

        else:
            average_distance = ip.calculate_average_vertical_distance(
                grid_remove_points, row_in_part
            )

            grid_final = ip.add_points_parts(
                average_distance, grid_remove_points, row_in_part, parts
            )

            if version == "S":
                ip.save_pics(grid_final, img, folder_name, out_folder, is_after)
                # Write the results to CSV files
                ip.calculate_distance_and_save_small(
                    grid_final, folder_name, out_folder
                )

            elif version == "M":
                grid_bigger_empty = ip.empty_grid(grid_final, row_in_part)

                grid_bigger_empty = ip.add_points_full_grid(
                    average_distance, grid_bigger_empty, row_in_part
                )

                grid_bigger_full = ip.create_new_grid(img, grid_bigger_empty)

                grid_final_final = ip.add_points_full_grid(
                    average_distance, grid_bigger_full, row_in_part
                )

                ip.save_pics(grid_final_final, img, folder_name, out_folder, is_after)

                # Write the results to CSV files
                ip.calculate_distance_and_save_big(
                    grid_final_final, folder_name, out_folder
                )
                n_rows, n_columns, _ = grid_final_final.shape
            else:
                logging.error(f"Error processing {folder_name}: Wrong version.")
                raise Exception(f"Error processing {folder_name}: Wrong version.")

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error processing {folder_name}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data in a specified directory"
    )
    parser.add_argument("data_root", type=str, help="Path to the data root directory")
    parser.add_argument(
        "--x1-thr",
        dest="x1_thr",
        type=int,
        default=40,
        help="filter points based on the x-coordinate deviation from the least square method",
    )
    parser.add_argument(
        "--x-threshold",
        dest="x_threshold",
        type=int,
        default=4,
        help="filter points based on the x-coordinate deviation from the median",
    )
    parser.add_argument(
        "--row-in-part",
        dest="row_in_part",
        type=int,
        default=11,
        help="number of rows in one part",
    )
    parser.add_argument(
        "--n-parts",
        dest="n_parts",
        type=int,
        default=3,
        help="number of parts that the grid will be split",
    )
    parser.add_argument(
        "--version",
        dest="version",
        type=str,
        default="M",
        choices=["M", "S"],
        help="M=middle, S=Small",
    )

    args = parser.parse_args()
    default_config = cfg.get_default_config()
    user_config = {
        "x1_thr": args.x1_thr,
        "x_threshold": args.x_threshold,
        "row_in_part": args.row_in_part,
        "parts": args.n_parts,
        "version": args.version,
    }

    config = default_config | user_config
    main(args.data_root, config)
