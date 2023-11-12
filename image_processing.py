import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

AVERAGE_DISTANCE_VERTICAL = 302  # set by previous observation
# Given distances between columns and points
COLUMN_DISTANCE = [300.52, 300.55]
POINT_DISTANCE = [302.14, 302.1, 302.14]

def get_indent_mask(img, threshold_h, threshold_l, close_size=41):
    # Apply preprocessing to highlight indents
    img_highlight = _suppress_non_grid_artifacts(img, threshold_h)
    mask = _segment_indents(img_highlight, threshold_l)
    # Morphological closing to enhance indent shapes
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        _get_diamond(close_size)
    ).astype(np.uint8)


def _subtract_background(img, background_sigma=151):

    img_bck = cv2.GaussianBlur(
        img,
        (background_sigma, background_sigma),
        0
    )
    res = img.astype(float) - img_bck
    res[res < 0] = 0
    return res


def _get_component_centers(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for c in contours:
        # Calculate the moments of the contour
        M = cv2.moments(c)

        if M["m00"] == 0:
            # Avoid division by zero
            cX, cY = np.nan, np.nan
        else:
            # Calculate the centroid
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        centers.append((cX, cY))

    return np.array(centers)


def _segment_indents(img, threshold, kernel_size=11):
    _, thr = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    kernel = _get_diamond(kernel_size)
    return cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel).astype(np.uint8)


def norm(img, max_val=255):

    mi, ma = np.min(img), np.max(img)
    if mi == ma:
        return img * max_val
    return (img - mi) / (ma - mi) * max_val


def _get_diamond(n):
    kernel = np.zeros((n, n), dtype=np.uint8)
    center = np.array([n // 2] * 2)

    xy = [(x, y) for x in range(n) for y in range(n)]
    for x, y in xy:
        p = np.array([x, y])
        if np.sum(np.abs(center - p)) <= n // 2:
            kernel[y, x] = 1

    return kernel


def _get_heatmap_height(
    mask,
    sigma_foreground=21,
    sigma_background=201
):
    _, w = mask.shape
    height_profile = np.sum(mask, axis=1)
    height_blur = gaussian_filter1d(height_profile, sigma_foreground)
    height_blur_bg = gaussian_filter1d(height_profile, sigma_background)
    hb = height_blur - height_blur_bg
    return np.vstack([hb] * w).T


def _calculate_candidates_heatmap(
    img,
    segment_threshold=40,
    sigma_foreground=21,
    sigma_background=201
):
    mask = _segment_indents(img, segment_threshold).astype(float)
    candidates_height = _get_heatmap_height(mask)
    candidates_width = _get_heatmap_height(mask.T).T

    heatmap = candidates_height.astype(float) * candidates_width
    res = norm(heatmap).astype(np.uint8)

    return res


def _suppress_non_grid_artifacts(img, threshold):
    # Normalize image and calculate candidate heatmap
    img_no_bck = norm(_subtract_background(img))
    candidates_heatmap = _calculate_candidates_heatmap(img_no_bck, threshold)
    return norm(img_no_bck * candidates_heatmap).astype(np.uint8)


def rearrange_grid(grid_w_real_points):
    """
    Rearrange the grid based on matching y-coordinates.

    Parameters:
    - grid_w_real_points (ndarray): A 3D array representing the grid with real points.

    Returns:
    - numpy ndarray: Rearranged grid with matched  x and y-coordinates.
    """

    n_rows, _, _ = grid_w_real_points.shape
    rearranged_grid = np.full_like(grid_w_real_points, np.nan)

    middle_column = grid_w_real_points[:, 1, :]
    right_column = grid_w_real_points[:, 0, :]
    left_column = grid_w_real_points[:, 2, :]

    rearranged_grid[:, 1, :] = middle_column

    for row in range(n_rows):
        target_y = middle_column[row, 1]
        y_min, y_max = target_y - 20, target_y + 20

        matching_indices_right = np.argwhere(
            (right_column[:, 1] >= y_min) & (right_column[:, 1] <= y_max))

        matching_indices_left = np.argwhere(
            (left_column[:, 1] >= y_min) & (left_column[:, 1] <= y_max))

        if matching_indices_right.size > 0:
            rearranged_grid[row, 0, :] = grid_w_real_points[matching_indices_right[0, 0], 0, :]
        if matching_indices_left.size > 0:
            rearranged_grid[row, 2, :] = grid_w_real_points[matching_indices_left[0, 0], 2, :]

    return rearranged_grid


def process_grid(rearranged_grid, X_THRESHOLD, X1_THRESHOLD, N_ROWS, PARTS):
    """
    Process a rearranged grid to remove outliers using the least square method.

    Parameters:
    - rearranged_grid (ndarray): A 3D array representing the rearranged grid with real points.
    - X_THRESHOLD: The threshold for filtering points based on x-coordinate deviation from the median.
    - X1_THRESHOLD: The threshold for filtering points based on x1_distance, used in the least square method.
    - N_ROWS: Number of rows in one PART
    - PARTS: Number of parts to process in the grid.

    Returns:
    - ndarray: Grid with outliers removed.
    - ndarray: Coefficients of linear regression for each column in the grid.
    """
    n_rows, n_columns, _ = rearranged_grid.shape
    coefficients = np.zeros((PARTS, n_columns, 2))
    for i, col in [(i, col) for i in range(PARTS) for col in range(n_columns)]:

        x_coordinates, y_coordinates = get_coordinates(
            rearranged_grid, i, col, N_ROWS, PARTS)

        valid_indices = ~np.isnan(x_coordinates)
        x_valid, y_valid = x_coordinates[valid_indices], y_coordinates[valid_indices]
        
        x_median = np.median(x_valid)

        # Filter points based on x_threshold
        valid_mask = np.abs(x_valid - x_median) <= X_THRESHOLD
        x_filtered, y_filtered = x_valid[valid_mask], y_valid[valid_mask]
        # Perform linear regression
        A = np.vstack([x_filtered, np.ones(len(x_filtered))]).T
        m, c = np.linalg.lstsq(A, y_filtered, rcond=None)[0]

        coefficients[i, col, 0] = m
        coefficients[i, col, 1] = c

        update_grid(rearranged_grid, x_coordinates, y_coordinates,
                    valid_indices, X1_THRESHOLD, m, c, i, N_ROWS, col)
        
    return rearranged_grid


def get_coordinates(grid, part, col, n_rows, parts):
    if parts == 2:
        part = 1
    row_start = part * n_rows
    row_end = row_start + n_rows
    x_coordinates = grid[row_start:row_end, col, 0]
    y_coordinates = grid[row_start:row_end, col, 1]
    return x_coordinates, y_coordinates


def update_grid(grid, x_coordinates, y_coordinates, valid_indices, X1_THRESHOLD, m, c, part, n_rows, col):
    row_start = part * n_rows
    row_end = row_start + n_rows

    for row in range(row_start, row_end):
        if valid_indices[row - row_start]:
            expected_x = (y_coordinates[row - row_start] - c) / m
            x1_distance = np.abs(x_coordinates[row - row_start] - expected_x)

            if x1_distance > X1_THRESHOLD:
                grid[row, col, :] = np.nan

    return grid


def calculate_average_vertical_distance(grid_remove_points, N_ROWS):
    n_rows, n_columns, _ = grid_remove_points.shape
    total_distance = 0
    count = 0
    for col in range(0,n_columns):

        y_coordinates = grid_remove_points[:N_ROWS, col, 1]
        valid_indices = ~np.isnan(y_coordinates)

        if np.sum(valid_indices) >= 2:
            for row in range(1, N_ROWS):
                if valid_indices[row - 1] and valid_indices[row]:
                    distance = abs(y_coordinates[row] - y_coordinates[row - 1])
                    total_distance += distance
                    count += 1

    if count > 0:
        average = total_distance/count
    total_distance = 0
    count = 0
    
    for col in range(n_columns):

        y_coordinates = grid_remove_points[N_ROWS:N_ROWS*2, col, 1]        
        valid_indices = ~np.isnan(y_coordinates)

        # Check if there are at least two valid neighboring points in the column
        if np.sum(valid_indices) >= 2:
            for row in range(1, N_ROWS):
                if valid_indices[row - 1] and valid_indices[row]:
                    distance = abs(y_coordinates[row] - y_coordinates[row - 1])
                    total_distance += distance
                    count += 1
    if count > 0:
        average1 = total_distance/count
    
    return ((average+average1)/2)


def calculate_distance_and_save_small(new_grid1, folder_name, data_root_path):
    """
    Calculate and store distance metrics in an Excel file for specific grid points.

    This function calculates distance metrics for specific grid points within grid and stores
    the results in an Excel file. The function computes the distances between these points and
    appends the calculated metrics to an existing or new Excel file.

    Parameters:
    new_grid1 (numpy.ndarray): The input grid data containing coordinates.
    folder_name (str): The name of the folder to save the output Excel file.
    data_root_path (str): The root path where the Excel file will be saved.

    Returns:
    None

    The function performs the following calculations:
    1. Calculates distances between  1 and last point in every col.
    2. Calculates distances between  2 and last point+1 in every col.
    3. Calculates distances between  3 and last point+2 in every col.


    Note:
    - The points considered for distance calculations are within the 3x3 grid, and distances are
      converted to millimeters and scaled by a factor of 0.2. known from previous calculations
    """
    n_rows, n_columns, _ = new_grid1.shape

    lengths_in_mm = []
    for i, j in [(i, j) for i in range(3) for j in range(3)]:

        x1 = new_grid1[j, i, 0]
        x2 = new_grid1[n_rows - j - 1, i, 0]
        y1 = new_grid1[j, i, 1]
        y2 = new_grid1[n_rows - j - 1, i, 1]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lengths_in_mm.append(length / 60 * (0.2))

    data_after = {
        "Points": ["Col_0", "Col_1", "Col_2", " "],
        "2_94": [lengths_in_mm[0], lengths_in_mm[3], lengths_in_mm[6], None],
        "95_187": [lengths_in_mm[1], lengths_in_mm[4], lengths_in_mm[7], None],
        "190_282": [lengths_in_mm[2], lengths_in_mm[5], lengths_in_mm[8], None],
    }
    data_before = {
        "Points": ["Col_0", "Col_1", "Col_2", " "],
        "2_94": [lengths_in_mm[0], lengths_in_mm[3], lengths_in_mm[6], "After"],
        "95_187": [lengths_in_mm[1], lengths_in_mm[4], lengths_in_mm[7], None],
        "190_282": [lengths_in_mm[2], lengths_in_mm[5], lengths_in_mm[8], None],
    }

    result_folder = data_root_path / "RESULT"
    os.makedirs(result_folder, exist_ok=True)
    os.chdir(result_folder)

    file_name = f"lengths_{folder_name}.xlsx"
    file_path = os.path.join(result_folder, file_name)

    if os.path.isfile(file_path):

        df_existing = pd.read_excel(file_path)

        df_updated = pd.concat([df_existing, pd.DataFrame(data_after)])

        df_updated.to_excel(file_path, index=False)

        df_done = pd.read_excel(file_path)

        diff_col_0_2_94 = df_done.at[4, "2_94"]-df_done.at[0, "2_94"]
        diff_col_0_95_187 = df_done.at[4, "95_187"]-df_done.at[0, "95_187"]
        diff_col_0_190_282 = df_done.at[4, "190_282"]-df_done.at[0, "190_282"]

        diff_col_1_2_94 = df_done.at[5, "2_94"]-df_done.at[1, "2_94"]
        diff_col_1_95_187 = df_done.at[5, "95_187"]-df_done.at[1, "95_187"]
        diff_col_1_190_282 = df_done.at[5, "190_282"]-df_done.at[1, "190_282"]

        diff_col_2_2_94 = df_done.at[6, "2_94"]-df_done.at[2, "2_94"]
        diff_col_2_95_187 = df_done.at[6, "95_187"]-df_done.at[2, "95_187"]

        diff_col_2_190_282 = df_done.at[6, "190_282"]-df_done.at[2, "190_282"]
        data_diff = {
            "Points": ["Col_0", "Col_1", "Col_2", " "],
            "2_94": [diff_col_0_2_94, diff_col_1_2_94, diff_col_2_2_94, "Difference"],
            "95_187": [diff_col_0_95_187, diff_col_1_95_187, diff_col_2_95_187, None],
            "190_282": [diff_col_0_190_282, diff_col_1_190_282, diff_col_2_190_282, None],
        }
        df_existing = pd.read_excel(file_path)
        df_updated = pd.concat([df_existing, pd.DataFrame(data_diff)])

        df_updated.to_excel(file_path, index=False)

    else:
        df = pd.DataFrame(data_before)
        df.to_excel(file_path, index=False)


def calculate_distance_and_save_big(new_grid1, folder_name, data_root_path):
    """
    Calculate distance metrics and store them in an Excel file.

    Function calculates average lengths and percentage increases based on the given grid data. 
    It then stores the results in an Excel file.

    Parameters:
    new_grid1 (numpy.ndarray): The input grid data containing coordinates.
    folder_name (str): The name of the folder to save the output Excel file.
    data_root_path (str): The root path where the Excel file will be saved.

    Returns:
    None

    The function performs the following calculations:
    1. Calculates the average vertical length for all points.
    2. Calculates the average length for the second column-this one is with the least disturbances.
    3. Calculates the average length for each adjacent pair of columns-horizontal lenght.
    """

    n_rows, n_columns, _ = new_grid1.shape
    average_length_mm, average_length, average_length1, average_length2 = 0, 0, 0, 0
    count, count1, count2 = 0, 0, 0
    total_length, total_length1, total_length2 = 0, 0, 0
    length, length1, length2 = 0, 0, 0
    for col, row in [(col, row) for col in range(n_columns) for row in range(n_rows-1)]:
        if not np.isnan(new_grid1[row, col, 1]) and not np.isnan(new_grid1[row + 1, col, 1]):
            x1 = new_grid1[row, col, 0]
            x2 = new_grid1[row + 1, col, 0]
            y1 = new_grid1[row, col, 1]
            y2 = new_grid1[row + 1, col, 1]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            total_length += length
            count += 1

    average_length = total_length / count
    average_length_mm = average_length / 60 * (0.2)

    for row in range(0, n_rows - 1):
        if not np.isnan(new_grid1[row, 1, 1]) and not np.isnan(new_grid1[row + 1, 1, 1]):
            x1 = new_grid1[row, 1, 0]
            x2 = new_grid1[row + 1, 1, 0]
            y1 = new_grid1[row, 1, 1]
            y2 = new_grid1[row + 1, 1, 1]
            length1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            total_length1 += length1
            count1 += 1

    average_length1 = total_length1 / count1
    average_length_mm1 = average_length1 / 60 * (0.2)
    for col, row in [(col, row) for col in range(2) for row in range(n_rows-1)]:
        if not np.isnan(new_grid1[row, 1, 1]) and not np.isnan(new_grid1[row + 1, 1, 1]):
            x1 = new_grid1[row, col, 0]
            x2 = new_grid1[row, col+1, 0]
            y1 = new_grid1[row, col, 1]
            y2 = new_grid1[row, col+1, 1]
            length2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            total_length2 += length2
            count2 += 1

    average_length2 = total_length2 / count2
    average_length_mm2 = average_length2 / 60 * (0.2)

    data = {
        "Points": ["All_rows", " "],
        "All_Coll": [average_length_mm, None],
        "Col_1": [average_length_mm1, None],
        "Hor": [average_length_mm2, None],
    }

    result_folder = data_root_path / "RESULT"
    os.makedirs(result_folder, exist_ok=True)
    os.chdir(result_folder)

    file_name = f"lengths_{folder_name}.xlsx"
    file_path = os.path.join(result_folder, file_name)

    if os.path.isfile(file_path):

        df_existing = pd.read_excel(file_path)

        df_updated = pd.concat([df_existing, pd.DataFrame(data)])

        df_updated.to_excel(file_path, index=False)

        df_done = pd.read_excel(file_path)

        diff_all_col = (
            df_done.at[2, "All_Coll"]-df_done.at[0, "All_Coll"])/df_done.at[0, "All_Coll"]*100
        diff_1_col = (df_done.at[2, "Col_1"]-df_done.at[0,
                      "Col_1"])/df_done.at[0, "Col_1"]*100
        diff_Vertic = (df_done.at[2, "Hor"] -
                       df_done.at[0, "Hor"])/df_done.at[0, "Hor"]*100
        data_diff = {
            "Points": ["All_rows", " "],
            "All_Coll": [diff_all_col, None],
            "Col_1": [diff_1_col, None],
            "Hor": [diff_Vertic, None],
        }
        df_existing = pd.read_excel(file_path)
        df_updated = pd.concat([df_existing, pd.DataFrame(data_diff)])

        df_updated.to_excel(file_path, index=False)

    else:

        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)


def find_origin_last(img):
    filtered_centers_last = np.array([])
    X_START, X_END, Y_START, Y_END = 400, 700, 28320, 28650
    cropped = img[Y_START:Y_END, X_START:X_END]
    threshold = 20
    img_highlight = _suppress_non_grid_artifacts(cropped, threshold)

    threshold_h = 15
    threshold_l = 15
    mask1 = np.array([])
    mask1 = get_indent_mask(img_highlight, threshold_h, threshold_l)

    filtered_centers_last = _get_component_centers(mask1)

    filtered_centers_last = np.array(filtered_centers_last)

    if filtered_centers_last.size == 0 or filtered_centers_last.size >= 4:
        filtered_centers_last = np.array([[548, 28581]], dtype=np.int32)
    else:
        # Transform the filtered coordinates back to the original image's coordinate system
        filtered_centers_last = filtered_centers_last + \
            np.array([X_START, Y_START])
    return filtered_centers_last


def find_origin_start(img):
    filtered_centers1 = np.array([])
    X_START, X_END, Y_START, Y_END = 450, 720, 500, 820
    cropped = img[Y_START:Y_END, X_START:X_END]
    threshold = 20
    img_highlight = _suppress_non_grid_artifacts(cropped, threshold)

    threshold_h = 15
    threshold_l = 15
    mask1 = np.array([])
    mask1 = get_indent_mask(img_highlight, threshold_h, threshold_l)

    filtered_centers1 = _get_component_centers(mask1)

    filtered_centers1 = np.array(filtered_centers1)
    if filtered_centers1.size == 0 or filtered_centers1.size >= 4:
        filtered_centers1_original = np.array([[548, 784]], dtype=np.int32)
    else:
        # Transform the filtered coordinates back to the original image's coordinate system
        filtered_centers1_original = filtered_centers1 + \
            np.array([X_START, Y_START])
    return filtered_centers1_original


def find_origin_middle(img):

    filtered_centers_m = np.array([])# region where the middle points is most of the time located
    X_START, X_END, Y_START, Y_END = 450, 720, 14500, 14800

    cropped = img[Y_START:Y_END, X_START:X_END]

    threshold = 50
    img_highlight = _suppress_non_grid_artifacts(cropped, threshold)

    threshold_h = 15
    threshold_l = 15
    mask1 = np.array([])
    mask1 = get_indent_mask(img_highlight, threshold_h, threshold_l)

    filtered_centers_m = _get_component_centers(mask1)

    filtered_centers_m_original = np.array([])
    filtered_centers_m = np.array(filtered_centers_m)

    if filtered_centers_m.size == 0 or filtered_centers_m.size >= 4:
        # If it's empty, set filtered_centers1_original to a specific value
        filtered_centers_m_original = np.array([[606, 14678]], dtype=np.int32)
    else:
        # Transform the filtered coordinates back to the original image's coordinate system
        filtered_centers_m_original = filtered_centers_m + \
            np.array([X_START, Y_START])

    return filtered_centers_m_original


def manual_grid(filtered_centers_m_original, filtered_centers1_original, filtered_centers_last_original, N_ROWS, parts):
    grid_manual = []

    for i, j in [(i, j) for i in range(1) for j in range(N_ROWS)]:
        col_x = filtered_centers1_original[0, 0] + sum(COLUMN_DISTANCE[:i])
        point_y = filtered_centers1_original[0, 1] + j * POINT_DISTANCE[i]
        left_x = col_x - POINT_DISTANCE[i]
        middle_x = col_x
        right_x = col_x + POINT_DISTANCE[i]
        grid_manual.append(
            [(left_x, point_y), (middle_x, point_y), (right_x, point_y)])

    if parts == 3:
        for i, j in [(i, j) for i in range(1) for j in range(N_ROWS)]:
            col_x = filtered_centers_m_original[0,
                                                0] + sum(COLUMN_DISTANCE[:i])
            point_y = filtered_centers_m_original[0,
                                                  1] + j * POINT_DISTANCE[i]
            left_x = col_x - POINT_DISTANCE[i]
            middle_x = col_x
            right_x = col_x + POINT_DISTANCE[i]

            grid_manual.append(
                [(left_x, point_y), (middle_x, point_y), (right_x, point_y)])

    for i, j in [(i, j) for i in range(1) for j in range(N_ROWS)]:
        col_x = filtered_centers_last_original[0,
                                               0] + sum(COLUMN_DISTANCE[:i])

        point_y = filtered_centers_last_original[0,
                                                 1] - (N_ROWS-1-j) * POINT_DISTANCE[i]
        left_x = col_x - POINT_DISTANCE[i]
        middle_x = col_x
        right_x = col_x + POINT_DISTANCE[i]

        grid_manual.append(
            [(left_x, point_y), (middle_x, point_y), (right_x, point_y)])

    grid_manual = np.array(grid_manual)
    return grid_manual


def find_real_points_around_point(img, x, y, threshold=20, threshold_h=30, threshold_l=30):

    x_start, x_end = x - 50, x + 50
    y_start, y_end = y - 50, y + 50

    cropped = img[y_start:y_end, x_start:x_end]

    img_highlight = _suppress_non_grid_artifacts(cropped, threshold)

    mask = get_indent_mask(img_highlight, threshold_h, threshold_l)

    centers = _get_component_centers(mask)

    if len(centers) == 0:
        centers_original = np.array([(np.nan, np.nan)])
    elif len(centers) >= 2:
        center_of_cropped = np.array([x_start + 50, y_start + 50])
        distances = np.linalg.norm(centers - center_of_cropped, axis=1)
        closest_center_idx = np.argmin(distances)
        centers1 = centers[closest_center_idx]
        centers_original = np.array([centers1]) + np.array([x_start, y_start])

    else:

        centers1 = np.array(centers)
        centers_original = centers1 + np.array([x_start, y_start])

    return centers_original


def create_new_grid(img, grid_manual):
    grid_w_real_points = np.empty_like(grid_manual)
    grid_w_real_points[:] = np.nan

    real_points = []
    n_rows, n_columns, _ = grid_manual.shape

    for col in range(n_columns):
        for row in range(n_rows):
            if not np.isnan(grid_manual[row, col, 0]) and not np.isnan(grid_manual[row, col, 1]):
                x = int(grid_manual[row, col, 0])
                y = int(grid_manual[row, col, 1])

                real_points_around_point = find_real_points_around_point(
                    img, x, y)
                grid_w_real_points[row, col,
                                   0] = real_points_around_point[0, 0]
                grid_w_real_points[row, col,
                                   1] = real_points_around_point[0, 1]
                if real_points_around_point.shape[0] > 0:
                    real_points.extend(real_points_around_point)

    real_points = np.array(real_points)
    return grid_w_real_points


def calculate_x_coordinate(y, m, c):
    # Assuming y = mx + c, solve for x: x = (y - c) / m
    return (y - c) / m


def empty_grid(grid_final, N_ROWS):
    """
    Expand the input grid with empty rows to fill gaps between existing rows.

    This function takes an input grid and adds empty rows between the rows starting from the specified 'N_ROWS'.
    The number of empty rows added is determined by calculating the average vertical distance between neighboring rows
    and inserting rows accordingly.

    Parameters:
    - grid_final (ndarray): The input grid, which is a 3D numpy array representing the grid with points.
    - N_ROWS (int): The row index from which to start expanding the grid with empty rows.

    Returns:
    - ndarray: The expanded grid with empty rows added to fill the gaps.
    """

    diff = grid_final[N_ROWS, 0, 1]-grid_final[N_ROWS-1, 0, 1]
    no_of_rows = +round(diff / AVERAGE_DISTANCE_VERTICAL - 1)

    diff2 = grid_final[N_ROWS*2, 0, 1]-grid_final[N_ROWS*2-1, 0, 1]

    no_of_rows2 = round(diff2 / AVERAGE_DISTANCE_VERTICAL - 1)

    nan_grid = np.full((no_of_rows, 3, 2), np.nan)
    nan_grid2 = np.full((no_of_rows2, 3, 2), np.nan)
    grid_final_bigger = np.insert(grid_final, N_ROWS, nan_grid, axis=0)
    grid_final_bigger = np.insert(
        grid_final_bigger, N_ROWS*2+no_of_rows, nan_grid2, axis=0)
    return grid_final_bigger


def add_points_parts(average_distance, new_grid1, N_ROWS,parts):
    """
    Add missing points to the grid and adjust their positions based on given criteria-linear dependency .

    Parameters:
    - average_distance: The average vertical distance between neighboring points.
    - new_grid1 (ndarray): Array representing the grid with points where some are missing.
    - coefficients (ndarray): Coefficients of linear regression for each column in the grid.

    Returns:
    - ndarray: The grid with added and adjusted points.
    """

    n_rows, n_columns, _ = new_grid1.shape
    i = 0
    nan_count = np.isnan(new_grid1[:, :, 1]).sum()

    def update_missing_points(row_start, row_end, col):
        for row in range(row_start, row_end):
            if np.isnan(new_grid1[row, col, 1]):
                if row > row_start and not np.isnan(new_grid1[row - 1, col, 1]):
                    new_grid1[row, col, 1] = new_grid1[row - 1, col, 1] + average_distance
                elif (row+1) < row_end and not np.isnan(new_grid1[row + 1, col, 1]):
                    new_grid1[row, col, 1] = new_grid1[row + 1, col, 1] - average_distance

    i = 0
    nan_count = np.isnan(new_grid1[:, :, 1]).sum()

    while nan_count != 0 and i < n_rows*n_columns:

        for col in range(n_columns):
            update_missing_points(0, N_ROWS, col)
            update_missing_points(N_ROWS, N_ROWS*2, col)

            if parts == 3:
                update_missing_points(N_ROWS*2, n_rows, col)

        nan_count = np.isnan(new_grid1[:, :, 1]).sum()
        i += 1
    nan_indices_x = np.argwhere(np.isnan(new_grid1[..., 0]))
    for col in range(n_columns):
        for row in range(N_ROWS):
            if np.isnan(new_grid1[row, col, 0]) and not np.isnan(new_grid1[row, col, 1]):
                x_median = np.nanmedian(new_grid1[:N_ROWS, col, 0])
                new_grid1[row, col, 0] = x_median

        for row in range(N_ROWS, n_rows):
            if np.isnan(new_grid1[row, col, 0]) and not np.isnan(new_grid1[row, col, 1]):
                x_median = np.nanmedian(new_grid1[N_ROWS:n_rows, col, 0])
                new_grid1[row, col, 0] = x_median

    return new_grid1


def add_points_full_grid(average_distance, grid_bigger_empty, N_ROWS):
    n_rows, _, _ = grid_bigger_empty.shape
    i = 0
    nan_indices_y = np.argwhere(np.isnan(grid_bigger_empty[..., 1]))
    nan_count = np.isnan(grid_bigger_empty[:, :, 1]).sum()
    while nan_count != 0 or i < 400:
        for idx in nan_indices_y:
            row, col = idx
            if not np.isnan(grid_bigger_empty[row - 1, col, 1]) and not row - 1 < 0:
                grid_bigger_empty[row, col, 1] = grid_bigger_empty[row - 1, col, 1] + average_distance
            elif row < n_rows - 1 and not np.isnan(grid_bigger_empty[row + 1, col, 1]):
                grid_bigger_empty[row, col, 1] = grid_bigger_empty[row + 1, col, 1] - average_distance

        nan_count = np.isnan(grid_bigger_empty[:, :, 1]).sum()
        i += 1
    nan_indices_x = np.argwhere(np.isnan(grid_bigger_empty[..., 0]))
    for idx in nan_indices_x:
        row, col = idx
        if not np.isnan(grid_bigger_empty[row, col, 1]):
            if row < N_ROWS:
                x_median = np.nanmedian(grid_bigger_empty[0:N_ROWS, col, 0])
            elif row > (N_ROWS*2):
                x_median = np.nanmedian(grid_bigger_empty[N_ROWS:N_ROWS*2, col, 0])
            else:
                x_median = np.nanmedian(grid_bigger_empty[N_ROWS:n_rows, col, 0])
            grid_bigger_empty[row, col, 0] = x_median

    return grid_bigger_empty
