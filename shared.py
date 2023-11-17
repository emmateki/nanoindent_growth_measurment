import logging
import os

def configure_logger(out_folder):
    log_folder = out_folder / "ERROR"
    os.makedirs(log_folder, exist_ok=True)

    log_file = log_folder / "error.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def log_error(folder_name, error_message):
    """
    Log an error message using the configured logger.
    Args:
        folder_name: Name of the folder where the error occurred.
        error_message: Error message to log.
    """
    logging.error(f"[{folder_name}] - {error_message}")
