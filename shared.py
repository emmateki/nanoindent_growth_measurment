import logging
import os

def configure_logger(folder_name, out_folder):
    log_folder = out_folder / "ERROR"
    os.makedirs(log_folder, exist_ok=True)
    log_file = log_folder / f"{folder_name}_error.log"

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
