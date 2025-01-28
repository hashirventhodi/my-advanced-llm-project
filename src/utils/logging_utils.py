import logging
import os

def setup_logging(logging_config):
    log_dir = logging_config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    logging_level = logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.info(f"Logging initialized. Logs directory: {log_dir}")
