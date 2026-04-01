import logging
import logging.config
from app.core import logging_config

# Run setup
logging_config.setup_logging()

# Get logger
logger = logging.getLogger("app")
