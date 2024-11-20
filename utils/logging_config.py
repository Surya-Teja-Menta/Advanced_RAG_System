import logging
import os

def setup_logging(log_level=logging.INFO, log_dir='logs'):
    """
    Set up logging configuration.
    
    Args:
        log_level (int): Logging level
        log_dir (str): Directory for log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'rag_system.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)