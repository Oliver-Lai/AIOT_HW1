# Utility Functions
# Logging and helper functions

import os
import datetime
from typing import Any

def log_prompt(prompt_text: str, log_file: str = "logs/prompts.log") -> None:
    """
    Log a prompt with timestamp to the specified log file
    
    Parameters:
    -----------
    prompt_text : str
        The prompt text to log
    log_file : str
        Path to the log file (relative to project root)
    """
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format log entry
    log_entry = f"[{timestamp}] {prompt_text}\n"
    
    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def read_prompts_log(log_file: str = "logs/prompts.log") -> list:
    """
    Read all prompts from the log file
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
        
    Returns:
    --------
    list
        List of log entries
    """
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, "r", encoding="utf-8") as f:
        return f.readlines()

def clear_prompts_log(log_file: str = "logs/prompts.log") -> None:
    """
    Clear the prompts log file
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    """
    if os.path.exists(log_file):
        open(log_file, "w").close()

def format_number(number: float, decimal_places: int = 4) -> str:
    """
    Format a number to a specified number of decimal places
    
    Parameters:
    -----------
    number : float
        Number to format
    decimal_places : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number string
    """
    return f"{number:.{decimal_places}f}"

def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible
    
    Parameters:
    -----------
    file_path : str
        Path to validate
        
    Returns:
    --------
    bool
        True if path is valid and accessible
    """
    try:
        return os.path.exists(file_path) and os.access(file_path, os.R_OK)
    except:
        return False

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)

def get_project_info() -> dict:
    """
    Get basic project information
    
    Returns:
    --------
    dict
        Dictionary containing project metadata
    """
    return {
        "project_name": "AutoDeployLR",
        "framework": "CRISP-DM",
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "description": "Automated Linear Regression Deployment following CRISP-DM methodology"
    }