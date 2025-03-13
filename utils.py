import os

def ensure_model_path(model_type):
    """Create model directory structure if it doesn't exist."""
    path = f"models/{model_type}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_model_path(model_type, filename="model", include_timestamp=False):
    """Get standardized path for model files."""
    directory = ensure_model_path(model_type)
    
    if include_timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{filename}_{timestamp}"
    
    extension = ".pth" if model_type == "dqn" else ".pkl"
    return os.path.join(directory, f"{filename}{extension}")

def get_plot_path(model_type, plot_name="training_progress"):
    """Get standardized path for plot files."""
    directory = ensure_model_path(model_type)
    return os.path.join(directory, f"{plot_name}.png")