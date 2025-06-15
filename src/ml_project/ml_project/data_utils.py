import os
import pandas as pd
from ament_index_python.packages import get_package_share_directory

def load_dataset(csv_name: str) -> pd.DataFrame:
    share_dir = get_package_share_directory('ml_project')
    csv_path = os.path.join(share_dir, 'data', csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Couldn’t find {csv_name} in {csv_path}")
    return pd.read_csv(csv_path)
