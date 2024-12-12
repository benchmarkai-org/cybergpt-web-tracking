import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from cybergpt.datasets.utils import normalise_domain


class Loader:
    def __init__(self):
        pass


class WebTrackingLoader(Loader):
    """Loader for German Web Tracking dataset."""

    def __init__(self, data_csv: str):
        self.user_data = self.load_user_data_from_csv(data_csv)

    @staticmethod
    def preprocess_user_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Load and preprocess the browsing data."""
        df["timestamp"] = pd.to_datetime(df["used_at"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["date"] = df["timestamp"].dt.date
        df["domain"] = df["domain"].apply(normalise_domain)
        cols = ["timestamp", "hour", "day_of_week", "date", "domain", "active_seconds"]
        return df[cols].sort_values(by="timestamp")

    @staticmethod
    def load_user_data_from_csv(data_csv: str) -> dict[str, pd.DataFrame]:
        """Load and preprocess CSV containing users' browsing data."""
        df = pd.read_csv(data_csv)
        user_data = {}

        users = df["panelist_id"].unique()
        print("Loading and preprocessing user data...")
        for user in tqdm(users):
            user_df = df[df["panelist_id"] == user].copy()
            user_data[user] = WebTrackingLoader.preprocess_user_frame(user_df)

        print(f"Loaded data for {len(user_data)} users")
        return user_data


class AliceLoader(Loader):
    """Loader for Catch Me If You Can dataset."""

    def __init__(self, data_directory: str):
        self.user_data = self.load_user_data_from_directory(data_directory)

    @staticmethod
    def preprocess_user_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Load and preprocess the browsing data."""
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["date"] = df["timestamp"].dt.date
        df["domain"] = df["site"].apply(normalise_domain)
        return df

    @staticmethod
    def load_user_data_from_directory(data_directory: str) -> dict[str, pd.DataFrame]:
        """Load and preprocess all user CSV files in directory."""
        data_dir = Path(data_directory)
        user_data = {}

        print("Loading and preprocessing user data...")
        for file_path in data_dir.glob("*.csv"):
            user_id = file_path.stem
            df = pd.read_csv(file_path)
            user_data[user_id] = AliceLoader.preprocess_user_frame(df)

        print(f"Loaded data for {len(user_data)} users")
        return user_data
