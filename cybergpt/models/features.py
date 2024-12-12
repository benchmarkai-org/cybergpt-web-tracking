import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
import argparse
import os
import pickle
from tqdm.auto import tqdm

from cybergpt.datasets.websites import WebsiteDataset
from cybergpt.datasets.loaders import WebTrackingLoader


class SessionFeatureExtractor:
    social_domains = [
        "facebook",
        "twitter",
        "instagram",
        "linkedin",
        "reddit",
        "pinterest",
        "tiktok",
    ]
    search_domains = ["google", "bing", "yahoo", "duckduckgo"]
    email_domains = ["gmail", "outlook", "yahoo"]
    commerce_domains = list(
        set(
            [
                "argos",
                "tesco",
                "ocado",
                "sainsburys",
                "asda",
                "waitrose",
                "morrisons",
                "boots",
                "superdrug",  # UK commerce
                "amazon",
                "ebay",
                "shopify",
                "etsy",
                "walmart",
                "target",
                "bestbuy",
                "costco",
                "ulta",
                "sephora",
                "macys",
                "nordstrom",  # US commerce
                "otto",
                "zalando",
                "dm",
                "rewe",
                "kaufland",
                "lidl",
                "edeka",
                "netto",  # German commerce
            ]
        )
    )
    news_domains = [
        "cnn",
        "bbc",
        "reuters",
        "news",
        "nytimes",
        "washingtonpost",
        "theguardian",
        "theverge",
        "techcrunch",
        "wired",
        "bloomberg",
        "forbes",
        "wsj",
        "nyt",
    ]

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = None

    def extract_features(
        self,
        dfs: List[pd.DataFrame],
        domain_col: str = "domain",
        timestamp_col: str = "timestamp",
        duration_col: str = "active_seconds",
    ) -> pd.DataFrame:
        """
        Extract features from session data.

        Args:
            df: DataFrame with columns for user_id, session_id, domain, timestamp, active_seconds

        Returns:
            DataFrame with extracted features, one row per session
        """
        features = []

        for df in tqdm(dfs, desc="Extracting features"):
            session_features = {}
            session_features.update(
                self._extract_temporal_features(df[timestamp_col].values)
            )
            session_features.update(
                self._extract_duration_features(df[duration_col].values)
            )
            session_features.update(
                self._extract_domain_features(df[domain_col].values)
            )
            session_features.update(
                self._extract_transition_features(
                    df[[timestamp_col, domain_col, duration_col]].values
                )
            )
            features.append(session_features)

        features_df = pd.DataFrame(features, index=range(len(dfs)))
        self.feature_names = features_df.columns
        return features_df

    def _extract_temporal_features(self, timestamps: np.ndarray) -> Dict:
        """Extract features related to timing of visits."""
        times = pd.to_datetime(timestamps, unit="s")
        time_diffs = (
            np.diff(timestamps).astype(float) if len(timestamps) > 1 else np.array([0])
        )

        time_of_day = times.min().hour
        return {
            "session_start_hour": times.min().hour,
            "session_end_hour": times.max().hour,
            "session_duration_minutes": (times.max() - times.min()).total_seconds()
            / 60,
            "num_visits": len(timestamps),
            "avg_time_between_visits": (
                np.mean(time_diffs) if len(timestamps) > 1 else 0
            ),
            "std_time_between_visits": np.std(time_diffs) if len(timestamps) > 1 else 0,
            "is_weekend": times.min().weekday() >= 5,
            "is_working_hours": (times.min().hour >= 9) & (times.max().hour <= 17),
            "spans_multiple_days": (times.max().date() - times.min().date()).days > 0,
            "session_time_category": (
                "morning"
                if 5 <= time_of_day < 12
                else (
                    "afternoon"
                    if 12 <= time_of_day < 17
                    else "evening" if 17 <= time_of_day < 22 else "night"
                )
            ),
            "day_of_week": times.min().dayofweek,
            "is_weekend": times.min().dayofweek >= 5,
        }

    def _extract_duration_features(self, durations: np.ndarray) -> Dict:
        """Extract features related to time spent on pages."""
        return {
            "total_active_time": np.sum(durations),
            "mean_page_duration": np.mean(durations),
            "std_page_duration": np.std(durations),
            "min_page_duration": np.min(durations),
            "max_page_duration": np.max(durations),
            "median_page_duration": np.median(durations),
            "quick_bounces": np.sum(durations < 10),  # pages viewed < 10 seconds
            "long_views": np.sum(durations > 300),  # pages viewed > 5 minutes
        }

    def _extract_domain_features(self, domains: np.ndarray) -> Dict:
        """Extract features related to domains visited."""
        domain_counts = Counter(domains)

        # Get top level domains
        tlds = [d.split(".")[-1] for d in domains]
        tld_counts = Counter(tlds)

        subdomain_depths = [len(d.split(".")) for d in domains]

        return {
            "unique_domains": len(domain_counts),
            "unique_tlds": len(tld_counts),
            "most_common_domain_freq": (
                domain_counts.most_common(1)[0][1] if domain_counts else 0
            ),
            "most_common_tld_freq": (
                tld_counts.most_common(1)[0][1] if tld_counts else 0
            ),
            "domain_entropy": self._calculate_entropy(list(domain_counts.values())),
            "has_social_media": any(
                domain in str(domains) for domain in self.social_domains
            ),
            "frac_social_media": sum(
                domain in self.social_domains for domain in domains
            )
            / len(domains),
            "has_search": any(domain in str(domains) for domain in self.search_domains),
            "frac_search": sum(domain in self.search_domains for domain in domains)
            / len(domains),
            "has_email": any(domain in str(domains) for domain in self.email_domains),
            "frac_email": sum(domain in self.email_domains for domain in domains)
            / len(domains),
            "has_commerce": any(
                domain in str(domains).lower() for domain in self.commerce_domains
            ),
            "frac_commerce": sum(domain in self.commerce_domains for domain in domains)
            / len(domains),
            "has_news": any(
                domain in str(domains).lower() for domain in self.news_domains
            ),
            "frac_news": sum(domain in self.news_domains for domain in domains)
            / len(domains),
            "domain_return_rate": 1
            - (len(domain_counts) / len(domains)),  # proportion of repeat visits
            "max_subdomain_depth": max(subdomain_depths),
            "avg_subdomain_depth": np.mean(subdomain_depths),
            "domain_diversity_ratio": len(set(domains)) / len(domains),
        }

    def _extract_transition_features(self, visit_data: np.ndarray) -> Dict:
        """Extract features related to transitions between pages."""
        if len(visit_data) <= 1:
            return {
                "avg_transition_time": 0,
                "rapid_transitions": 0,
                "domain_switch_rate": 0,
                "back_forth_patterns": 0,
                "rapid_sequence_changes": 0,
                "avg_dwell_time": 0,
                "frac_time_social_media": 0,
                "frac_time_search": 0,
                "frac_time_email": 0,
                "frac_time_commerce": 0,
                "frac_time_news": 0,
            }

        timestamps = visit_data[:, 0]
        domains = visit_data[:, 1]
        dwell_times = visit_data[:, 2]

        timestamps = (
            pd.to_datetime(timestamps).astype(np.int64) / 1e9
        )  # Convert to seconds
        time_diffs = np.diff(timestamps)

        domain_switches = np.array(
            [domains[i] != domains[i - 1] for i in range(1, len(domains))]
        )

        return {
            "avg_transition_time": np.mean(time_diffs),
            "rapid_transitions": np.sum(time_diffs < 2),
            "domain_switch_rate": np.mean(domain_switches),
            "back_forth_patterns": sum(
                domains[i] == domains[i - 2] and domains[i - 1] != domains[i]
                for i in range(2, len(domains))
            ),
            "rapid_sequence_changes": np.sum(
                np.diff(dwell_times) < 1
            ),  # Changes under 1 second
            "avg_dwell_time": np.mean(dwell_times),
            "frac_time_social_media": sum(
                dwell_times[np.isin(domains, self.social_domains)]
            )
            / sum(dwell_times),
            "frac_time_search": sum(dwell_times[np.isin(domains, self.search_domains)])
            / sum(dwell_times),
            "frac_time_email": sum(dwell_times[np.isin(domains, self.email_domains)])
            / sum(dwell_times),
            "frac_time_commerce": sum(
                dwell_times[np.isin(domains, self.commerce_domains)]
            )
            / sum(dwell_times),
            "frac_time_news": sum(dwell_times[np.isin(domains, self.news_domains)])
            / sum(dwell_times),
        }

    @staticmethod
    def _calculate_entropy(frequencies: List[int]) -> float:
        """Calculate entropy of a distribution."""
        frequencies = np.array(frequencies)
        probabilities = frequencies / frequencies.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))


if __name__ == "__main__":
    """
    Usage:

    python -m cybergpt.models.features \
        --data_csv data/web_tracking/web_routineness_release/raw/browsing.csv \
        --output_dir data/features
    """
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent.parent)
    sys.path.append(project_root)

    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_csv",
        type=str,
        default="data/web_tracking/web_routineness_release/raw/browsing.csv",
    )
    args.add_argument("--output_dir", type=str, default="data/features")
    args = args.parse_args()

    user_data = WebTrackingLoader(args.data_csv).user_data
    dataset = WebsiteDataset(user_data)

    sequences = dataset.extract_sequences(
        split="pause", pause_threshold=3600, combine_sequential=True
    )
    dfs = [s["df"] for s in sequences]
    labels = [s["user_id"] for s in sequences]

    extractor = SessionFeatureExtractor()
    features_df = extractor.extract_features(dfs)

    print("Size of feature matrix:", features_df.shape)
    print("Extracted features:", extractor.feature_names)

    output = {
        "features": features_df,
        "feature_names": extractor.feature_names,
        "labels": labels,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump(output, open(os.path.join(args.output_dir, "features.pkl"), "wb"))
