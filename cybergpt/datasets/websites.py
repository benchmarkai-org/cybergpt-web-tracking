import pandas as pd
from datetime import timedelta
from collections import defaultdict
from typing import Dict, List
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm

from cybergpt.datasets.sequences import SequenceDataset
from cybergpt.datasets.base import BaseDataset
from cybergpt.datasets.utils import jensen_shannon_distance


class WebsiteDataset(BaseDataset):
    """Dataset for browsing data. Primarily for investigating German Web Tracking dataset."""

    def __init__(self, user_data: Dict[str, DataFrame]):
        super().__init__()
        self.user_data = user_data
        # Cache user stats
        self.user_stats: DataFrame | None = None

    @staticmethod
    def split_by_pause(df: DataFrame, pause_threshold: int = 3600) -> list[DataFrame]:
        """Split a dataframe on pauses threshold"""
        dfs = []
        df = df.reset_index(drop=True)
        df["diff_ts"] = (
            df["timestamp"].shift(-1).diff().apply(lambda x: x.total_seconds())
        )
        df["jump"] = np.abs(df["diff_ts"] - df["active_seconds"]) > pause_threshold
        split_indices = df.index[df["jump"]].tolist()
        all_indices = [-1] + split_indices + [len(df)]
        df = df.drop(columns=["diff_ts", "jump"])
        for start, end in zip(all_indices[:-1], all_indices[1:]):
            dfs.append(df.loc[start + 1 : end, :])
        return dfs

    @staticmethod
    def combine_sequential(df: DataFrame, break_threshold: int = 60) -> DataFrame:
        """Combine sequential requests from the same domain into a single request.
        The parameter break_threshold is the additional seconds beyond active_seconds
        to consider as same session."""
        df = df.sort_values("timestamp").copy()
        time_diffs = df["timestamp"].diff().shift(-1).dt.total_seconds()

        should_combine = (df["domain"].shift(-1) == df["domain"]) & (
            time_diffs <= (df["active_seconds"] + break_threshold)
        )
        groups = (~should_combine.fillna(False)).cumsum()

        agg_dict = {
            "timestamp": "first",
            "domain": "first",
            "active_seconds": "sum",
            "date": "first",
            "hour": "first",
            "day_of_week": "first",
        }

        return df.groupby(groups).agg(agg_dict).reset_index(drop=True)

    @staticmethod
    def split_history_into_chunks(
        df: pd.DataFrame,
        split: str = "date",
        pause_threshold: int = 3600,
        combine_sequential: bool = False,
    ) -> List[pd.DataFrame]:
        """Split user browsing history into chunks."""

        # Split by date or between pauses
        if split == "date":
            dfs = [day_df for _, day_df in df.groupby("date")]
        elif split == "pause":
            dfs = WebsiteDataset.split_by_pause(df, pause_threshold)

        # Remove empty dataframes
        dfs = [df for df in dfs if not df.empty]

        # Combine sequential requests, if required
        if combine_sequential:
            dfs = [WebsiteDataset.combine_sequential(df) for df in dfs]

        return dfs

    def extract_sequences(
        self,
        split: str = "date",
        pause_threshold: int = 3600,
        combine_sequential: bool = False,
    ) -> List[dict]:
        """Extract sequences from user data."""
        sequences = []
        for user_id, df in tqdm(self.user_data.items(), desc="Processing user data"):
            dfs = WebsiteDataset.split_history_into_chunks(
                df, split, pause_threshold, combine_sequential
            )
            sequences.extend(
                {"user_id": user_id, "start_time": df["timestamp"].min(), "df": df}
                for df in dfs
            )
        return sequences

    def to_sequence_dataset(
        self,
        split: str = "date",
        pause_threshold: int = 3600,
        combine_sequential: bool = False,
    ) -> SequenceDataset:
        """Convert user data to a sequence dataset, splitting sequences by user and date or on pauses."""
        dict_sequences = self.extract_sequences(
            split, pause_threshold, combine_sequential
        )
        sequences = [s["df"]["domain"].tolist() for s in dict_sequences]
        return SequenceDataset(sequences)

    @staticmethod
    def categorise_domain(domain: str) -> str:
        """Categorise a domain into predefined categories."""
        categories = {
            "social_media": ["facebook", "twitter", "instagram", "linkedin"],
            "video": ["youtube", "vimeo", "dailymotion", "googlevideo"],
            "cloud": ["drive.google", "dropbox", "onedrive"],
            "email": ["gmail", "outlook", "yahoo"],
        }

        domain_lower = domain.lower()
        for category, keywords in categories.items():
            if any(keyword in domain_lower for keyword in keywords):
                return category

        # Check if domain is an IP address
        if all(part.isdigit() and int(part) < 256 for part in domain.split(".")):
            return "ip_address"

        return "other"

    def compute_all_user_stats(self) -> DataFrame:
        """Compute basic statistics for all users."""
        if self.user_stats is not None:
            return self.user_stats

        stats = {
            user_id: self.compute_user_stats(df)
            for user_id, df in self.user_data.items()
        }
        self.user_stats = DataFrame(stats).T
        return self.user_stats

    @staticmethod
    def analyse_temporal_patterns(df: DataFrame) -> tuple[Series, Series, Series]:
        """analyse temporal patterns in browsing behaviour."""
        hourly_activity = df.groupby("hour").size()
        daily_activity = df.groupby("day_of_week").size()
        daily_volume = df.groupby("date").size()
        return hourly_activity, daily_activity, daily_volume

    @staticmethod
    def detect_anomalous_patterns(df: DataFrame) -> tuple[int, int]:
        """Detect potential anomalous patterns in browsing behaviour."""
        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff()

        # Identify rapid succession requests (potential automated behaviour)
        rapid_requests = time_diffs[time_diffs < timedelta(seconds=1)].count()

        # Identify unusual hours activity (outside 7am-11pm)
        unusual_hours_activity = df[~df["hour"].between(7, 23)].shape[0]

        return rapid_requests, unusual_hours_activity

    @staticmethod
    def analyse_domain_patterns(df: DataFrame) -> tuple[Series, Series, Series]:
        """analyse patterns in domain access."""
        domain_counts = df["domain"].value_counts()

        large_volume_domains = domain_counts[
            domain_counts > domain_counts.mean() + 2 * domain_counts.std()
        ]

        df["domain_category"] = df["domain"].apply(WebsiteDataset.categorise_domain)
        category_distribution = df["domain_category"].value_counts()

        return domain_counts, large_volume_domains, category_distribution

    @staticmethod
    def analyse_temporal_stability(df: DataFrame) -> tuple[DataFrame, DataFrame]:
        """analyse the stability of temporal patterns for each day of the week."""
        hourly_by_date = (
            df.groupby(["date", "day_of_week", "hour"]).size().reset_index(name="count")
        )
        hourly_by_date["normalized_count"] = hourly_by_date.groupby(
            ["date", "day_of_week"]
        )["count"].transform(lambda x: x / x.sum())
        distributions = hourly_by_date.pivot_table(
            index=["date", "day_of_week"],
            columns="hour",
            values="normalized_count",
            fill_value=0,
        )

        # Jensen-Shannon distances between distributions for each day of week
        days = range(7)
        similarity_results = []

        for day in days:
            day_distributions = distributions[
                distributions.index.get_level_values("day_of_week") == day
            ]
            dates = day_distributions.index.get_level_values("date").unique()

            for i, date1 in enumerate(dates[:-1]):
                for date2 in dates[i + 1 :]:
                    dist1 = day_distributions.loc[date1].values
                    dist2 = day_distributions.loc[date2].values
                    js_distance = jensen_shannon_distance(dist1, dist2)
                    similarity_results.append(
                        {
                            "day_of_week": day,
                            "date1": date1,
                            "date2": date2,
                            "js_distance": js_distance,
                        }
                    )

        similarity_df = DataFrame(similarity_results)
        return similarity_df, distributions

    @staticmethod
    def analyse_domain_patterns_across_users(
        user_data: Dict[str, DataFrame]
    ) -> DataFrame:
        """analyse domain patterns across all users"""
        common_domains = set()
        domain_stats = defaultdict(list)

        for df in user_data.values():
            domains = set(df["domain"].value_counts().head(10).index)
            common_domains.update(domains)

        for df in user_data.values():
            domain_counts = df["domain"].value_counts()
            total_requests = len(df)

            for domain in common_domains:
                count = domain_counts.get(domain, 0)
                ratio = count / total_requests if total_requests > 0 else 0
                domain_stats[domain].append(ratio)

        return DataFrame(domain_stats, index=user_data.keys())

    @staticmethod
    def analyse_temporal_patterns_across_users(
        user_data: Dict[str, DataFrame]
    ) -> tuple[DataFrame, DataFrame]:
        """analyse temporal patterns across users"""
        hourly_patterns = {}
        daily_patterns = {}

        for user_id, df in user_data.items():
            hourly = df.groupby("hour").size()
            hourly = hourly / hourly.sum()
            hourly_patterns[user_id] = hourly

            daily = df.groupby("day_of_week").size()
            daily = daily / daily.sum()
            daily_patterns[user_id] = daily

        return DataFrame(hourly_patterns).T, DataFrame(daily_patterns).T

    def detect_anomalous_users(self) -> dict[str, list[str]]:
        """Detect users with unusual patterns"""
        anomalies = defaultdict(list)

        for metric in [
            "avg_domains_per_day",
            "avg_requests_per_day",
            "night_activity_ratio",
        ]:
            mean = self.user_stats[metric].mean()
            std = self.user_stats[metric].std()
            threshold = mean + 2 * std

            anomalous_users = self.user_stats[self.user_stats[metric] > threshold].index
            if len(anomalous_users) > 0:
                anomalies[metric].extend(anomalous_users)

        return anomalies

    def generate_summary_report(
        self,
    ) -> tuple[dict, DataFrame, dict[str, list[str]]]:
        """Generate comprehensive summary report"""
        user_stats = self.compute_all_user_stats()
        anomalies = self.detect_anomalous_users()
        domain_stats = self.analyse_domain_patterns_across_users(self.user_data)
        eff_dim = self.analyse_user_diversity(domain_stats)

        summary = {
            "total_users": len(self.user_data),
            "avg_records_per_user": user_stats["total_records"].mean(),
            "avg_domains_per_user": user_stats["unique_domains"].mean(),
            "avg_active_days": user_stats["active_days"].mean(),
            "users_with_night_activity": sum(user_stats["night_activity_ratio"] > 0.1),
            "users_with_weekend_activity": sum(
                user_stats["weekend_activity_ratio"] > 0.3
            ),
            "effective_dimension_of_browsing_patterns": eff_dim,
            "anomalous_users": sum(len(users) for users in anomalies.values()),
        }

        return (
            summary,
            user_stats,
            anomalies,
        )

    def analyse_browsing_patterns_across_users(
        self,
    ) -> tuple[DataFrame, dict[str, list[str]]]:
        """Main function to analyse browsing patterns across all users"""

        (
            summary,
            user_stats,
            anomalies,
        ) = self.generate_summary_report()

        print("User Analysis Summary")
        print("-----------------------")
        for key, value in summary.items():
            print(f"{key}: {value:.2f}")
        print("")
        print("User Statistics Summary")
        print("-----------------------")
        print(user_stats.describe())
        print("")
        print("Anomalous Users")
        print("---------------")
        for metric, users in anomalies.items():
            print(f"\n{metric}:")
            print(f"Number of anomalous users: {len(users)}")

        return user_stats, anomalies

    def compute_top_domains(self, top_by_user: int = 10) -> DataFrame:
        """Compute the top domains by user"""
        all_domains = []
        for df in self.user_data.values():
            domains = df["domain"].value_counts().head(top_by_user).index
            all_domains.extend(domains)
        return pd.Series(all_domains).value_counts()

    @staticmethod
    def compute_user_stats(df: DataFrame) -> Series:
        """Compute basic statistics for a single user's browsing data."""
        total_records = len(df)
        unique_domains = df["domain"].nunique()
        active_days = df["date"].nunique()

        avg_domains_per_day = df.groupby("date")["domain"].nunique().mean()
        avg_requests_per_day = df.groupby("date").size().mean()

        night_hours = set(range(0, 6))  # 12am-6am
        night_activity = df[df["hour"].isin(night_hours)].shape[0]
        night_activity_ratio = (
            night_activity / total_records if total_records > 0 else 0
        )

        weekend_days = {5, 6}  # Saturday and Sunday
        weekend_activity = df[df["day_of_week"].isin(weekend_days)].shape[0]
        weekend_activity_ratio = (
            weekend_activity / total_records if total_records > 0 else 0
        )

        return Series(
            {
                "total_records": total_records,
                "unique_domains": unique_domains,
                "active_days": active_days,
                "avg_domains_per_day": avg_domains_per_day,
                "avg_requests_per_day": avg_requests_per_day,
                "night_activity_ratio": night_activity_ratio,
                "weekend_activity_ratio": weekend_activity_ratio,
            }
        )

    @staticmethod
    def analyse_user_diversity(domain_stats: DataFrame) -> tuple[DataFrame, float]:
        """Analyze diversity between users based on their domain usage patterns."""
        normalized_stats = domain_stats.div(domain_stats.sum(axis=1), axis=0)
        user_corr_matrix = normalized_stats.T.corr()
        evals, _ = np.linalg.eig(user_corr_matrix)
        evals = np.real(evals)
        evals = evals[evals > 0]
        p = evals / sum(evals)
        eff_dim = 2 ** (-np.sum(p * np.log2(p)))
        return eff_dim


def plot_temporal_patterns(
    hourly_activity, daily_activity, domain_counts, category_distribution
):
    """Create visualizations for the analysis."""
    plt.figure(figsize=(10, 7))

    # Hourly activity pattern
    plt.subplot(2, 2, 1)
    hourly_activity.plot(kind="bar")
    plt.title("Hourly Browsing Pattern")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Requests")

    # Daily activity pattern
    plt.subplot(2, 2, 2)
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily_activity_plot = daily_activity.reindex(
        range(7)
    )  # Ensure all days are included
    ax = daily_activity_plot.plot(kind="bar")
    plt.title("Daily Browsing Pattern")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Requests")
    plt.xticks(range(7), days, rotation=45)

    # Top domains
    plt.subplot(2, 2, 3)
    domain_counts.head(10).plot(kind="bar")
    plt.title("Top 10 Domains")
    plt.xticks(rotation=45, ha="right")

    # Domain categories - replacing pie chart with bar chart
    plt.subplot(2, 2, 4)
    category_distribution.plot(kind="bar")
    plt.title("Domain Categories Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Requests")

    plt.tight_layout()
    return plt.gcf()


def plot_temporal_stability(similarity_df, distributions):
    """Visualize the temporal stability analysis."""
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Box plot of Jensen-Shannon distances by day of week
    sns.boxplot(data=similarity_df, x="day_of_week", y="js_distance", ax=ax1)
    ax1.set_title("Distribution Similarity by Day of Week\n(Lower = More Similar)")
    ax1.set_xlabel("Day of Week")
    ax1.set_ylabel("Jensen-Shannon Distance")
    ax1.set_xticklabels(days)

    # Heatmap of average hourly patterns by day of week
    avg_patterns = distributions.groupby(level="day_of_week").mean()
    sns.heatmap(avg_patterns, cmap="YlOrRd", ax=ax2)
    ax2.set_title("Average Hourly Patterns by Day of Week")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Day of Week")
    ax2.set_yticklabels(days, rotation=0)

    plt.tight_layout()
    return plt.gcf()


def plot_distributions(user_stats, hourly_patterns, top_domain_stats):
    """Create visualizations for distributions"""
    plt.figure(figsize=(20, 15))

    # 1. Distribution of records per user
    plt.subplot(2, 2, 1)
    sns.histplot(user_stats["total_records"])
    plt.title("Distribution of Records per User")
    plt.xlabel("Number of Records")

    # 2. Average domains per day distribution
    plt.subplot(2, 2, 2)
    sns.histplot(user_stats["avg_domains_per_day"])
    plt.title("Distribution of Average Domains per Day")
    plt.xlabel("Average Domains per Day")

    # 3. Hourly patterns
    plt.subplot(2, 2, 3)
    sns.boxplot(data=hourly_patterns)
    plt.title("Hourly Activity Patterns Across Users")
    plt.xlabel("Hour of Day")
    plt.ylabel("Activity Ratio")

    # 4. Domain patterns
    plt.subplot(2, 2, 4)
    domain_freq = top_domain_stats.head(10)
    sns.barplot(x=domain_freq.values, y=domain_freq.index)
    plt.title("Most Common Domains Across All Users")
    plt.xlabel("Number of Users Domain Appears in Top 10")

    plt.tight_layout()
    return plt.gcf()
