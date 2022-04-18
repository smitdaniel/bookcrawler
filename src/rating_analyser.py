from data_reader import BookReviewData
from typing import Optional, Callable
from definitions import data
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RatingAnalyser:

    brd: BookReviewData = BookReviewData()

    def __init__(self):
        self.ratings: pd.DataFrame = self.brd.ratings
        self.lotr: pd.DataFrame = self.brd.lotr.copy()
        self.lotr_ratings: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "LOTR-Ratings.csv")
        self.bracket_lotr: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "LOTR-Bracket.csv")
        self.close_books: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "LOTR-Close-Books.csv")
        self.rating_stats: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "Rating-Stats.csv")
        self.close_books_rank: Optional[pd.DataFrame] = None
        print("RatingAnalyser object initialized.")

    def get_lotr_rating_by_age(self, plot: bool = False) -> None:
        """Get rating count and mean rating by age decades

        Note: This takes quite a lot of time to compute.
        """
        bracket_len: int = 10
        brackets: np.ndarray = np.arange(0, 120, bracket_len)
        bracket_lotr: pd.DataFrame = pd.DataFrame(index=brackets, columns=["Count", "Mean"])
        for low_b in brackets:
            bracket = (low_b, low_b + bracket_len)
            print(f"\rAnalyzing age bracket {bracket}.", end="")
            bracketed_means = self.brd.get_mean_rating(bracket)[["Rating-Count", "Rating-Mean"]]
            bracket_lotr.loc[low_b] = [bracketed_means["Rating-Count"].sum(), bracketed_means["Rating-Mean"].mean()]
        self.bracket_lotr = bracket_lotr
        if not (data / "LOTR-Bracket.csv").is_file():
            RatingAnalyser.write_csv(self.bracket_lotr, data / "LOTR-Bracket.csv")
        if plot:
            self.bracket_lotr.plot.bar()
            plt.show()

    def find_close_books(self) -> pd.DataFrame:
        positive_lotr_ratings = self.lotr_ratings\
            .groupby("User-ID")\
            .filter(lambda group: (group.loc[group["ISBN"].isin(self.lotr["ISBN"]), "Book-Rating"].max() >= 8) |
                    all(group.loc[group["ISBN"].isin(self.lotr["ISBN"]), "Book-Rating"] == 0))
        related_ratings: pd.DataFrame = positive_lotr_ratings.loc[~positive_lotr_ratings["ISBN"].isin(self.lotr["ISBN"])]
        close_books = related_ratings.groupby("ISBN").apply(self._group_isbn)
        self.close_books: pd.DataFrame = close_books.loc[close_books["count"] > 5]
        if not (data / "LOTR-Close-Books.csv").is_file():
            RatingAnalyser.write_csv(self.close_books, data / "LOTR-Close-Books.csv")
        return self.close_books.nlargest(100, "count")

    def get_general_rating_stats(self) -> pd.DataFrame:
        self.rating_stats: pd.DataFrame = self.ratings.groupby("ISBN").apply(self._group_isbn)
        self.rating_stats = self.rating_stats.loc[self.rating_stats["count"] > 5]
        if not (data / "Rating-Stats.csv").is_file():
            RatingAnalyser.write_csv(self.rating_stats, data / "Rating-Stats.csv")
        return self.rating_stats

    def plot_by_country(self, n_large: int = 20, lotr_only: bool = False) -> None:
        ratings = self.ratings if not lotr_only else self.ratings[self.ratings["ISBN"].isin(self.lotr["ISBN"])]
        rating_origins = ratings.groupby("Country").count().nlargest(n_large, "Book-Rating")
        rating_origins.plot.bar(y="Book-Rating")
        plt.show()

    def plot_by_age(self, year_range: int = 10, lotr_only: bool = False) -> None:
        ratings = self.ratings if not lotr_only else self.ratings[self.ratings["ISBN"].isin(self.lotr["ISBN"])]
        rating_age = ratings.groupby(pd.cut(self.ratings["Age"], np.arange(0, 120, year_range)))\
            .count()
        rating_age.plot.bar(y="Age")
        plt.show()

    def filter_lotr_specific(self) -> pd.DataFrame:
        relevant_close: pd.DataFrame = self.close_books.loc[self.close_books["rating-mean"] > 7].nlargest(1000, "count")
        relevant_close['rank'] = relevant_close['count'].rank(method='max').apply(lambda r: 100 * (r-1)/1000)
        relevant_close.set_index("ISBN", inplace=True)
        relevant_close.sort_index(inplace=True)
        general: pd.DataFrame = self.rating_stats.nlargest(10000, "count")
        general['rank'] = general['count'].rank(method='max').apply(lambda r: 100 * (r-1)/10000)
        relevant_general = general[general["ISBN"].isin(relevant_close.index)]
        relevant_general.set_index("ISBN", inplace=True)
        relevant_general.sort_index(inplace=True)
        relevant_close['rank-gain'] = relevant_close['rank'] - relevant_general['rank']
        self.close_books_rank = relevant_close
        return self.close_books_rank

    @classmethod
    def write_ratings_with_lotr(cls) -> pd.DataFrame:
        lotr_ratings = cls.brd.ratings.groupby("User-ID")\
            .filter(lambda group: any(group["ISBN"].isin(cls.brd.lotr["ISBN"])))
        RatingAnalyser.write_csv(lotr_ratings, data / "LOTR-Ratings.csv")
        return lotr_ratings

    @classmethod
    def read_or_warn(cls, inpath: Path) -> Optional[pd.DataFrame]:
        if inpath.is_file():
            return cls.brd.read_csv(inpath)
        else:
            print(f"The file in path {inpath} does not exist. Please run the corresponding method to generate it.")
            return None

    @classmethod
    def write_csv(cls, pdf: pd.DataFrame, outpath: Path) -> None:
        pdf.to_csv(outpath, sep=";", index=True, encoding="latin-1", quoting=1, escapechar="\\", na_rep="N/A")

    @staticmethod
    def _group_isbn(isbn: pd.DataFrame) -> pd.Series:
        """Take a df for ratings of each isbn and compute count, mean and std of rating"""
        grouped = {}
        grouped['count'] = len(isbn.index)
        grouped['rating-mean'] = isbn.loc[isbn["Book-Rating"] != 0, "Book-Rating"].mean()
        grouped['rating-std'] = isbn.loc[isbn["Book-Rating"] != 0, "Book-Rating"].std()
        return pd.Series(grouped, index=['count', 'rating-mean', 'rating-std'])


if __name__ == "__main__":
    ra = RatingAnalyser()
    ra.filter_lotr_specific()
    pass
