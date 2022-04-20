from data_reader import BookReviewData
from typing import Optional, Callable, List
from definitions import data
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RatingAnalyser:

    brd: BookReviewData = BookReviewData(only_lotr=True)
    print("Loaded book review data")

    def __init__(self):
        self.ratings: pd.DataFrame = self.brd.ratings
        self.lotr: pd.DataFrame = self.brd.lotr.copy()
        interest_ratings: str = "LOTR-Ratings.csv" if self.brd.only_lotr else "Tolkien-Ratings.csv"
        close_books: str = "LOTR-Close-Books.csv" if self.brd.only_lotr else "Tolkien-Close-Books.csv"
        self.lotr_ratings: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / interest_ratings)
        self.bracket_lotr: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "LOTR-Bracket.csv")
        self.close_books: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / close_books)
        self.rating_stats: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "Rating-Stats.csv")
        self.close_books_rank: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "Ranked-Books.csv")
        self.relevant: Optional[pd.DataFrame] = None
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
        outpath: Path = data / "LOTR-Close-Books.csv" if self.brd.only_lotr else data / "Tolkien-Close-Books.csv"
        positive_lotr_ratings = self.lotr_ratings\
            .groupby("User-ID")\
            .filter(lambda group: (group.loc[group["ISBN"].isin(self.lotr["ISBN"]), "Book-Rating"].max() >= 8) |
                    all(group.loc[group["ISBN"].isin(self.lotr["ISBN"]), "Book-Rating"] == 0))
        positive_lotr_ratings['Group-Weight'] = positive_lotr_ratings.groupby("User-ID")['ISBN']\
            .transform(lambda group: len(group.loc[group.isin(self.lotr['ISBN'])]))
        related_ratings: pd.DataFrame = positive_lotr_ratings.loc[~positive_lotr_ratings["ISBN"].isin(self.lotr["ISBN"])]
        close_books = related_ratings.groupby("ISBN").apply(self._group_isbn)
        self.close_books: pd.DataFrame = close_books.loc[close_books["count"] > 5]
        if not outpath.is_file():
            RatingAnalyser.write_csv(self.close_books, outpath)
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

    def filter_lotr_specific(self, thresh_rating: int = 7) -> pd.DataFrame:
        mask: Callable[[pd.DataFrame], pd.Series] = lambda pdf: (pdf['rating-mean'] >= thresh_rating) | \
                                                                (pdf['rating-mean'].isna())
        print("Started comparing lotr-specific and general bookset.")
        print(f"Initiating with {len(self.close_books.index)} lotr-books "
              f"and {len(self.rating_stats.index)} general books.")
        relevant_close: pd.DataFrame = self.close_books.loc[mask(self.close_books)]
        relevant_close['weighted-count'] = relevant_close['count'] * relevant_close['normalized-weight']
        relevant_close.loc[:, 'rank'] = relevant_close.loc[:, 'weighted-count']\
                                                      .rank(method='max')\
                                                      .apply(lambda r: 100 * (r-1)/len(relevant_close.index))
        relevant_close.set_index("ISBN", inplace=True)
        relevant_close.sort_index(inplace=True)
        general: pd.DataFrame = self.rating_stats.loc[mask(self.rating_stats)]
        general.loc[:, 'rank'] = general.loc[:, 'count']\
                                        .rank(method='max')\
                                        .apply(lambda r: 100 * (r-1)/len(general.index))
        print(f"After removing ratings < {thresh_rating}, there are {len(relevant_close.index)} lotr-books "
              f"and {len(general.index)} books.")
        relevant_general = general[general["ISBN"].isin(relevant_close.index)]
        relevant_general.set_index("ISBN", inplace=True)
        relevant_general.sort_index(inplace=True)
        relevant_close.loc[:, 'rank-gain'] = relevant_close.loc[:, 'rank'] - relevant_general.loc[:, 'rank']
        self.close_books_rank = relevant_close
        if not (data / "Ranked-Books.csv").is_file():
            RatingAnalyser.write_csv(self.close_books_rank, data / "Ranked-Books.csv")
        return self.close_books_rank

    def filter_relevant(self, book_limit: int = 500, only_rank_gain: bool = True,
                        plot_type: str = 'scatter') -> None:
        if plot_type not in ['scatter', 'heatmap', 'none']:
            print(f"Allowed plot_type values are 'scatter', 'none', or 'heatmap'. "
                  f"You passed {plot_type}. Falling back to 'scatter'.")
            plot_type = 'scatter'
        mask: pd.Series = (self.close_books_rank['rank-gain'] >= 0) if only_rank_gain \
            else pd.Series(True, index=self.close_books_rank.index)
        relevant: pd.DataFrame = self.close_books_rank[mask].nlargest(book_limit, "rank-gain")
        print(f"Selected {len(relevant.index)} books, "
              f"of {(self.close_books_rank['rank-gain'] > 0).sum()} books with rank gain >= 0.")
        print(f"Mean rank-gain is {relevant['rank-gain'].mean()}, mean count is {relevant['count'].mean()}")
        relevant['rank-spread'] = relevant['rank-gain'] - relevant['rank-gain'].mean()
        relevant['count-spread'] = relevant['count'] - relevant['count'].mean()
        print(f"Relevant books have rank-gain std {relevant['rank-gain'].std()}"
              f"and count std {relevant['count'].std()}.")
        if plot_type == 'scatter':
            relevant.plot.scatter(x='rank-spread', y='count-spread', c='rating-mean', colormap='viridis')
            plt.show()
        elif plot_type == 'heatmap':
            RatingAnalyser._plot_relevant_heatmap(relevant)
        self.relevant: pd.DataFrame = relevant

    def link_relevant(self) -> None:
        reduced_ratings: pd.DataFrame = self.ratings.groupby('User-ID')\
                                                    .filter(lambda userid: len(userid.index) < 1000)
        reduced_ratings = reduced_ratings.loc[reduced_ratings['ISBN'].isin(self.relevant['ISBN'])]\
                                         .groupby('User-ID')\
                                         .filter(lambda uid: len(uid.index) > 1)
        userid_group: pd.DataFrameGroupBy = reduced_ratings.groupby('User-ID')
        isbns: List[str] = self.relevant['ISBN'].to_list()
        link_matrix: np.ndarray = np.zeros((len(isbns), len(isbns)))
        for num_a, isbn_a in enumerate(isbns):
            print(f"\rProcessing {num_a} of {len(isbns)}.", end="")
            for num_b, isbn_b in enumerate(isbns[num_a:]):
                num_b = num_b + num_a
                link_matrix[num_a, num_b] = userid_group.apply(lambda uid: any(uid['ISBN'] == isbn_a) &
                                                                           any(uid['ISBN'] == isbn_b))\
                                                        .sum()
        self.link_matrix = link_matrix



    @staticmethod
    def _plot_relevant_heatmap(relevant: pd.DataFrame) -> None:
        x = relevant['rank-spread'].to_numpy()
        y = relevant['count-spread'].to_numpy()

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(20, 10))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, interpolation='bilinear', extent=extent, origin='lower', aspect='auto')
        plt.xlabel('rank gain')
        plt.ylabel('count')
        plt.show()

    @classmethod
    def write_ratings_with_lotr(cls) -> pd.DataFrame:
        lotr_ratings = cls.brd.ratings.groupby("User-ID")\
            .filter(lambda group: any(group["ISBN"].isin(cls.brd.lotr["ISBN"])))
        outpath: Path = data / "LOTR-Ratings.csv" if cls.brd.only_lotr else data / "Tolkien-Ratings.csv"
        lotr_ratings.set_index("User-ID", inplace=True)
        RatingAnalyser.write_csv(lotr_ratings, outpath)
        return lotr_ratings

    @classmethod
    def read_or_warn(cls, inpath: Path) -> Optional[pd.DataFrame]:
        if inpath.is_file():
            print(f"Found and reading {inpath}.")
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
        if 'Group-Weight' in isbn.columns:
            grouped['group-weight'] = isbn["Group-Weight"].mean()
            grouped['normalized-weight'] = isbn["Group-Weight"].mean() / len(isbn.index)
        else:
            grouped['group-weight'] = 0
            grouped['normalized-weight'] = 0
        return pd.Series(grouped, index=['count', 'rating-mean', 'rating-std', 'group-weight', 'normalized-weight'])


if __name__ == "__main__":
    ra = RatingAnalyser()
    ra.filter_relevant(book_limit=100, only_rank_gain=True, plot_type='none')
    ra.link_relevant()
    pass
