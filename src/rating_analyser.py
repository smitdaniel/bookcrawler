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

    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 26}
    plt.rc('font', **font)

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
        self.link_matrix: Optional[pd.DataFrame] = RatingAnalyser.read_or_warn(data / "Link-Matrix.csv")
        if self.link_matrix is not None:
            self.link_matrix.set_index(self.link_matrix.columns[0], inplace=True)
        print("RatingAnalyser object initialized.")

    def get_lotr_rating_by_age(self, plot: bool = False) -> None:
        """Get rating count and mean rating by age decades

        Note: This takes quite a lot of time to compute.
        """
        bracket_len: int = 10
        brackets: np.ndarray = np.arange(self.ratings["Age"].min(), self.ratings["Age"].max(), bracket_len)
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

    def plot_by_country(self, n_large: int = 20, lotr_only: bool = False) -> None:
        ratings = self.ratings if not lotr_only else self.ratings[self.ratings["ISBN"].isin(self.lotr["ISBN"])]
        rating_origins = ratings.groupby("Country").count().nlargest(n_large, "Book-Rating")
        rating_origins.plot.bar(y="Book-Rating")
        plt.show()

    def plot_by_age(self, year_range: int = 10, lotr_only: bool = False) -> None:
        ratings = self.ratings if not lotr_only else self.ratings[self.ratings["ISBN"].isin(self.lotr["ISBN"])]
        rating_age = ratings.groupby(pd.cut(self.ratings["Age"],
                                            np.arange(self.ratings["Age"].min(), self.ratings["Age"].max(), year_range)))\
                            .count()
        rating_age.plot.bar(y="Age")
        plt.show()

    def plot_by_user_reviews(self, bin_count: int = 10, lotr_only: bool = False) -> None:
        ratings = self.ratings if not lotr_only else self.ratings[self.ratings["ISBN"].isin(self.lotr["ISBN"])]
        rating_byuser: pd.DataFrame = ratings.groupby("User-ID").count()
        plt.hist(rating_byuser['Book-Rating'], log=True, bins=bin_count)
        plt.xlabel("ratings per user")
        plt.ylabel("count")
        plt.title("Ratings per user histogram")
        plt.show()

    def find_close_books(self) -> pd.DataFrame:
        outpath: Path = data / "LOTR-Close-Books.csv" if self.brd.only_lotr else data / "Tolkien-Close-Books.csv"
        positive_lotr_ratings = self.lotr_ratings\
            .groupby("User-ID")\
            .filter(lambda group: (group.loc[group["ISBN"].isin(self.lotr["ISBN"]), "Book-Rating"].max() >= 8) |
                    all(group.loc[group["ISBN"].isin(self.lotr["ISBN"]), "Book-Rating"] == 0))
        positive_lotr_ratings['Group-Weight'] = positive_lotr_ratings.groupby("User-ID")['ISBN']\
            .transform(lambda group: len(group.loc[group.isin(self.lotr['ISBN'])]))
        positive_lotr_ratings['Group-Size'] = positive_lotr_ratings.groupby("User-ID")['ISBN']\
            .transform(lambda group: len(group))
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
            relevant.plot.scatter(x='rank-gain', y='count', c='rating-mean', s=120, colormap='viridis',
                                  title="Relevant books stats")
            plt.show()
        elif plot_type == 'heatmap':
            RatingAnalyser._plot_relevant_heatmap(relevant)
        self.relevant: pd.DataFrame = relevant

    def link_relevant(self, best_lotr: int = 15) -> pd.DataFrame:
        top_lotr_isbn: List[str] = self.lotr_ratings.loc[self.lotr_ratings['ISBN'].isin(self.lotr['ISBN'])]\
                                                    .groupby('ISBN')\
                                                    .apply(lambda isbn: len(isbn.index))\
                                                    .nlargest(best_lotr)\
                                                    .index\
                                                    .to_list()
        isbn_list: List[str] = self.relevant['ISBN'].to_list() + top_lotr_isbn
        print(f"Selected {len(isbn_list)} books, of which {len(top_lotr_isbn)} are LOTR books.")
        reduced_ratings: pd.DataFrame = self.ratings.groupby('User-ID')\
                                                    .filter(lambda userid: len(userid.index) < 1000)
        reduced_ratings = reduced_ratings.loc[reduced_ratings['ISBN'].isin(isbn_list)]\
                                         .groupby('User-ID')\
                                         .filter(lambda uid: len(uid.index) > 1)
        print(f"Selected {len(reduced_ratings.index)} ratings for relevant books, with "
              f" < 1000 books per user, and > 1 selected book per user.")
        userid_group: pd.DataFrameGroupBy = reduced_ratings.groupby('User-ID')
        link_matrix: np.ndarray = np.zeros((len(isbn_list), len(isbn_list)))
        for num_a, isbn_a in enumerate(isbn_list):
            print(f"\rProcessing {num_a} of {len(isbn_list)}.", end="")
            for num_b, isbn_b in enumerate(isbn_list[num_a:]):
                num_b = num_b + num_a
                link_matrix[num_a, num_b] = userid_group.apply(lambda uid: any(uid['ISBN'] == isbn_a) &
                                                                           any(uid['ISBN'] == isbn_b))\
                                                        .sum()
                link_matrix[num_b, num_a] = link_matrix[num_a, num_b]
        self.link_matrix = pd.DataFrame(link_matrix, columns=isbn_list, index=isbn_list)
        if not (data / "Link-Matrix.csv").is_file():
            RatingAnalyser.write_csv(self.link_matrix, data / "Link-Matrix.csv")
        return self.link_matrix

    def print_relevant(self, isbn_list: Optional[List[str]] = None):
        if isbn_list is None:
            isbn_list: List[str] = self.relevant['ISBN'].to_list()
        relevant_names: pd.Series = self.brd.books.loc[self.brd.books['ISBN'].isin(isbn_list), "Book-Title"]
        print(f"The following books were selected {relevant_names.to_string()}")

    def explore_links(self, iterations: int = 50, best_lotr: int = 15):
        norm: Callable[[np.ndarray], np.ndarray] = lambda v: v / np.linalg.norm(v, ord=1)
        link_matrix: np.ndarray = self.link_matrix.to_numpy()
        nls = link_matrix.shape[0] - best_lotr
        lotr_matrix: np.ndarray = link_matrix[nls:, nls:]
        book_matrix: np.ndarray = link_matrix[:nls, :nls]
        mixing_matrix: np.ndarray = link_matrix[:nls, :]
        lotr_weights: np.ndarray = norm(lotr_matrix.diagonal())
        candidate_books = np.concatenate([np.zeros(nls), lotr_weights])
        candidate_books = norm(mixing_matrix @ candidate_books)
        candidate_std: np.ndarray = np.zeros(iterations)
        for it in range(iterations):
            old_candidates = candidate_books.copy()
            candidate_books = norm(book_matrix @ candidate_books)
            candidate_std[it] = np.std(candidate_books)
            if np.linalg.norm(old_candidates-candidate_books, ord=1) < 1e-6:
                print(f"The weights change by less than 1e-6 after {it} iterations")
                break
        top_candidates: pd.DataFrame = pd.DataFrame(data=candidate_books, index=self.link_matrix.index[:nls],
                                                    columns=['book-weights'])
        ten_tops: pd.DataFrame = top_candidates.nlargest(30, 'book-weights')
        self.print_relevant(isbn_list=ten_tops.index.to_list())

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
        if 'Group-Weight' in isbn.columns and 'Group-Size' in isbn.columns:
            grouped['group-size'] = isbn["Group-Size"].mean()
            grouped['group-weight'] = isbn["Group-Weight"].mean()
            grouped['normalized-weight'] = isbn["Group-Weight"].mean() / grouped['group-size']
        else:
            grouped['group-size'] = 0
            grouped['group-weight'] = 0
            grouped['normalized-weight'] = 0
        return pd.Series(grouped, index=['count', 'rating-mean', 'rating-std',
                                         'group-size', 'group-weight', 'normalized-weight'])

    @staticmethod
    def _plot_relevant_heatmap(relevant: pd.DataFrame) -> None:
        x = relevant['rank-gain'].to_numpy()
        y = relevant['count'].to_numpy()

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(20, 10))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, interpolation='bilinear', extent=extent, origin='lower', aspect='auto')
        plt.xlabel('rank gain')
        plt.ylabel('count')
        plt.title('Relevant books density')
        plt.show()


if __name__ == "__main__":
    ra = RatingAnalyser()
    ra.filter_relevant(book_limit=300, only_rank_gain=True, plot_type='none')
    # ra.print_relevant()
    # ra.link_relevant()
    # ra.explore_links()
