from data_reader import BookReviewData
from typing import Optional, Callable, List, Union
from definitions import data
from pathlib import Path, _PosixFlavour, _WindowsFlavour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class CsvPath(Path):
    """Path class with added safe-read method"""

    _flavour = _PosixFlavour() if os.name == 'posix' else _WindowsFlavour()

    def read_or_warn(self) -> Optional[pd.DataFrame]:
        if self.is_file():
            print(f"Found and reading {self}.")
            return BookReviewData.read_csv(self)
        else:
            print(f"The file in path {self} does not exist. Please run the corresponding method to generate it.")
            return None

    def write_csv(self, pdf: pd.DataFrame) -> None:
        if self.is_file():
            print(f"Forcing update of existing {self}.")
        pdf.to_csv(self, sep=";", index=True, encoding="latin-1", quoting=1, escapechar="\\", na_rep="N/A")

    def write_if_absent(self, pdf: pd.DataFrame) -> None:
        if not self.is_file():
            self.write_csv(pdf)
        else:
            print(f"Cached {self} already exists, no update.")


class Cache:
    """Manages locations of cached calculation data."""

    def __init__(self, cache_dir: Path, prefixed: List[str], non_prefixed: List[str], prefix: str):
        for pref in prefixed:
            setattr(self, pref.lower().replace("-", "_"), self.set_prefixed_path(cache_dir, pref, prefix))
        for non_pref in non_prefixed:
            setattr(self, non_pref.lower().replace("-", "_"), self.set_prefixed_path(cache_dir, non_pref))
        self.cache_pattern: CsvPath = self.set_prefixed_path(cache_dir, "some_file", prefix)

    def __str__(self) -> str:
        return str(self.cache_pattern)

    @staticmethod
    def set_prefixed_path(dirpath: Path, filename: str, prefix: Optional[str] = None) -> CsvPath:
        return CsvPath(dirpath / ((prefix + "-" if prefix is not None else "") + filename + ".csv"))


class RatingAnalyser:
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 26}
    plt.rc('font', **font)

    def __init__(self, author: Union[List[str], str], book_title: Optional[str] = None,
                 strict_title: bool = False,
                 cache_prefix: Optional[str] = None, searched_in_link_analysis: int = 15):
        """
        Create instance containing all the definitions for analysis.
        Note that nothing is computed during construction. The init method tries to read the cached computations
        from files. If the files are not find, one needs to call appropriate methods to calculate them.
        :param author: list of strings to match as author name
        :param book_title: string to match as book title
        :param cache_prefix: prefix to use for cached data files
        :param searched_in_link_analysis: how many books from searched set (with highest occurence) should be used in
        links analysis
        """
        # create BookReviewData object; if additional filtering is needed, add the functions
        # like `get_mean_rating` and `filter_low_impact`
        self.brd: BookReviewData = BookReviewData(author, book_title, True, cache_prefix, strict_title=strict_title)
        print(f"Loaded book review data {self.brd}")
        self.best_searched: int = np.minimum(searched_in_link_analysis, len(self.brd.searched_books.index))
        # read data from the BRD object
        self.ratings: pd.DataFrame = self.brd.ratings
        self.searched_books: pd.DataFrame = self.brd.searched_books
        self.prefix: str = self.brd.prefix
        self.cached: Cache = Cache(data, ["Ratings", "Bracket", "Close-Books", "Ranked-Books", "Link-Matrix"],
                                   ["Rating-Stats"], self.prefix)
        print(f"Set cached files prefix to {self.prefix}, attempting to read the files from {self.cached}")
        # read cached data
        self.rating_stats: Optional[pd.DataFrame] = self.cached.rating_stats.read_or_warn()
        self.searched_ratings: Optional[pd.DataFrame] = self.cached.ratings.read_or_warn()
        self.bracket_searched: Optional[pd.DataFrame] = self.cached.bracket.read_or_warn()
        self.close_books: Optional[pd.DataFrame] = self.cached.close_books.read_or_warn()
        self.close_books_rank: Optional[pd.DataFrame] = self.cached.ranked_books.read_or_warn()
        self.link_matrix: Optional[pd.DataFrame] = self.cached.link_matrix.read_or_warn()
        if self.link_matrix is not None:
            self.link_matrix.set_index(self.link_matrix.columns[0], inplace=True)
        self.relevant: Optional[pd.DataFrame] = None  # this doesn't need to be cached, it's calculated quickly
        print("RatingAnalyser object initialized.")

    """
    Insight methods. Plotting, histograms, age-bracketing
    """

    def get_searched_rating_by_age(self, bracket_len: int = 10, plot: bool = False,
                                   upper_age: int = 100, plot_which: str = "Count") -> None:
        """Get rating count and mean rating by age decades
        """
        allowed_plots: List[str] = ["Count", "Mean"]
        if plot_which not in allowed_plots:
            plot_which = "Count"
            print(f"plot_which parameter accepts only {allowed_plots}. Falling back to 'Rating-Count'.")
        brackets: np.ndarray = np.arange(self.ratings["Age"].min(),
                                         np.minimum(upper_age, self.ratings["Age"].max()),
                                         bracket_len)
        bracket_searched: pd.DataFrame = pd.DataFrame(index=brackets, columns=["Count", "Mean"])
        for low_b in brackets:
            bracket = (low_b, low_b + bracket_len)
            print(f"\rAnalyzing age bracket {bracket}.", end="")
            bracketed_means = self.brd.get_mean_searched_rating(bracket, silent=True)[["Rating-Count", "Rating-Mean"]]
            bracket_searched.loc[low_b] = [bracketed_means["Rating-Count"].sum(), bracketed_means["Rating-Mean"].mean()]
        self.bracket_searched = bracket_searched
        self.cached.bracket.write_if_absent(self.bracket_searched)
        if plot:
            self.bracket_searched.plot.bar(y=plot_which)
            plt.show()

    def plot_by_country(self, n_large: int = 20, searched_only: bool = False) -> None:
        """Count of ratings of books (or searched books) by country"""
        ratings = self.ratings if not searched_only else self.ratings[
            self.ratings["ISBN"].isin(self.searched_books["ISBN"])]
        rating_origins = ratings.groupby("Country").count().nlargest(n_large, "Book-Rating")
        rating_origins.plot.bar(y="Book-Rating")
        plt.show()

    def plot_by_age(self, year_range: int = 10, searched_only: bool = False) -> None:
        """Count of ratings of books (or searched books) by age bracket"""
        ratings = self.ratings if not searched_only else self.ratings[
            self.ratings["ISBN"].isin(self.searched_books["ISBN"])]
        rating_age = ratings.groupby(pd.cut(self.ratings["Age"],
                                            np.arange(self.ratings["Age"].min(), self.ratings["Age"].max(),
                                                      year_range))) \
                            .count()
        rating_age.plot.bar(y="Age")
        plt.show()

    def plot_by_user_reviews(self, bin_count: int = 10, searched_only: bool = False) -> None:
        """Histogram of the number of reviews per User-ID"""
        ratings = self.ratings if not searched_only else self.ratings[
            self.ratings["ISBN"].isin(self.searched_books["ISBN"])]
        rating_byuser: pd.DataFrame = ratings.groupby("User-ID").count()
        plt.hist(rating_byuser['Book-Rating'], log=True, bins=bin_count)
        plt.xlabel("ratings per user")
        plt.ylabel("count")
        plt.title("Ratings per user histogram")
        plt.show()

    """
    General statistics and filtering
    """

    def run_full_analysis(self):
        """Runs the full pipeline which recalculated files, if not cached"""
        if self.rating_stats is None:
            self.get_general_rating_stats()
        if self.searched_ratings is None:
            self.filter_searched_ratings()
        if self.close_books is None:
            self.find_close_books()
        if self.close_books_rank is None:
            self.rank_books()
        self.filter_rank_gain(plot_type='none')
        self.print_relevant()
        if self.link_matrix is None:
            self.link_relevant()
        self.explore_links()

    def get_general_rating_stats(self, minimal_isbn_count: int = 4) -> pd.DataFrame:
        """Compute general statistics of ratings (count, mean, std) of all books

        :param minimal_isbn_count: minimal count of ISBN ratings to be included in statistics
        """
        print("Calculating general book rating statistics.")
        self.rating_stats: pd.DataFrame = self.ratings.groupby("ISBN").apply(self._group_isbn)
        self.rating_stats = self.rating_stats.loc[self.rating_stats["count"] >= minimal_isbn_count]
        self.cached.rating_stats.write_if_absent(self.rating_stats)
        return self.rating_stats

    def filter_searched_ratings(self) -> pd.DataFrame:
        """Filter ratings set to keep only the User-ID groups with books of interest"""
        print("Filtering rating groups which contain searched books.")
        searched_ratings = self.ratings.groupby("User-ID") \
            .filter(lambda group: any(group["ISBN"].isin(self.brd.searched_books["ISBN"])))
        self.searched_ratings = searched_ratings
        self.cached.ratings.write_if_absent(self.searched_ratings.set_index("ISBN"))
        return self.searched_ratings

    def find_close_books(self, minimum_close_book_count: int = 4, required_rating: int = 7) -> pd.DataFrame:
        """Find and score books which appear in the same rating groups as searched books

        :param minimum_close_book_count: drop books which appear less times than this threshold
        :param required_rating: filter out groups where searched books best rating doesn't reach this mark
        """
        # filter for good ratings and count measure group significance and size
        print("Calculating co-occurring weights and group sizes.")
        searched_mask: Callable[[pd.DataFrame], pd.Series] = lambda group: \
            group.loc[group["ISBN"].isin(self.searched_books["ISBN"]), "Book-Rating"]
        positive_searched_ratings = self.searched_ratings \
            .groupby("User-ID") \
            .filter(lambda uid: ((searched_mask(uid).max() >= required_rating) | all(searched_mask(uid) == 0)))
        positive_searched_ratings['Group-Weight'] = positive_searched_ratings.groupby("User-ID")['ISBN'] \
            .transform(lambda group: len(group.isin(self.searched_books['ISBN'])))
        positive_searched_ratings['Group-Size'] = positive_searched_ratings.groupby("User-ID")['ISBN'] \
            .transform(lambda group: len(group))
        # keep only related books (not searched), and compute weights for those books
        related_ratings: pd.DataFrame = \
            positive_searched_ratings.loc[~positive_searched_ratings["ISBN"].isin(self.searched_books["ISBN"])]
        print("Converting book selection into a dataframe of weights")
        close_books = related_ratings.groupby("ISBN").apply(self._group_isbn)
        # filter out books with less than 5 occurrences
        self.close_books: pd.DataFrame = close_books.loc[close_books["count"] > minimum_close_book_count]
        self.cached.close_books.write_if_absent(self.close_books)
        self.close_books['ISBN'] = self.close_books.index   # add isbn also as the index
        return self.close_books

    def rank_books(self, thresh_rating: int = 6) -> pd.DataFrame:
        """
        Rank books by their count for general books and
        rank close (co-occurring) books by count weighted by co-occurrence with searched books

        :param thresh_rating: remove close books with lower mean rating (i.e. not a good match)
        """
        mask: Callable[[pd.DataFrame], pd.Series] = lambda pdf: (pdf['rating-mean'] >= thresh_rating) | \
                                                                (pdf['rating-mean'].isna())
        print("Started comparing relevant books (close to searched) and general book set.")
        print(f"Initiating with {len(self.close_books.index)} close books "
              f"and {len(self.rating_stats.index)} general books.")
        # rank relevant book set (co-occurring with searched books)
        relevant_close: pd.DataFrame = self.close_books.loc[mask(self.close_books)]
        relevant_close['weighted-count'] = relevant_close['count'] * relevant_close['normalized-weight']
        relevant_close.loc[:, 'rank'] = relevant_close.loc[:, 'weighted-count'] \
            .rank(method='max') \
            .apply(lambda r: 100 * (r - 1) / len(relevant_close.index))
        relevant_close.set_index("ISBN", inplace=True)
        relevant_close.sort_index(inplace=True)
        # rank full book set
        general: pd.DataFrame = self.rating_stats.loc[mask(self.rating_stats)]
        general.loc[:, 'rank'] = general.loc[:, 'count'] \
            .rank(method='max') \
            .apply(lambda r: 100 * (r - 1) / len(general.index))
        print(f"After removing ratings < {thresh_rating}, there are {len(relevant_close.index)} close books "
              f"and {len(general.index)} books.")
        # filter non-close books, and compare ranking of close books in two sets
        relevant_general = general[general["ISBN"].isin(relevant_close.index)]
        relevant_general.set_index("ISBN", inplace=True)
        relevant_general.sort_index(inplace=True)
        relevant_close.loc[:, 'rank-gain'] = relevant_close.loc[:, 'rank'] - relevant_general.loc[:, 'rank']
        self.close_books_rank = relevant_close
        self.cached.ranked_books.write_if_absent(self.close_books_rank)
        self.close_books_rank['ISBN'] = self.close_books_rank.index  # add isbn also as column
        return self.close_books_rank

    def filter_rank_gain(self, book_limit: int = 500, only_rank_gain: bool = True,
                         plot_type: str = 'scatter') -> None:
        """
        Filter close books to those of positive rank gain, up to a limit

        :param book_limit: maximum number of filtered books
        :param only_rank_gain: filter only those with positive gain
        :param plot_type: plot (scatter, heatmap, or none)
        """
        if plot_type not in ['scatter', 'heatmap', 'none']:
            print(f"Allowed plot_type values are 'scatter', 'none', or 'heatmap'. "
                  f"You passed {plot_type}. Falling back to 'scatter'.")
            plot_type = 'scatter'
        mask: pd.Series = (self.close_books_rank['rank-gain'] >= 0) if only_rank_gain \
            else pd.Series(True, index=self.close_books_rank.index)
        relevant: pd.DataFrame = self.close_books_rank[mask].nlargest(book_limit, "rank-gain")
        relevant.reset_index(drop=True, inplace=True)
        print(f"Selected {len(relevant.index)} books, "
              f"of {(self.close_books_rank['rank-gain'] > 0).sum()} books with rank gain >= 0.")
        print(f"Mean rank-gain is {relevant['rank-gain'].mean()}, mean count is {relevant['count'].mean()}")
        # compute spread of rank and count in the selected set
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

    def link_relevant(self) -> pd.DataFrame:
        """
        Calculate links between each pair of pre-selected, relevant books
        """
        if self.relevant is None:
            self.filter_rank_gain()
        top_searched_isbn: List[str] = self.searched_ratings.loc[
            self.searched_ratings['ISBN'].isin(self.searched_books['ISBN'])] \
            .groupby('ISBN') \
            .apply(lambda isbn: len(isbn.index)) \
            .nlargest(self.best_searched) \
            .index \
            .to_list()
        isbn_list: List[str] = self.relevant['ISBN'].to_list() + top_searched_isbn
        print(f"Selected {len(isbn_list)} books, of which {len(top_searched_isbn)} are searched books.")
        reduced_ratings: pd.DataFrame = self.ratings.groupby('User-ID') \
                                                    .filter(lambda userid: len(userid.index) < 1000)
        reduced_ratings = reduced_ratings.loc[reduced_ratings['ISBN'].isin(isbn_list)] \
                                         .groupby('User-ID') \
                                         .filter(lambda uid: len(uid.index) > 1)
        print(f"Selected {len(reduced_ratings.index)} ratings for relevant books, with "
              f" < 1000 books per user, and > 1 selected book per user.")
        userid_group: pd.DataFrameGroupBy = reduced_ratings.groupby('User-ID')
        link_matrix: np.ndarray = np.zeros((len(isbn_list), len(isbn_list)))
        # double for loop to calculate the linking matrix
        for num_a, isbn_a in enumerate(isbn_list):
            print(f"\rProcessing {num_a} of {len(isbn_list)}.", end="")
            for num_b, isbn_b in enumerate(isbn_list[num_a:]):
                num_b = num_b + num_a
                link_matrix[num_a, num_b] = userid_group.apply(lambda uid: (any(uid['ISBN'] == isbn_a) &
                                                                            any(uid['ISBN'] == isbn_b))) \
                                                        .sum()
                link_matrix[num_b, num_a] = link_matrix[num_a, num_b]
        self.link_matrix = pd.DataFrame(link_matrix, columns=isbn_list, index=isbn_list)
        self.cached.link_matrix.write_if_absent(self.link_matrix)
        return self.link_matrix

    def print_relevant(self, top_matches: Optional[pd.DataFrame] = None):
        """Print the list of relevant books with their book weights"""
        if top_matches is None:
            if self.relevant is None:
                self.filter_rank_gain(plot_type='none')
            relevant_names: pd.DataFrame = self.brd.books[["ISBN", "Book-Title"]]\
                .merge((self.relevant[["ISBN", "rank-gain"]].nlargest(30, 'rank-gain')), on="ISBN")\
                .sort_values(by=['rank-gain'], ascending=False)
        else:
            relevant_names: pd.DataFrame = self.brd.books[["ISBN", "Book-Title"]]\
                .merge(top_matches[["ISBN", "book-weights"]], on="ISBN")\
                .sort_values(by=['book-weights'], ascending=False)
        print(f"The following books were selected {relevant_names.to_string()}")

    def explore_links(self, iterations: int = 50, top_books: int = 30, no_self_link: bool = True):
        """Explore links between books by applying link matrix

        :param iterations: maximum number of matrix multiplications
        :param top_books: how many best books to print out (with weights)
        :param no_self_link: consider only co-occurrence between candidate books, not single book occurrence
        """
        norm: Callable[[np.ndarray], np.ndarray] = lambda v: v / np.linalg.norm(v, ord=1)
        link_matrix: np.ndarray = self.link_matrix.to_numpy()
        nls = link_matrix.shape[0] - self.best_searched
        searched_matrix: np.ndarray = link_matrix[nls:, nls:]
        book_matrix: np.ndarray = link_matrix[:nls, :nls]
        mixing_matrix: np.ndarray = link_matrix[:nls, :]
        searched_weights: np.ndarray = norm(searched_matrix.diagonal())
        candidate_books = np.concatenate([np.zeros(nls), searched_weights])
        candidate_books = norm(mixing_matrix @ candidate_books)
        candidate_std: np.ndarray = np.zeros(iterations)
        if no_self_link:
            np.fill_diagonal(book_matrix, 0)
        for it in range(iterations):
            old_candidates = candidate_books.copy()
            candidate_books = norm(book_matrix @ candidate_books)
            candidate_std[it] = np.std(candidate_books)
            if np.linalg.norm(old_candidates - candidate_books, ord=1) < 1e-6:
                print(f"The weights change by less than 1e-6 after {it} iterations")
                break
        top_candidates: pd.DataFrame = pd.DataFrame(data=candidate_books, index=self.link_matrix.index[:nls],
                                                    columns=['book-weights'])
        top_candidates['ISBN'] = top_candidates.index
        top_matches: pd.DataFrame = top_candidates.nlargest(top_books, 'book-weights')
        self.print_relevant(top_matches)

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
    tolkien_names: List[str] = ["J. R. R. Tolkien", "J.R.R. Tolkien", "J.R.R.Tolkien",
                                "J.R.R. TOLKIEN", "John Ronald Reuel Tolkien"]
    ra = RatingAnalyser(author="Frank Herbert", book_title='Dune', cache_prefix="Dune")
    ra.run_full_analysis()
