import pandas as pd
from typing import Tuple, Optional, Union, List
from definitions import data
from functools import partial


class BookReviewData:

    # shorthand for csv reader formatting
    read_csv: partial = partial(pd.read_csv, encoding='latin-1', memory_map=True,
                                sep=';', doublequote=True, quotechar='"', escapechar="\\")

    def __init__(self, author: Union[List[str], str], book_title: Optional[str] = None,
                 filter_sole: bool = True, cache_prefix: Optional[str] = None,
                 strict_title: bool = False):
        # set up book titles and forms of author names to search for
        if book_title is None:
            self.all_author_books: bool = True
            self.book_title: Optional[str] = None
        else:
            self.all_author_books: bool = False
            self.book_title: str = book_title
        self.author: List[str] = [author] if isinstance(author, str) else author
        if cache_prefix is None:
            self.prefix: str = (self.author[0]+("All" if self.all_author_books else self.book_title))\
                .replace(" ", "").replace(".", "")
        else:
            self.prefix: str = cache_prefix
        print(f"Cached files prefix for this search set to {self.prefix}.")
        # reading data
        self.books: pd.DataFrame = self.read_csv(data / "BX-Books.csv")
        self.users: pd.DataFrame = self.read_csv(data / "BX-Users.csv")
        self.users['Location'] = self.users['Location'].apply(self._alpha_or_na)
        self.ratings: pd.DataFrame = self.read_csv(data / "BX-Full-Ratings.csv")
        if filter_sole: self._filter_sole_ratings()
        interest_mask: pd.DataFrame = self.books['Book-Author'].isin(self.author)
        if strict_title:
            title_mask: pd.DataFrame = self.books['Book-Title'] == book_title
        else:
            title_mask: pd.DataFrame = self.books['Book-Title'].str.contains(book_title, case=False)
        if not self.all_author_books:
            interest_mask = (interest_mask & title_mask)
        self.searched_books: pd.DataFrame = self.books[interest_mask]\
            .drop(["Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"],
                  axis=1)
        print("BookReviewData read.")
        print(f"Sole ratings {'were' if filter_sole else 'were not'} removed.")
        print(f"Set of interest are {book_title if not self.all_author_books else 'all author'} books.")

    def __str__(self):
        """String representation of class"""
        return f"BookReviewData for author name forms {self.author}, " \
               f"for {'all books' if self.all_author_books else self.book_title}, " \
               f"with cached data files prefixed with {self.prefix}.\n" \
               f"The list of matching books is:\n {self.searched_books.to_string()}"

    @classmethod
    def write_full_ratings(cls) -> pd.DataFrame:
        """
        Append user age and location to each rating; export to CSV if not prsent
        :return: data frame of full rating information
        """
        users: pd.DataFrame = cls.read_csv(data / "BX-Users.csv")
        users['Location'] = users['Location'].apply(cls._alpha_or_na)
        full_ratings: pd.DataFrame = cls.read_csv(data / "BX-Book-Ratings.csv")
        full_ratings = full_ratings.merge(users, on="User-ID").rename(columns={'Location': 'Country'})
        print(f"User and Ratings data were read and combined:\n {full_ratings}")
        print(f"Writing into file.")
        full_ratings.to_csv(data / "BX-Full-Ratings.csv", sep=";", index=False, encoding="latin-1", quoting=1,
                            escapechar="\\", na_rep="N/A")
        return full_ratings

    def get_mean_searched_rating(self, rating_age: Optional[Tuple[int, int]] = None, silent: bool = False) -> pd.DataFrame:
        """Count number of ratings and mean rating for each book in searched books"""
        relevant_ratings: pd.DataFrame = self.ratings.loc[(self.ratings['ISBN'].isin(self.searched_books['ISBN'])) &
                                                          (BookReviewData._is_in_age(self.ratings['Age'], rating_age))]
        grouped_relevant: pd.DataFrameGroupBy = relevant_ratings[["ISBN", "Book-Rating"]].groupby('ISBN')
        relevant_measure: pd.DataFrame = pd.DataFrame(columns=['ISBN', 'Rating-Count', 'Rating-Mean'])
        if len(grouped_relevant) > 0:
            relevant_measure.drop(columns='ISBN', inplace=True)
            relevant_measure['Rating-Count'] = grouped_relevant.count()
            relevant_measure['Rating-Mean'] = grouped_relevant\
                .apply(lambda isbn: isbn.loc[isbn['Book-Rating'] != 0, "Book-Rating"].mean())
        searched_books: pd.DataFrame = self.searched_books.copy()
        searched_books = searched_books.merge(relevant_measure, on='ISBN', how='left')
        if not silent:
            print(f"Computed columns of rating count and mean rating for searched books:\n {searched_books}")
            print("Ratings of value 0 were excluded from the mean computation.")
        return searched_books

    def filter_low_impact(self, least_count: int = 5) -> None:
        """Filter entries in searched_books set by rating count and presence of rating"""
        self.searched_books = self.searched_books.loc[(self.searched_books['Rating-Count'] > least_count) &
                                                      (~self.searched_books['Rating-Mean'].isnull())]
        print(f"Filtered out searched books with rating count < {least_count} or with N/A mean ratings.")

    @staticmethod
    def _is_in_age(age_col: Union[pd.DataFrame, pd.Series], bracket: Optional[Tuple[int, int]] = None) -> bool:
        """Create mask to filter ages within a particular bracket
        :param age_col: column containing age to bracket
        :param bracket: age bracket; all true if None
        """
        if bracket is None:
            return True
        else:
            if bracket[0] >= bracket[1]:
                raise RuntimeError("Age bracket lower value must be larger than upper value. "
                                   f"({bracket[0]}, {bracket[1]}) provided.")
            return (age_col >= bracket[0]) & (age_col <= bracket[1])

    @staticmethod
    def _alpha_or_na(label: str) -> str:
        """Reduce geolocation to country, or to N/A if not provided"""
        country_lbl: str = label.rsplit(", ")[-1]
        return country_lbl if country_lbl.replace(" ", "").isalpha() else 'N/A'

    def _filter_sole_ratings(self) -> None:
        """Remove ratings which are the only rating by a given user, or is only rating for an ISBN"""
        linked_ratings: pd.DataFrame = self.ratings.groupby("User-ID").filter(lambda x: len(x) > 1)
        self.ratings = linked_ratings.groupby("ISBN").filter(lambda x: len(x) > 1)


if __name__ == "__main__":
    tolkien_names: List[str] = ["J. R. R. Tolkien", "J.R.R. Tolkien", "J.R.R.Tolkien",
                                "J.R.R. TOLKIEN", "John Ronald Reuel Tolkien"]
    brd = BookReviewData(author=tolkien_names, book_title="lord of the rings", cache_prefix="LOTR")
    brd.get_mean_searched_rating()
    brd.filter_low_impact()
    print(f"Set of books to search:\n {brd.searched_books.to_string()}")
