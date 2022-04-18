import pandas as pd
from typing import Tuple, Optional, Union
from definitions import data
from functools import partial


class BookReviewData:

    read_csv: partial = partial(pd.read_csv, encoding='latin-1',
                                sep=';', doublequote=True, quotechar='"', escapechar="\\")

    def __init__(self, filter_sole: bool = True):
        self.books: pd.DataFrame = self.read_csv(data / "BX-Books.csv")
        self.users: pd.DataFrame = self.read_csv(data / "BX-Users.csv")
        self.users['Location'] = self.users['Location'].apply(self._alpha_or_na)
        self.ratings: pd.DataFrame = self.read_csv(data / "BX-Full-Ratings.csv")
        if filter_sole: self._filter_sole_ratings()
        self.lotr: pd.DataFrame = self.books[self.books['Book-Author']
            .isin(["J. R. R. Tolkien", "J.R.R. Tolkien", "J.R.R.Tolkien",
                   "J.R.R. TOLKIEN", "John Ronald Reuel Tolkien"])]\
            .drop(["Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"],
                  axis=1)
        print("BookReviewData read.")

    @classmethod
    def write_full_ratings(cls):
        users: pd.DataFrame = cls.read_csv(data / "BX-Users.csv")
        users['Location'] = users['Location'].apply(cls._alpha_or_na)
        full_ratings: pd.DataFrame = cls.read_csv(data / "BX-Book-Ratings.csv")
        full_ratings['Country'] = "N/A"
        full_ratings['Age'] = None
        for i, irow in enumerate(users.iterrows()):
            print(f"\rProcessing line {i} of {len(users.index)}", end='')
            row: pd.Series = irow[1]
            full_ratings.loc[full_ratings['User-ID'] == row['User-ID'], ['Country', 'Age']] = \
                row[['Location', 'Age']].to_list()
        full_ratings.to_csv(data / "BX-Full-Ratings.csv", sep=";", index=False, encoding="latin-1", quoting=1,
                            escapechar="\\", na_rep="N/A")

    def get_mean_rating(self, rating_age: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        self.lotr['Rating-Count'] = self.lotr['ISBN']\
            .apply(lambda isbn: len(self.ratings.loc[(self.ratings['ISBN'] == isbn) &
                                                     (BookReviewData._is_in_age(self.ratings['Age'], rating_age))]))
        self.lotr['Rating-Mean'] = self.lotr['ISBN'] \
            .apply(lambda isbn: self.ratings
                   .loc[(self.ratings['ISBN'] == isbn) &
                        (self.ratings['Book-Rating'] != 0) &
                        (BookReviewData._is_in_age(self.ratings['Age'], rating_age)), "Book-Rating"].mean())
        return self.lotr

    def filter_low_impact(self, least_count: int = 5) -> None:
        self.lotr = self.lotr.loc[(self.lotr['Rating-Count'] > least_count) & (~self.lotr['Rating-Mean'].isnull())]

    @staticmethod
    def _is_in_age(age_col: Union[pd.DataFrame, pd.Series], bracket: Optional[Tuple[int, int]] = None) -> bool:
        if bracket is None:
            return True
        else:
            if bracket[0] >= bracket[1]:
                raise RuntimeError("Age bracket lower value must be larger than upper value. "
                                   f"({bracket[0]}, {bracket[1]}) provided.")
            return (age_col >= bracket[0]) & (age_col <= bracket[1])

    @staticmethod
    def _alpha_or_na(label: str) -> str:
        country_lbl: str = label.rsplit(", ")[-1]
        return country_lbl if country_lbl.replace(" ", "").isalpha() else 'N/A'

    def _filter_sole_ratings(self) -> None:
        linked_ratings: pd.DataFrame = self.ratings.groupby("User-ID").filter(lambda x: len(x) > 1)
        self.ratings = linked_ratings.groupby("ISBN").filter(lambda x: len(x) > 1)


if __name__ == "__main__":
    # BookReviewData.write_full_ratings()
    brd = BookReviewData()
    # brd.get_mean_rating()
    # brd.filter_low_impact()

    pass
