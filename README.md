# bookcrawler

A short program to analyse book relationships based on the database available 
[here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

The program is split into two classes, `BookReviewData` and `RatingAnalyser`.

## BookReviewData

Defined in `data_reader` loads the data about rating users (BX-Users), rated books (BX-Books) and the 
actual set of all ratings (BX-Book-Ratings). It cleans and filters them, and is set up to prepare rating combined
with user geolocation and age (BX-Full-Ratings), which is used later.

## RatingAnalyser

Performs the actual analysis for a particular book and author. It searches the ratings database and finds relevant 
related books. Since the process can be lengthy, it caches intermediate results, prefixed by an identifier of the current 
analysis. To do this, it uses `Cache` class.

To perform the analysis, one needs to create a `RatingAnalyser` instance (see the docstrings for details). This only 
creates the object, and loads the data from cache if any.

To run the full analysis, use the `run_full_analysis` method (see docstrings for details). It checks if cached data is 
available and recalculates if necessary. Note that the threshold can have a big impact on the results and the time of 
computation.

## Bugs and todos

### Caching

The caching system is very precarious. There are sometimes exceptions raised, when calculated and read data don't match
in a column which was used as an index. (Should be fixed by now, but still.) The whole caching class should be overhauled.

### Testing

There are not tests, and the program was not thoroughly tested on many books. So quite many things might have 
been missed and could cause failures.

