### SETUP ###

# Import libraries
import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Import data
movie_ratings_df = pd.read_csv(r'C:\Users\Terence\Desktop\Rec_system\movie_ratings_data_set.csv')
movies_df = pd.read_csv(r'C:\Users\Terence\Desktop\Ex_Files_ML_EssT_Recommendations\Ex_Files_ML_EssT_Recommendations\Exercise Files\Chapter 5\movies.csv', index_col='movie_id')

# Create Sparse Matrix (UserID x MovieNames)
user_ratings_df = pd.pivot_table(movie_ratings_df, index='user_id', columns='movie_id', aggfunc=np.max)

# Matrix Factorization to get U x M
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(user_ratings_df.values, num_features=11, regularization_amount=1.1)

# Predict all user ratings
predicted_ratings = np.matmul(U,M)

####################################################################################################################################################################################

## All Time Most Popular ###
def allTime():
    print('Here are the 15 most popular movies:')
    most_popular = pd.DataFrame(movie_ratings_df.groupby('movie_id')['value'].count())
    most_popular = most_popular.join(movies_df, on = 'movie_id')
    most_popular = most_popular.sort_values('value', ascending = False)
    most_popular = most_popular[['title', 'genre']]
    print(most_popular.head(15))

####################################################################################################################################################################################

### Most Highly Rated ###
def highestRated():
    print('Here are the 15 highest rated movies:')
    highest_ratings = pd.DataFrame(movie_ratings_df.groupby('movie_id')['value'].mean())
    highest_ratings = highest_ratings.join(movies_df, on='movie_id')
    highest_ratings = highest_ratings.sort_values('value', ascending = False)
    print(highest_ratings.head(15))

####################################################################################################################################################################################

### Recommended for You ###

def userRecommendations():
    ## Prompt User to Choose user_id to search
    min_user_id = movie_ratings_df.user_id.min()
    max_user_id = movie_ratings_df.user_id.max()

    print('Enter a user_id between ' + str(min_user_id) + ' and ' + str(max_user_id) + ':')
    search_user_id = int(input())

    ## Show previously watched movies
    print('Movies previously watched by user_id ' + str(search_user_id) + ':')
    watched_movies = movie_ratings_df[movie_ratings_df['user_id'] == search_user_id]
    watched_movies = watched_movies.join(movies_df, on = 'movie_id')
    print(watched_movies.title)

    ## Show recommended movies
    input("Press enter to continue.")
    print("Here are some recommended movies based on past movie reviews:")

    # Merging predicted user ratings with movie list
    user_ratings = predicted_ratings[search_user_id - 1]
    movies_df['ratings'] = user_ratings

    # Removing movies that have already been watched
    already_watched = watched_movies['movie_id']
    recommended_movies = movies_df[movies_df.index.isin(already_watched) == False]

    # Sort recommended movies from highest to lowest rating
    recommended_movies = recommended_movies.sort_values(by='ratings', ascending = False)
    print(recommended_movies.head(10))

####################################################################################################################################################################################

### Explore and find similar favourites ###

def similarMovies(M):
    # Prompt user for movie id
    print(movies_df)
    movie_id = int(input('Choose a movie to find similar movies to (USE MOVIE ID #): '))
    movie_info = movies_df.loc[movie_id]

    print("We are finding movies similar to this movie:")
    print("Movie title: {}".format(movie_info.title))
    print("Genre: {}".format(movie_info.genre))

    ## Find similar movies
    # 1) Get features for movie
    M = np.transpose(M)
    movie_features = M[movie_id - 1]

    # 2) Subtract current movie features from every other movie features and take absolute value
    difference = M - movie_features
    absolute_difference = np.abs(difference)

    # 3) Sum all features to get a 'total difference score' for each movie
    total_difference = np.sum(absolute_difference, axis=1)

    # 4) Create a new column with difference score for each movie
    movies_df['difference_score'] = total_difference

    # 5) Sort movies from least different to most different and show results
    sorted_movie_list = movies_df.sort_values('difference_score', ascending = True)
    print('The ten most similar movies are: ')
    print(sorted_movie_list[['title', 'genre', 'difference_score']][0:11])

####################################################################################################################################################################################

### Main ###
def main():
    print('1) See All Time Most Popular Movies')
    print('2) See Highest Rated Movies')
    print('3) Find Recommended Movies based on User')
    print('4) Find Recommended Movies based on Movie')
    menu_choice = int(input('Choose from the following options above: '))
    if menu_choice == 1:
        allTime()
    elif menu_choice == 2:
        highestRated()
    elif menu_choice == 3:
        userRecommendations()
    elif menu_choice == 4:
        similarMovies(M)

print('Welcome to my recommendation system.')
cont = 'Y'
while cont == 'Y':
    main()
    cont = input('Would you like to try another option? (Y/N)').upper()
print('See you soon!')






