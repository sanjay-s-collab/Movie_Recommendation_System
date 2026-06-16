import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")


@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[['movie_id', 'title', 'overview',
                     'genres', 'keywords', 'cast', 'crew']]

    movies.dropna(inplace=True)

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert_cast(obj):
        L = []
        count = 0
        for i in ast.literal_eval(obj):
            if count != 3:
                L.append(i['name'])
                count += 1
            else:
                break
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies[feature] = movies[feature].apply(
            lambda x: [i.replace(" ", "") for i in x]
        )

    movies['tags'] = (
        movies['overview']
        + movies['genres']
        + movies['keywords']
        + movies['cast']
        + movies['crew']
    )

    new_df = movies[['movie_id', 'title', 'tags']]

    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words='english')

    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


new_df, similarity = load_data()


def recommend(movie):

    movie = movie.lower()

    movie_index = None

    for i in range(len(new_df)):
        if new_df.iloc[i].title.lower() == movie:
            movie_index = i
            break

    if movie_index is None:
        return []

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(
            new_df.iloc[i[0]].title
        )

    return recommendations


movie_name = st.text_input("Enter Movie Name")

if st.button("Recommend"):

    recommendations = recommend(movie_name)

    if len(recommendations) == 0:
        st.error("Movie not found!")
    else:
        st.subheader("Recommended Movies")

        for movie in recommendations:
            st.write(movie)