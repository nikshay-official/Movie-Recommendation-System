from flask import Flask, render_template, request
import pickle
import requests

# Initialize app
app = Flask(__name__)

# Load saved data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# Home route
@app.route('/')
def home():
    movie_list = movies['title'].values
    return render_template('index.html', movies=movie_list)


# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie = request.form.get('movie')
    names, posters = recommend(movie)

    return render_template(
        'index.html',
        movies=movies['title'].values,
        recommendations=names,
        posters=posters
    )

import requests

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d14e3d76543380a02dfae74019898a46&language=en-US"
    data = requests.get(url).json()
    
    poster_path = data.get('poster_path')
    
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)