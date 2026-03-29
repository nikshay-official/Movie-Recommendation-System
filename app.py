from flask import Flask, render_template, request
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))

# Recompute similarity (since we removed similarity.pkl)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)


# Fetch poster
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=YOUR_API_KEY&language=en-US"
        data = requests.get(url, timeout=5).json()

        poster_path = data.get('poster_path')

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"

    except:
        return "https://via.placeholder.com/500x750?text=Error"


# Recommend function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)),
                        key=lambda x: x[1],
                        reverse=True)[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# Routes
@app.route('/')
def home():
    return render_template('index.html', movies=movies['title'].values)


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


# Run app
if __name__ == "__main__":
    app.run(debug=True)