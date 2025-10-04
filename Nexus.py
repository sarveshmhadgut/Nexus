import time
import random
import joblib
import requests
import pandas as pd
from PIL import Image
import streamlit as st
from io import BytesIO
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

API_KEY = st.secrets["API_KEY"]

if not API_KEY:
    st.error(
        "API key not configured. Set API_KEY in .streamlit/secrets.toml or .env, "
        "or export API_KEY in your shell."
    )
    st.stop()

try:
    with open("./static/style.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("static/style.css not found. Skipping custom styles.")

# Session state init
if "page" not in st.session_state:
    st.session_state.page = "main"
if "movie_name" not in st.session_state:
    st.session_state.movie_name = ""
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Load dataset
try:
    data = pd.read_csv("./datasets/main_data.csv")
except FileNotFoundError:
    st.error("datasets/main_data.csv not found. Please add your dataset.")
    st.stop()
movie_titles = data["movie_title"].tolist()


# Robust HTTP session with retries
def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 504],
        allowed_methods=frozenset(["GET", "POST"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


@st.cache_data
def create_similarity():
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data["comb"])
    similarity = cosine_similarity(count_matrix)
    return similarity


@st.cache_resource
def load_model():
    try:
        return joblib.load("nlp_model.pkl")
    except Exception as e:
        st.warning(f"Unable to load nlp_model.pkl: {e}")
        return None


def rcmd(m):
    m = m.lower()
    try:
        similarity = create_similarity()
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return []
    if m not in data["movie_title"].unique():
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with another movie."
    i = data.loc[data["movie_title"] == m].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)[1:11]
    return [data["movie_title"][a] for a, _ in lst]


def rcmd_with_model(m):
    m = m.lower()
    if m not in data["movie_title"].unique():
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with another movie."
    try:
        similarity = create_similarity()
        i = data.loc[data["movie_title"] == m].index[0]
        lst = sorted(list(enumerate(similarity[i])), key=lambda x: x[1], reverse=True)[
            1:11
        ]
        recommended_movies = [data["movie_title"][a] for a, _ in lst]
        nlp_model = load_model()
        if nlp_model is None:
            return recommended_movies
        processed = []
        for title in recommended_movies:
            desc_series = data.loc[data["movie_title"] == title]["description"]
            if desc_series.empty:
                score = 0.0
            else:
                score = float(nlp_model.predict([desc_series.values[0]])[0])
            processed.append({"title": title, "predicted_score": score})
        processed.sort(key=lambda x: x["predicted_score"], reverse=True)
        return [m["title"] for m in processed]
    except Exception as e:
        st.error(f"Error in recommendations: {e}")
        return []


def fetch_movie_details(movie_name):
    session = create_session()
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("results"):
            st.warning("No movies found. Try another search.")
            return None
        movie = payload["results"][0]
        movie_id = movie["id"]
        movie["original_title"] = movie.get(
            "original_title", movie.get("title", movie_name)
        )
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        details_resp = session.get(details_url, timeout=20)
        details_resp.raise_for_status()
        movie_details = details_resp.json()
        genres = movie_details.get("genres", [])
        movie["genres"] = [g["name"] for g in genres]
        movie["poster_path"] = movie.get("poster_path") or movie_details.get(
            "poster_path"
        )
        return movie
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie details: {e}")
        return None


@st.cache_data
def fetch_recommendations(movie_id, retries=3, delay=5):
    session = create_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={API_KEY}"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("results", [])[:5]
        except requests.exceptions.RequestException as e:
            st.warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
    st.error(
        f"Failed to fetch recommendations for movie ID {movie_id} after {retries} attempts."
    )
    return []


def fetch_cast_details(movie_id, retries=3, delay=5):
    session = create_session()
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            cast = payload.get("cast", [])[:5]
            return [
                {
                    "name": c["name"],
                    "character": c.get("character", ""),
                    "profile_path": c.get("profile_path"),
                }
                for c in cast
            ]
        except requests.exceptions.RequestException as e:
            st.warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
    st.error(
        f"Failed to fetch cast details for movie ID {movie_id} after {retries} attempts."
    )
    return []


def fetch_poster(movie_name):
    session = create_session()
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("results"):
            return None, movie_name
        first = payload["results"][0]
        title = first.get("title", movie_name)
        poster_path = first.get("poster_path")
        if not poster_path:
            return None, title
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        poster_resp = session.get(poster_url, timeout=10)
        if poster_resp.status_code == 200:
            return poster_resp.content, title
        return None, title
    except requests.exceptions.RequestException as e:
        # Do not recurse on error; just return no poster
        return None, movie_name


def fetch_posters_in_parallel(movie_names):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_poster, name): name for name in movie_names}
        for future in futures:
            try:
                results.append(future.result())
            except Exception:
                results.append((None, futures[future]))
    return results


def display_movie_details(movie_details):
    mov_details = {}
    with st.spinner("Loading..."):
        if movie_details.get("poster_path"):
            poster_url = (
                f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
            )
            response = requests.get(poster_url)
            if response.status_code == 200:
                mov_details["poster"] = Image.open(BytesIO(response.content))
        else:
            st.text("No poster available")
        mov_details["details"] = [
            movie_details.get("overview", ""),
            movie_details.get("vote_average", "N/A"),
            movie_details.get("release_date", "N/A"),
            movie_details.get("vote_count", "N/A"),
            movie_details.get("original_language", "N/A"),
        ]
        genres = movie_details.get("genres", [])
        if genres:
            mov_details["genre"] = [genre for genre in genres]
        cast_details = fetch_cast_details(movie_details["id"])
        mov_details["cast"] = []
        if cast_details:
            cast_cols = st.columns(len(cast_details))
            for i, cast_member in enumerate(cast_details):
                with cast_cols[i]:
                    if cast_member["profile_path"]:
                        cast_image_url = f"https://image.tmdb.org/t/p/w500{cast_member['profile_path']}"
                        response = requests.get(cast_image_url)
                        if response.status_code == 200:
                            mov_details["cast"].append(
                                [cast_member, Image.open(BytesIO(response.content))]
                            )

    st.markdown("## Overview:")
    col1, col2 = st.columns([1, 2])
    with col1:
        if mov_details.get("poster"):
            st.image(mov_details["poster"], width=200)
    with col2:
        st.markdown(f"\t {mov_details['details'][0]}")
        st.markdown(f"**Rating:** {mov_details['details'][1]} / 10")
        st.markdown(f"**Release Date:** {mov_details['details'][2]}")
        st.markdown(f"**Vote Count:** {mov_details['details'][3]}")
        st.markdown(f"**Original Language:** {mov_details['details'][4]}")
        if mov_details.get("genre"):
            st.markdown(f"**Genres:** {', '.join(mov_details['genre'])}")

    st.markdown("## Top Casts:")
    if mov_details["cast"]:
        cols = st.columns(len(mov_details["cast"]))
        for i, cast in enumerate(mov_details["cast"]):
            with cols[i]:
                st.image(cast[1], width=100)
                st.text(cast[0]["name"])


def display_recommended_movie(recommendations):
    st.markdown("## Recommendations:")
    row_count = 2
    movies_per_row = 5
    rows = [
        recommendations[i : i + movies_per_row]
        for i in range(0, len(recommendations), movies_per_row)
    ][:row_count]
    recommend_details = []
    with st.spinner("Fetching recommendations..."):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(fetch_poster, rec): rec for rec in recommendations
            }
            for future in futures:
                try:
                    poster_data, fetched_name = future.result()
                    if poster_data:
                        recommend_details.append(
                            [fetched_name, Image.open(BytesIO(poster_data))]
                        )
                    else:
                        recommend_details.append([fetched_name, None])
                except Exception:
                    recommend_details.append([futures[future], None])

    for row_index, row in enumerate(rows):
        cols = st.columns(movies_per_row)
        for i, _ in enumerate(row):
            with cols[i]:
                idx = row_index * movies_per_row + i
                if idx < len(recommend_details):
                    movie_name, poster_image = recommend_details[idx]
                    if poster_image:
                        st.image(poster_image)
                    else:
                        st.text("No Poster Available")
                    unique_key = f"{movie_name}-{row_index}-{i}"
                    if st.button(f"{movie_name}", key=unique_key):
                        st.session_state.selected_movie = movie_name
                        print_data(movie_name)
                        st.session_state.movie_name = movie_name
                        with st.spinner("Redirecting..."):
                            st.rerun()


def print_data(movie_name):
    movie_details = fetch_movie_details(movie_name)
    if st.button("Get recommendations"):
        if movie_details:
            st.session_state.selected_movie = movie_details
            display_movie_details(movie_details)
            recommendations = rcmd(movie_name)
            if isinstance(recommendations, list):
                display_recommended_movie(recommendations)
            else:
                st.warning(recommendations)


def main_page():
    st.title("Nexus")
    movie_name = st.text_input(
        "Enter a Movie Name", value=st.session_state.get("movie_name", "")
    )
    if movie_name:
        st.session_state.movie_name = movie_name
        st.session_state.page = "recommendations"
        with st.spinner("Redirecting..."):
            st.rerun()
    if not st.session_state.get("suggestions"):
        st.write("Suggested Movies:")
        st.session_state.suggestions = random.sample(
            movie_titles, min(5, len(movie_titles))
        )
    cols = st.columns(len(st.session_state.suggestions))
    suggestions_data = []
    if st.session_state.suggestions:
        with st.spinner("Fetching suggestions... "):
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(fetch_poster, s): s
                    for s in st.session_state.suggestions
                }
                for future in futures:
                    try:
                        poster_data, fetched_name = future.result()
                        if poster_data:
                            suggestions_data.append(
                                [fetched_name, Image.open(BytesIO(poster_data))]
                            )
                        else:
                            suggestions_data.append([fetched_name, None])
                    except Exception:
                        suggestions_data.append([futures[future], None])
        for i, suggestion in enumerate(suggestions_data):
            with cols[i]:
                if suggestion[1]:
                    st.image(suggestion[1], width=200)
                else:
                    st.text("No Poster Available")
                if st.button(f"{suggestion[0]}", key=f"random-{suggestion[0]}"):
                    st.session_state.movie_name = suggestion[0]
                    st.session_state.page = "recommendations"
                    with st.spinner("Redirecting..."):
                        st.rerun()


def recommendations_page():
    if st.button("Home"):
        st.session_state.page = "main"
        st.session_state.movie_name = ""
        st.session_state.suggestions = []
        with st.spinner("Redirecting..."):
            st.rerun()
    movie_name = st.session_state.movie_name
    movie_details = fetch_movie_details(movie_name)
    if movie_details:
        st.title(f"{movie_details.get('title', movie_name)}")
        display_movie_details(movie_details)
        recommendations = rcmd(movie_name)
        if isinstance(recommendations, list):
            display_recommended_movie(recommendations)


if st.session_state.page == "main":
    st.session_state.selected_movie = None
    st.session_state.movie_name = ""
    main_page()
elif st.session_state.page == "recommendations":
    st.session_state.suggestions = []
    recommendations_page()
