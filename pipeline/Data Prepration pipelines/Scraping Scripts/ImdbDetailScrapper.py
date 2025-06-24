import pandas as pd
import urllib.parse
import requests
import time
from bs4 import BeautifulSoup
import ImdbDetailScrapper
import csv
import os
from tqdm import tqdm


def url_encode(input_string):
    return urllib.parse.quote(input_string)


def search_imdb_url(movie_name):
    search_url = f"https://www.imdb.com/find/?q={url_encode(movie_name)}&s=tt&ref_=fn_tt"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        result = soup.select_one(".ipc-metadata-list-summary-item__t")
        if result and result.get("href"):
            return f"https://www.imdb.com{result['href']}"

        result = soup.select_one(".findList tr.findResult td.result_text a")
        if result and result.get("href"):
            return f"https://www.imdb.com{result['href']}"

    except Exception as e:
        print(f"❌ IMDb search error for '{movie_name}': {e}")

    return None


def write_to_csv(file_path, data_dict, fieldnames):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)


def process_movies(movie_titles, completed_titles, output_file, retry_mode=False):
    failed_movies = []

    for idx, movie_name in enumerate(tqdm(movie_titles, desc="🎬 Processing", unit="movie")):
        if not retry_mode and movie_name in completed_titles:
            continue

        print(f"\n🔍 [{idx + 1}/{len(movie_titles)}] Searching IMDb for: {movie_name}")
        imdb_url = search_imdb_url(movie_name)

        if not imdb_url:
            print(f"❌ No IMDb result found for: {movie_name}")
            failed_movies.append(movie_name)
            continue

        movie_data = ImdbDetailScrapper.fetch_imdb_movie_details(imdb_url)

        if movie_data:
            print(f"✅ Fetched: {movie_data['title']}")

            # Add genre to movie_data
            movie_data['genres'] = genre_map.get(movie_name, '')

            write_to_csv(output_file, movie_data, [
                "title", "storyline", "description", "directors",
                "writers", "cast", "tagline", "genres"
            ])
        else:
            print(f"❌ Failed to fetch details for: {movie_name}")
            failed_movies.append(movie_name)

        time.sleep(2)

    return failed_movies


def run_imdb_pipeline():
    # Load movies
    movies_df = pd.read_csv(
        '../dataset/movies.dat',
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1')

    movies_df['CleanTitle'] = movies_df['Title'].str.extract(r'^(.*?)(?: \(\d{4}\))?$')[0].str.strip()
    movie_titles = movies_df['CleanTitle'].dropna().unique().tolist()

    # Build a mapping of CleanTitle to Genre
    global genre_map
    genre_map = dict(zip(movies_df['CleanTitle'], movies_df['Genres']))

    # Output
    output_file = "imdb_movie_metadata.csv"
    completed_titles = set()

    if os.path.isfile(output_file):
        existing_df = pd.read_csv(output_file)
        completed_titles = set(existing_df['title'].dropna().tolist())

    print(f"📄 Starting pipeline. Movies to process: {len(movie_titles)}")

    # First pass
    failed_movies = process_movies(movie_titles, completed_titles, output_file)

    # Retry failed
    if failed_movies:
        print(f"\n🔁 Retrying {len(failed_movies)} failed movies...\n")
        time.sleep(3)
        retry_failed = process_movies(failed_movies, set(), output_file, retry_mode=True)

        if retry_failed:
            print(f"\n❌ Still failed after retry: {len(retry_failed)}")
            with open("failed_movies.txt", "w", encoding="utf-8") as f:
                for title in retry_failed:
                    f.write(title + "\n")
        else:
            print("✅ All failed movies fetched on retry.")
    else:
        print("✅ All movies processed successfully on first attempt.")


if __name__ == "__main__":
    run_imdb_pipeline()
