import pandas as pd
import urllib.parse
import requests
import time
from bs4 import BeautifulSoup
import os
import csv
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def url_encode(input_string):
    return urllib.parse.quote(input_string)

def get_imdb_id(movie_name):
    search_url = f"https://www.imdb.com/find/?q={url_encode(movie_name)}&s=tt&ref_=fn_tt"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        logger.info(f"Searching IMDb ID for: {movie_name}")
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try new IMDb layout selector 
        result = soup.select_one(".ipc-metadata-list-summary-item__t")
        if result and result.get("href"):
            href = result['href']
        else:
            # Fallback to older layout
            result = soup.select_one(".findList tr.findResult td.result_text a")
            if result and result.get("href"):
                href = result['href']
            else:
                logger.warning(f"No IMDb ID found for '{movie_name}'")
                return None

        imdb_id = href.split('/')[2]
        logger.info(f"✅ Found IMDb ID: {imdb_id}")
        return imdb_id

    except Exception as e:
        logger.error(f"❌ IMDb search error for '{movie_name}': {e}")
        return None


def scrape_imdb_reviews(imdb_id, max_reviews=5):
    url = f"https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ql_urv&sort=num_votes%2Cdesc"
    headers = {"User-Agent": "Mozilla/5.0"}
    reviews = []

    try:
        logger.info(f"Fetching top {max_reviews} reviews for {imdb_id}")
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        review_blocks = soup.select('article.user-review-item')[:max_reviews]

        if not review_blocks:
            logger.warning(f"No reviews found for {imdb_id}")
            return []

        for block in review_blocks:
            title_elem = block.select_one('a.ipc-title-link-wrapper h3') or block.select_one('.title')
            rating_elem = block.select_one('.ipc-rating-star--base')
            content_elem = block.select_one('.ipc-html-content-inner-div') or block.select_one('.text.show-more__control')
            name_elem = block.select_one('.display-name-link a')
            date_elem = block.select_one('.review-date')

            reviews.append({
                "imdb_id": imdb_id,
                "review_title": title_elem.text.strip() if title_elem else None,
                "review_rating": rating_elem.text.split("/")[0].strip() if rating_elem else None,
                "review_text": content_elem.text.strip() if content_elem else None,
                "reviewer_name": name_elem.text.strip() if name_elem else None,
                "review_date": date_elem.text.strip() if date_elem else None
            })

        logger.info(f"✅ Retrieved {len(reviews)} reviews for {imdb_id}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch reviews for {imdb_id}: {e}")

    return reviews


def append_to_csv(file_path, data_dict, fieldnames):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)


def append_many_to_csv(file_path, data_list, fieldnames):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in data_list:
            writer.writerow(row)


def run_pipeline():
    input_path = "/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/iitp/SonyResearchMovieRecommendation/dataset/movies.dat"
    imdb_id_csv = "imdb_ids.csv"
    review_csv = "imdb_reviews.csv"
    failed_txt = "failed_movies.txt"

    print("📥 Loading movie dataset...")
    df = pd.read_csv(input_path, sep="::", engine="python", names=["MovieID", "Title", "Genres"], encoding="latin-1")
    df['CleanTitle'] = df['Title'].str.extract(r'^(.*?)(?: \(\d{4}\))?$')[0].str.strip()

    imdb_fieldnames = ["MovieID", "OriginalTitle", "CleanTitle", "IMDb_ID"]
    review_fieldnames = ["imdb_id", "review_title", "review_rating", "review_text", "reviewer_name", "review_date"]
    failed_titles = []

    print(f"🎬 Total movies to process: {len(df)}\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Processing Movies"):
        movie_id = row['MovieID']
        original_title = row['Title']
        clean_title = row['CleanTitle']

        print(f"\n🔎 [{idx + 1}] Searching IMDb ID for: {clean_title}")
        imdb_id = get_imdb_id(clean_title)
        time.sleep(1)

        if imdb_id:
            print(f"✅ IMDb ID found: {imdb_id}")
            imdb_row = {
                "MovieID": movie_id,
                "OriginalTitle": original_title,
                "CleanTitle": clean_title,
                "IMDb_ID": imdb_id
            }
            append_to_csv(imdb_id_csv, imdb_row, imdb_fieldnames)
            print(f"💾 IMDb ID written to {imdb_id_csv}")

            print(f"📝 Fetching top reviews for {imdb_id}")
            reviews = scrape_imdb_reviews(imdb_id, max_reviews=5)
            if reviews:
                append_many_to_csv(review_csv, reviews, review_fieldnames)
                print(f"💾 {len(reviews)} reviews written to {review_csv}")
            else:
                print("⚠️ No reviews found.")

        else:
            print(f"❌ IMDb ID not found for: {clean_title}")
            failed_titles.append(clean_title)

        time.sleep(1)

    if failed_titles:
        with open(failed_txt, "w", encoding="utf-8") as f:
            for title in failed_titles:
                f.write(title + "\n")
        print(f"\n⚠️ {len(failed_titles)} titles failed. Saved to {failed_txt}")
    else:
        print("\n✅ All movies processed successfully!")


if __name__ == "__main__":
    run_pipeline()