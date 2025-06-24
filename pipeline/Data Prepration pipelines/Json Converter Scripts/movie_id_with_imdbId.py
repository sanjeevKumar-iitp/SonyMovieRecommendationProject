import csv
import json
import re

# File paths
DAT_FILE = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/dataset/movies.dat'
CSV_FILE = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/pipeline/output/csv/imdb_ids.csv'
TSV_FILE = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/dataset/imdb_title_master_data.tsv'
OUTPUT_JSON = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/pipeline/output/json/movie_master_data.json'

# Dictionary to store final JSON data
movies_json = {}

# Helper: Clean title function
def get_clean_title(title):
    # Remove year in parentheses at the end
    title = re.sub(r'\s+$[0-9]{4}$', '', title)
    return title.strip()

# Step 1: Read movies.dat
with open(DAT_FILE, 'r', encoding='latin1') as f:
    for line in f:
        parts = line.strip().split('::')
        if len(parts) != 3:
            continue
        movie_id, title, genres = parts
        movies_json[movie_id] = {
            "movie_id": movie_id,
            "title": title,
            "clean_title": get_clean_title(title),
            "genres": genres,
            "imdb_id": None
        }

# Step 2: Try to fill IMDb IDs and clean titles from movies.csv
csv_title_to_imdb = {}
csv_title_to_clean = {}

with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        original_title = row['OriginalTitle']
        clean_title = row['CleanTitle']
        imdb_id = row['IMDb_ID']
        csv_title_to_imdb[original_title] = imdb_id
        csv_title_to_imdb[clean_title] = imdb_id
        csv_title_to_clean[original_title] = clean_title
        csv_title_to_clean[clean_title] = clean_title

# Match titles from movies.dat with CSV using either title version
for movie_id, data in movies_json.items():
    title = data['title']
    if title in csv_title_to_imdb:
        data['imdb_id'] = csv_title_to_imdb[title]
        data['clean_title'] = csv_title_to_clean[title]

# Step 3: For remaining missing IMDb IDs, use IMDb master TSV (streaming mode for performance)
still_missing = {mid for mid, data in movies_json.items() if data['imdb_id'] is None}

if still_missing:
    print("Trying to match remaining movies with IMDb master TSV...")
    matched = set()
    with open(TSV_FILE, 'r', encoding='utf-8', errors='ignore') as tsvfile:
        headers = tsvfile.readline().strip().split('\t')
        title_idx = headers.index('primaryTitle')
        orig_title_idx = headers.index('originalTitle')
        id_idx = headers.index('tconst')
        type_idx = headers.index('titleType')

        for line in tsvfile:
            cols = line.strip().split('\t')
            if cols[type_idx] != 'movie':
                continue

            primary_title = cols[title_idx]
            original_title_tsv = cols[orig_title_idx]
            imdb_id = cols[id_idx]

            for movie_id in list(still_missing):
                data = movies_json[movie_id]
                if data['title'] == primary_title or data['title'] == original_title_tsv:
                    data['imdb_id'] = imdb_id
                    # Already has clean_title from DAT file; optionally override here
                    matched.add(movie_id)
                    still_missing.remove(movie_id)
                    print(f"Matched '{data['title']}' -> {imdb_id}")
                    break  # Break inner loop to avoid redundant checks

            if not still_missing:
                print("All missing IMDb IDs matched.")
                break

# Final Output
with open(OUTPUT_JSON, 'w', encoding='utf-8') as jsonfile:
    json.dump(movies_json, jsonfile, indent=2)

print(f"Final JSON saved to '{OUTPUT_JSON}'.")