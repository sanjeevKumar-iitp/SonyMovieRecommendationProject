# ----------------------------
# jsonDataConverter.py
# Orchestrated JSON Converter for Clustering Output
# Converts CSV outputs into structured JSON for downstream use
# ----------------------------

import os
import pandas as pd
import json
from collections import defaultdict

# ----------------------------
# Step 1: Define Project Paths
# ----------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "..", "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json")

os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

CLUSTER_CSV_PATH = os.path.join(OUTPUT_DIR, "user_sorted_clusters.csv")
USERS_DAT_PATH = os.path.join(DATASET_DIR, "users.dat")
IMDB_CSV_PATH = os.path.join(PROJECT_ROOT, "imdb_movie_metadata.csv")


# ----------------------------
# Helper Function: Write dictionary to JSON file
# ----------------------------
def write_json(data, filename):
    output_path = os.path.join(JSON_OUTPUT_DIR, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved JSON file: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {filename}: {e}")


# ----------------------------
# 1. Process Cluster Data (CSV -> JSON)
# Format: [ { userID: "1", clusters: [ { clusterID: "0", movies: [...] }, ... ] }, ... ]
# ----------------------------
def process_cluster_csv():
    print("🧠 Processing cluster data...")

    user_clusters = defaultdict(list)

    try:
        df = pd.read_csv(CLUSTER_CSV_PATH, low_memory=False)
        print("📌 Columns found:", df.columns.tolist())
        print("🔍 First 5 rows:\n", df.head(5).to_string(index=False))

        required_columns = {'user_id', 'cluster_id', 'movie_title', 'timestamp_readable', 'timestamp_raw'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

        for _, row in df.iterrows():
            try:
                user_id = str(int(row['user_id']))
                cluster_id = str(int(row['cluster_id']))
            except (ValueError, TypeError):
                continue

            if pd.isna(row['movie_title']):
                continue

            try:
                timestamp_raw = int(row['timestamp_raw'])
            except (ValueError, TypeError):
                timestamp_raw = 0

            movie_entry = {
                "movie_title": row['movie_title'],
                "timestamp_readable": row.get('timestamp_readable', ''),
                "timestamp_raw": timestamp_raw
            }

            movie_key = (
                movie_entry['movie_title'],
                movie_entry['timestamp_readable'],
                movie_entry['timestamp_raw']
            )

            # If user has no clusters yet or last clusterID doesn't match, create new cluster
            if not user_clusters[user_id] or user_clusters[user_id][-1]["clusterID"] != cluster_id:
                user_clusters[user_id].append({
                    "clusterID": cluster_id,
                    "movies": [],
                    "seen": set()  # Track duplicates within this cluster
                })

            current_cluster = user_clusters[user_id][-1]

            # Add movie only if not already seen
            if movie_key not in current_cluster["seen"]:
                current_cluster["movies"].append(movie_entry)
                current_cluster["seen"].add(movie_key)

        # Remove 'seen' keys from output
        result = []
        for user_id, clusters in user_clusters.items():
            cleaned_clusters = []
            for cluster in clusters:
                cleaned_clusters.append({
                    "clusterID": cluster["clusterID"],
                    "movies": cluster["movies"]
                })
            result.append({
                "userID": user_id,
                "clusters": cleaned_clusters
            })

        print("✅ Cluster data processed. Total users:", len(result))
        return result

    except Exception as e:
        print(f"[ERROR] Failed to process cluster CSV: {e}")
        import traceback
        traceback.print_exc()
        return []


# ----------------------------
# 2. Process Users DAT (DAT -> JSON)
# Format: {user_id: {"Gender": ..., "Age": ..., ...}}
# ----------------------------
def process_users_dat():
    print("🧠 Loading user demographic data...")
    try:
        users_df = pd.read_csv(
            USERS_DAT_PATH,
            sep='::',
            engine='python',
            names=['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode'],
            encoding='latin-1'
        )
        users_dict = {
            str(row['UserID']): {
                "Gender": row['Gender'],
                "Age": str(row['Age']),
                "Occupation": str(row['Occupation']),
                "Zip-code": str(row['ZipCode'])
            }
            for _, row in users_df.iterrows()
        }
        print("✅ User data loaded.")
        return users_dict
    except Exception as e:
        print(f"[ERROR] Failed to load users.dat: {e}")
        return {}


# ----------------------------
# 3. Process IMDb Metadata (CSV -> JSON)
# Format: {title: {"storyline": ..., "genres": [...], ...}}
# ----------------------------
def process_imdb_metadata():
    print("🧠 Loading IMDb metadata...")
    try:
        imdb_df = pd.read_csv(IMDB_CSV_PATH)
        movies_dict = {}

        for idx, row in imdb_df.iterrows():
            title = row.get('title')
            if not title:
                continue

            genres = row.get('genres', '').split('|') if isinstance(row.get('genres'), str) else []
            directors = row.get('directors', '').split('|') if isinstance(row.get('directors'), str) else []
            cast = row.get('cast', '').split('|') if isinstance(row.get('cast'), str) else []
            writers = row.get('writers', '').split('|') if isinstance(row.get('writers'), str) else []

            movie_id = str(idx)
            movies_dict[movie_id] = {
                "movie_id": movie_id,
                "title": title,
                "storyline": row.get("storyline", ""),
                "description": row.get("description", ""),
                "directors": directors,
                "cast": cast,
                "tagline": row.get("tagline", ""),
                "genres": genres,
                "writers": writers
            }

        print("✅ IMDb metadata loaded.")
        return movies_dict
    except Exception as e:
        print(f"[ERROR] Failed to load IMDb CSV: {e}")
        return {}


# ----------------------------
# Main Orchestrator
# ----------------------------
def main():
    """
    Main function orchestrating all data conversion steps.
    """

    print("[STEP 1/3] Converting user-cluster data...")
    clusters_data = process_cluster_csv()
    print("blah",clusters_data)
    write_json(clusters_data, "clusters.json")

    print("[STEP 2/3] Converting user metadata...")
    users_data = process_users_dat()
    write_json(users_data, "users.json")

    print("[STEP 3/3] Converting movie metadata...")
    movies_data = process_imdb_metadata()
    write_json(movies_data, "movies.json")

    print("🎉 All JSON files have been generated successfully!")


# ----------------------------
# Run the script
# ----------------------------
if __name__ == "__main__":
    main()
