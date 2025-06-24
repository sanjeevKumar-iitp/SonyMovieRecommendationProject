# ----------------------------
# full_user_clustering_pipeline_gpu_batch.py
# Batch User-Based Clustering with Hybrid Features + KMeans + Elbow Method + Cluster Labeling
# Includes average count per cluster
# Adds GPU support for SentenceTransformer
# Batch processing all users if MAX_USERS_TO_PROCESS=None
# ----------------------------

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import os
import csv
import warnings
from tqdm import tqdm
import torch

warnings.filterwarnings("ignore")

# Set working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "..", "dataset")
IMDB_CSV = os.path.join(PROJECT_ROOT, "imdb_movie_metadata.csv")

# Output directories and files
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV file paths
USER_KMEANS_REPORT_CSV = os.path.join(OUTPUT_DIR, "user_kmeans_report.csv")  # Summary stats
USER_MOVIE_CLUSTERS_CSV = os.path.join(OUTPUT_DIR, "user_movie_clusters.csv")  # Movie-cluster mapping
CLUSTER_INTERPRETATIONS_CSV = os.path.join(OUTPUT_DIR, "user_cluster_interpretations.csv")  # Interpretation
CLUSTER_LABELS_CSV = os.path.join(OUTPUT_DIR, "user_cluster_labels.csv")  # Auto-generated labels
USER_SORTED_CLUSTERS_CSV = os.path.join(OUTPUT_DIR, "user_sorted_clusters.csv")  # Sorted output

sorted_cluster_fields = ["user_id", "cluster_id", "movie_title", "timestamp_readable", "timestamp_raw"]

# CONFIGURATION
MAX_USERS_TO_PROCESS = None  # Set to None to process ALL users in batches
BATCH_SIZE = 50  # Number of users per batch if MAX_USERS_TO_PROCESS is None

# Check if GPU is available for PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# Initialize SentenceTransformer model globally once, and move to device
_model = SentenceTransformer('all-MiniLM-L6-v2')
_model.to(DEVICE)


# ----------------------------
# Load MovieLens Data
# ----------------------------
def load_movielens_data():
    print("[INFO] Loading MovieLens data...")
    movies = pd.read_csv(os.path.join(DATASET_DIR, "movies.dat"),
                         sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'],
                         encoding='latin-1')
    ratings = pd.read_csv(os.path.join(DATASET_DIR, "ratings.dat"),
                          sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    users = pd.read_csv(os.path.join(DATASET_DIR, "users.dat"),
                        sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode'],
                        encoding='latin-1')
    return movies, ratings, users


# ----------------------------
# Load IMDb Metadata
# ----------------------------
def load_imdb_metadata():
    df = pd.read_csv(IMDB_CSV)
    print(f"[SUCCESS] Loaded IMDb metadata for {len(df)} movies.")
    return df


# ----------------------------
# Preprocessing Helper
# ----------------------------
def preprocess_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return "unknown"
    return text.strip().lower()


# ----------------------------
# Feature Encoders
# ----------------------------

def encode_genres_tags(data_series):
    sentences = []
    for x in data_series:
        if isinstance(x, str):
            tags = [t.strip() for t in x.lower().split('|') if t.strip()]
            sentences.append(tags)
        else:
            sentences.append([])
    # Train Word2Vec model on tags
    model = Word2Vec(sentences=[s for s in sentences if s], vector_size=100,
                     window=5, min_count=1, workers=4)
    vectors = []
    for tags in sentences:
        if tags:
            vecs = [model.wv[tag] for tag in tags if tag in model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(100))
        else:
            vectors.append(np.zeros(100))
    return np.array(vectors)


def encode_sentences(sentence_list):
    """Encode a list of sentences using SentenceTransformer with batching and GPU."""
    cleaned = [str(s) if isinstance(s, str) and s.strip() != '' else "unknown" for s in sentence_list]
    # SentenceTransformer handles batching internally
    embeddings = _model.encode(cleaned, batch_size=32, show_progress_bar=False, device=DEVICE)
    return embeddings


def tfidf_features(text_list):
    """TF-IDF features for directors/cast"""
    cleaned = [str(t) if isinstance(t, str) and t.strip() != '' else "unknown" for t in text_list]
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(cleaned).toarray()


# ----------------------------
# Find Optimal K Using Elbow Method
# ----------------------------
def find_optimal_k_elbow(X, max_k=10):
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("No samples provided for clustering.")
    max_k = min(max_k, n_samples)
    if max_k < 2:
        max_k = 1
    k_range = range(1, max_k + 1)
    inertias = []

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        except Exception:
            inertias.append(float('inf'))
    if len(inertias) == 0:
        return {"optimal_k": 1, "inertias": []}
    try:
        deltas = np.diff(inertias)
        acc = np.diff(deltas)
        optimal_k = k_range[np.argmin(acc) + 1]
    except ValueError:
        optimal_k = 1
    return {
        "optimal_k": optimal_k,
        "inertias": inertias
    }


# ----------------------------
# Convert Unix Timestamp to Readable Format
# ----------------------------
def convert_unix_to_readable(unix_time):
    return pd.to_datetime(unix_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')


# ----------------------------
# Analyze One User
# ----------------------------
def analyze_user(user_id, movies_df, ratings_df, users_df, imdb_df):
    try:
        print(f"\n[DEBUG] Processing User ID: {user_id}")
        watched = ratings_df[ratings_df['UserID'] == user_id]
        merged = pd.merge(watched, movies_df, on='MovieID', how='inner')

        # Extract clean titles (to match with IMDb)
        merged['CleanTitle'] = merged['Title'].str.extract(r'^(.*?)(?: $|\()')[0].str.strip()
        matched = pd.merge(merged[['CleanTitle', 'Timestamp']], imdb_df,
                           left_on='CleanTitle', right_on='title', how='inner')

        if len(matched) < 1:
            print(f"[WARNING] Skipping user {user_id}: No IMDb matches found.")
            return None, None, None, None, []

        # Preprocess required text fields
        cols_to_preprocess = ['genres', 'storyline', 'description', 'directors', 'cast']
        for col in cols_to_preprocess:
            matched[col] = matched[col].apply(preprocess_text)

        # Feature extraction
        genre_vectors = encode_genres_tags(matched['genres'])
        story_vectors = encode_sentences(matched['storyline'])
        desc_vectors = encode_sentences(matched['description'])
        director_tfidf = tfidf_features(matched['directors'])
        cast_tfidf = tfidf_features(matched['cast'])

        # Combine all features horizontally
        combined_features = np.hstack([
            genre_vectors,
            story_vectors,
            desc_vectors,
            director_tfidf,
            cast_tfidf
        ])

        # Find optimal Ks for different feature sets
        result_a = find_optimal_k_elbow(combined_features)
        result_b = find_optimal_k_elbow(genre_vectors)
        no_genre_features = np.hstack([story_vectors, desc_vectors, director_tfidf, cast_tfidf])
        result_c = find_optimal_k_elbow(no_genre_features)

        k_final = result_a["optimal_k"]
        if k_final > len(matched):
            k_final = len(matched)

        kmeans = KMeans(n_clusters=k_final, random_state=42)
        labels = kmeans.fit_predict(combined_features)

        titles = matched['title'].tolist()
        movie_cluster_list = []
        for idx, cluster_id in enumerate(labels):
            movie_cluster_list.append({
                "user_id": user_id,
                "movie_title": titles[idx],
                "cluster_id": int(cluster_id)
            })

        interpretations = interpret_clusters(matched, labels, k_final, user_id)
        labels_info = label_clusters(interpretations)

        user_info = users_df[users_df['UserID'] == user_id].iloc[0]
        watched_count = len(matched)
        user_report = {
            "user_id": user_id,
            "gender": user_info['Gender'],
            "age": user_info['Age'],
            "occupation": user_info['Occupation'],
            "watched_count": watched_count,
            "a_optimal_k": result_a["optimal_k"],
            "b_optimal_k": result_b["optimal_k"],
            "c_optimal_k": result_c["optimal_k"],
            "avg_count_all": round(watched_count / result_a["optimal_k"], 2),
            "avg_count_genre_only": round(watched_count / result_b["optimal_k"], 2),
            "avg_count_no_genre": round(watched_count / result_c["optimal_k"], 2)
        }

        # Prepare list of movies with clusters and timestamps
        movies_with_clusters = []
        for idx, cluster_id in enumerate(labels):
            row = matched.iloc[idx]
            movies_with_clusters.append({
                "user_id": user_id,
                "movie_title": row['title'],
                "cluster_id": int(cluster_id),
                "timestamp_raw": row['Timestamp'],
                "timestamp_readable": convert_unix_to_readable(row['Timestamp'])
            })

        df_movies = pd.DataFrame(movies_with_clusters)

        # ---- FIXED SORTING ISSUE ----
        cluster_first_times = (
            df_movies.groupby(['user_id', 'cluster_id'])['timestamp_raw']
            .min().reset_index()
            .rename(columns={'timestamp_raw': 'first_cluster_time'})
        )
        df_movies = df_movies.merge(cluster_first_times, on=['user_id', 'cluster_id'], how='left')

        df_sorted = df_movies.sort_values(
            by=['first_cluster_time', 'cluster_id', 'timestamp_raw'],
            ascending=[True, True, True]
        )

        df_sorted = df_sorted.drop(columns=['first_cluster_time'])
        final_sorted_clusters = df_sorted[sorted_cluster_fields].to_dict(orient='records')
        # ---- END FIX ----

        return user_report, movie_cluster_list, interpretations, labels_info, final_sorted_clusters

    except Exception as e:
        print(f"[ERROR] Failed to process user {user_id}: {e}")
        return None, None, None, None, []


# ----------------------------
# Interpret Clusters
# ----------------------------
def interpret_clusters(df, labels, k, user_id):
    interpretations = []
    df = df.copy()
    df['cluster'] = labels
    for cluster_id in range(k):
        cluster_movies = df[df['cluster'] == cluster_id]
        titles = cluster_movies['title'].tolist()
        genres = "|".join(cluster_movies['genres']).split("|")
        directors = "|".join(cluster_movies['directors']).split("|")
        cast = "|".join(cluster_movies['cast']).split("|")
        interpretations.append({
            "user_id": user_id,
            "cluster_id": cluster_id,
            "sample_movies": "; ".join(titles[:3]),
            "dominant_genres": ", ".join(pd.Series(genres).value_counts().head(2).index.tolist()),
            "dominant_directors": ", ".join(pd.Series(directors).value_counts().head(2).index.tolist()),
            "dominant_cast": ", ".join(pd.Series(cast).value_counts().head(2).index.tolist())
        })
    return interpretations


# ----------------------------
# Label Clusters
# ----------------------------
def label_clusters(interpretations):
    labels = []
    for item in interpretations:
        genre_label = item['dominant_genres'].strip()
        director_label = item['dominant_directors'].strip()
        cast_label = item['dominant_cast'].strip()
        label_parts = []
        if genre_label:
            label_parts.append(genre_label.title())
        if director_label and director_label not in genre_label:
            label_parts.append(f"{director_label.title()} Films")
        elif cast_label and cast_label not in genre_label:
            label_parts.append(f"{cast_label.split(',')[0]} Movies")
        final_label = " / ".join(label_parts[:2]) if label_parts else "Unknown Theme"
        labels.append({
            "user_id": item['user_id'],
            "cluster_id": item['cluster_id'],
            "label": final_label
        })
    return labels


# ----------------------------
# Main Batch Runner with batching if needed
# ----------------------------
def run_full_analysis():
    print("🚀 Starting full batch KMeans analysis...")
    movies_df, ratings_df, users_df = load_movielens_data()
    imdb_df = load_imdb_metadata()

    all_users = users_df['UserID'].unique()
    if MAX_USERS_TO_PROCESS is None:
        print(f"[INFO] Running for ALL {len(all_users)} users in batches of size {BATCH_SIZE}")
    else:
        print(f"[INFO] Running for first {MAX_USERS_TO_PROCESS} users")
        all_users = all_users[:MAX_USERS_TO_PROCESS]

    report_fields = ["user_id", "gender", "age", "occupation", "watched_count",
                     "a_optimal_k", "b_optimal_k", "c_optimal_k",
                     "avg_count_all", "avg_count_genre_only", "avg_count_no_genre"]
    cluster_fields = ["user_id", "movie_title", "cluster_id"]
    interpretation_fields = ["user_id", "cluster_id", "sample_movies",
                             "dominant_genres", "dominant_directors", "dominant_cast"]
    label_fields = ["user_id", "cluster_id", "label"]

    def init_csv(file_path, fieldnames):
        if not os.path.isfile(file_path):
            with open(file_path, mode="w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    init_csv(USER_KMEANS_REPORT_CSV, report_fields)
    init_csv(USER_MOVIE_CLUSTERS_CSV, cluster_fields)
    init_csv(CLUSTER_INTERPRETATIONS_CSV, interpretation_fields)
    init_csv(CLUSTER_LABELS_CSV, label_fields)
    init_csv(USER_SORTED_CLUSTERS_CSV, sorted_cluster_fields)

    # Process users in batches to avoid memory overhead
    total_users = len(all_users)
    for start_idx in range(0, total_users, BATCH_SIZE):
        batch_users = all_users[start_idx:start_idx + BATCH_SIZE]
        print(f"[INFO] Processing users batch: {start_idx} to {start_idx + len(batch_users) - 1}")
        for user_id in tqdm(batch_users, desc=f"Processing batch {start_idx//BATCH_SIZE + 1}"):
            user_report, movie_clusters, interpretations, labels_info, final_sorted_clusters = analyze_user(
                user_id, movies_df, ratings_df, users_df, imdb_df
            )

            if user_report:
                with open(USER_KMEANS_REPORT_CSV, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=report_fields)
                    writer.writerow(user_report)

            if movie_clusters:
                with open(USER_MOVIE_CLUSTERS_CSV, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=cluster_fields)
                    writer.writerows(movie_clusters)

            if interpretations:
                with open(CLUSTER_INTERPRETATIONS_CSV, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=interpretation_fields)
                    writer.writerows(interpretations)

            if labels_info:
                with open(CLUSTER_LABELS_CSV, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=label_fields)
                    writer.writerows(labels_info)

            if final_sorted_clusters:
                with open(USER_SORTED_CLUSTERS_CSV, "a", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=sorted_cluster_fields)
                    writer.writerows(final_sorted_clusters)

    print(f"📂 All results saved to: {OUTPUT_DIR}")


# ----------------------------
# Run It!
# ----------------------------
if __name__ == "__main__":
    run_full_analysis()
