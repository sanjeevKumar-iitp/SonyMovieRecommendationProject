import json
from collections import defaultdict

def load_ratings(ratings_file):
    user_ratings = defaultdict(dict)
    with open(ratings_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("::")
            user_id = parts[0]
            movie_id = parts[1]
            timestamp = int(parts[3])
            user_ratings[user_id][movie_id] = timestamp
    return user_ratings

def load_clusters(cluster_json_file):
    with open(cluster_json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def add_missing_movies_to_new_cluster(clusters_data, user_ratings):
    updated_clusters = []

    for user_data in clusters_data:
        user_id = user_data["userID"]

        if user_id not in user_ratings:
            print(f"❌ UserID '{user_id}' not found in ratings.")
            updated_clusters.append(user_data)
            continue

        rated_movies = user_ratings[user_id]

        existing_clusters = []
        for cluster in user_data.get("clusters", []):
            cleaned_movies = []

            for entry in cluster.get("movies", []):
                if isinstance(entry, dict):
                    movie_id = entry.get("movie_ID")
                else:
                    movie_id = entry

                if movie_id and movie_id in rated_movies:
                    cleaned_movies.append(movie_id)
                else:
                    print(f"🗑️ Removed un-rated movie {movie_id} from cluster {cluster['clusterID']}")

            # Only keep cluster if it still has valid movies
            if cleaned_movies:
                updated_cluster = {
                    "clusterID": cluster["clusterID"],
                    "movies": cleaned_movies
                }
                existing_clusters.append(updated_cluster)
            else:
                print(f"🧹 Cluster {cluster['clusterID']} became empty after cleaning — removed.")

        # Now find missing movies (already rated, not in any cluster)
        clustered_movies = set()
        for c in existing_clusters:
            clustered_movies.update(c["movies"])

        missing_movies = [mid for mid in rated_movies if mid not in clustered_movies]

        # Add missing movies sorted by timestamp into a new cluster
        if missing_movies:
            print(f"📎 Adding {len(missing_movies)} missing movies to User {user_id}")

            missing_with_time = [(mid, rated_movies[mid]) for mid in missing_movies]
            missing_with_time.sort(key=lambda x: x[1])
            sorted_missing = [mid for mid, _ in missing_with_time]

            existing_ids = [int(c["clusterID"]) for c in existing_clusters]
            next_id = max(existing_ids) + 1 if existing_ids else 0

            new_cluster = {
                "clusterID": str(next_id),
                "movies": sorted_missing
            }

            existing_clusters.append(new_cluster)

        # Final user data
        updated_user_data = {
            "userID": user_data["userID"],
            "clusters": existing_clusters
        }

        updated_clusters.append(updated_user_data)

    return updated_clusters


# ———————————————— #
#       RUN SCRIPT (Only when executed directly)
# ———————————————— #

if __name__ == "__main__":
    CLUSTER_JSON_PATH = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation copy/pipeline/Data Prepration pipelines/output/json/cluster_json_data_with_id.json'
    RATINGS_DAT_PATH = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation copy/dataset/ratings.dat'
    OUTPUT_JSON_PATH = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation copy/pipeline/Data Prepration pipelines/output/json/master_cluster_data.json'

    print("🔍 Loading ratings and clusters...")
    ratings = load_ratings(RATINGS_DAT_PATH)
    clusters = load_clusters(CLUSTER_JSON_PATH)

    print("\n🛠️ Cleaning clusters and adding missing rated movies...")
    updated_clusters = add_missing_movies_to_new_cluster(clusters, ratings)

    print(f"\n💾 Saving cleaned & updated clusters to {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_clusters, f, indent=2)

    print("\n✅ Done! Clusters are now cleaned and complete.")