import json

def load_ratings(ratings_file):
    """Load ratings grouped by userID"""
    user_ratings = {}
    with open(ratings_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("::")
            user_id = parts[0]
            movie_id = parts[1]
            if user_id not in user_ratings:
                user_ratings[user_id] = set()
            user_ratings[user_id].add(movie_id)
    return user_ratings

def load_clusters(cluster_json_file):
    """Load the cluster JSON data"""
    with open(cluster_json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_clusters_against_ratings(clusters, user_ratings, output_file="/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/pipeline/test/output.txt"):
    full_output = []
    total_users = 0
    perfect_users = 0
    total_missing_movies = 0

    for user_data in clusters:
        total_users += 1
        user_id = user_data["userID"]

        if user_id not in user_ratings:
            msg = f"❌ UserID '{user_id}' not found in ratings."
            print(msg)
            full_output.append(msg)
            continue

        rated_movies = user_ratings[user_id]

        clustered_movies = set()
        for cluster in user_data.get("clusters", []):
            for entry in cluster.get("movies", []):
                if isinstance(entry, dict):
                    movie_id = entry.get("movie_ID") or entry.get("MovieID")
                else:
                    movie_id = entry  # assuming it's just the ID as string
                if movie_id:
                    clustered_movies.add(movie_id)

        missing_movies = rated_movies - clustered_movies

        if missing_movies:
            total_missing_movies += len(missing_movies)
            msg_list = [f"\n❌ UserID {user_id} has rated movies not found in any cluster:"]
            msg_list.append(f"   Missing Movie IDs: {', '.join(sorted(missing_movies))}")
            full_output.extend(msg_list)
            print("\n".join(msg_list))
        else:
            perfect_users += 1

    # Summary
    full_output.append("\n📊 Validation Summary\n" + "-" * 40)
    full_output.append(f"Total users processed: {total_users}")
    full_output.append(f"Users with 100% match: {perfect_users}/{total_users}")
    full_output.append(f"Total missing rated movies: {total_missing_movies}")

    accuracy = 0
    if total_users > 0:
        accuracy = (perfect_users / total_users) * 100
    full_output.append(f"Accuracy (% of users fully matched): {accuracy:.2f}%")
    full_output.append("-" * 40)

    if accuracy == 100:
        full_output.append("✅ All users have their rated movies correctly clustered.")
    else:
        full_output.append("❌ Some users are missing rated movies in their clusters.")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(full_output))

    return accuracy, perfect_users, total_users

# ———————————————— #
#       CONFIGURATION
# ———————————————— #

CLUSTER_JSON_PATH = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation copy/pipeline/Data Prepration pipelines/output/json/master_cluster_data.json'
RATINGS_DAT_PATH = '/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation copy/dataset/ratings.dat'

# ———————————————— #
#       RUN TEST
# ———————————————— #

print("🔍 Validating that all rated movies appear in at least one cluster...\n")
clusters = load_clusters(CLUSTER_JSON_PATH)
ratings = load_ratings(RATINGS_DAT_PATH)
accuracy, perfect_users, total_users = validate_clusters_against_ratings(clusters, ratings)

print(f"\n📊 Final Accuracy: {accuracy:.2f}%")
print(f"💯 Users with 100% match: {perfect_users}/{total_users}")