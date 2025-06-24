import json
import normalising_cluster

# Load the cluster data
with open('/Volumes/Sanjeev HD/M.TECH  IIT-P/Sem - 3 research/SonyResearchMovieRecommendation/pipeline/output/json/clusters.json', 'r', encoding='utf-8') as f:
    cluster_data = json.load(f)

# Load the movie master data
with open('pipeline/output/json/movie_master_data.json', 'r', encoding='utf-8') as f:
    movie_master = json.load(f)

# Build a dictionary to map clean_title -> movie_id
title_to_id = {
    movie["clean_title"]: movie["movie_id"]
    for movie in movie_master.values()
}

# Also build a list of all movies for fallback matching (substring on title)
fallback_movies = [
    {"title": movie["title"], "movie_id": movie["movie_id"]}
    for movie in movie_master.values()
]

# Replace each movie_title in clusters with just the movie_id string
for user in cluster_data:
    for cluster in user["clusters"]:
        new_movies = []
        for movie_entry in cluster["movies"]:
            movie_title = movie_entry["movie_title"]

            # Try exact match via clean_title first
            matched_id = title_to_id.get(movie_title)

            if not matched_id:
                # Fallback: substring match on full title
                for m in fallback_movies:
                    if movie_title.lower() in m["title"].lower():
                        matched_id = m["movie_id"]
                        break

            if matched_id:
                new_movies.append(matched_id)
            else:
                print(f"⚠️ No match found for '{movie_title}'")

        cluster["movies"] = new_movies  # Replace the list of objects with list of strings

# Save updated cluster data to a new file
output_path = 'pipeline/output/json/cluster_json_data_with_id.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(cluster_data, f, indent=2)

print(f"✅ Successfully cleaned movie lists. Output saved to {output_path}")


#normalising cluster with missing cluster ID

cluster_json_path = 'pipeline/output/json/cluster_json_data_with_id.json'
ratings_dat_path = 'dataset/ratings.dat'
output_json_path = 'pipeline/output/json/master_cluster_data.json'

print("🔄 Loading data and normalizing clusters...")
ratings = normalising_cluster.load_ratings(ratings_dat_path)
clusters = normalising_cluster.load_clusters(cluster_json_path)

updated_clusters = normalising_cluster.add_missing_movies_to_new_cluster(clusters, ratings)

print(f"\n💾 Saving updated clusters to {output_json_path}")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(updated_clusters, f, indent=2)

print("✅ Clusters successfully updated from main script!")