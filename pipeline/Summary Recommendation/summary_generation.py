import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd


# Load JSON data
def load_data(json_path):
    print("[DEBUG] Loading JSON data...")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Pick up to `max_reviews` reviews per movie
def get_movie_reviews(reviews, max_reviews=5):
    return [rev['review_text'] for rev in reviews[:max_reviews] if 'review_text' in rev and rev['review_text']]


# Prompt for Positive Summary
def build_positive_prompt(movie_title, reviews):
    prompt = f"""Suppose you are skilled at summarizing user feedback on movies according to specific requirements. You are given a list of user reviews of the movie “{movie_title}”. These reviews may contain both positive and negative feedback. Your task is to carefully analyze all the reviews and generate a concise, coherent, and appealing positive summary that highlights the strengths and likable aspects of the movie. Focus only on the movie’s positive features such as acting, storytelling, direction, cinematography, music, emotional depth, or any other praised elements.

The summary must be written entirely in terms of the movie itself. Do not mention or refer to users, viewers, audiences, or reviewers in any way. All sentences should directly describe the movie’s qualities and strengths.

Ensure that this response is generated independently and does not rely on or get influenced by any previous summaries or responses. Treat this as a standalone task with no prior context. Limit the summary to a maximum of 100 words.

Reviews:
"""
    for i, r in enumerate(reviews, start=1):
        prompt += f"{i}. {r}\n"

    prompt += """
Positive Summary:
"""
    return prompt


# Prompt for Negative Summary
def build_negative_prompt(movie_title, reviews):
    prompt = f"""You are given a collection of user reviews of the movie “{movie_title}”. These reviews may include both positive and negative feedback. Your task is to carefully analyze the reviews and generate a concise, focused negative summary that highlights the weaknesses, criticisms, and disappointing aspects of the movie. Concentrate only on the negative features such as poor acting, weak plot, pacing issues, clichés, bad direction, low production quality, or any other concerns repeatedly mentioned in the reviews.

Focus purely on the movie itself—do not mention or refer to users, reviewers, critics, or audiences in any way. All sentences must directly describe the negative aspects of the movie.

Ensure that this response is generated independently and does not rely on or get influenced by any previous summaries or responses. Treat this as a standalone task with no prior context. Limit the summary to a maximum of 100 words.

Reviews:
"""
    for i, r in enumerate(reviews, start=1):
        prompt += f"{i}. {r}\n"

    prompt += """
Negative Summary:
"""
    return prompt


# Parse output into summary
def parse_output(output, keyword="Positive Summary:"):
    try:
        start_idx = output.find(keyword) + len(keyword)
        end_idx = None

        rest = output[start_idx:]
        if "\n\n" in rest:
            end_idx = start_idx + rest.find("\n\n")
        elif "Negative Summary:" in rest:
            end_idx = start_idx + rest.find("Negative Summary:")

        summary = output[start_idx:end_idx].strip()
        return summary
    except Exception as e:
        print("[ERROR] Failed to parse output:", str(e))
        return ""


# Generate summary using model
def summarize_with_llama(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Main pipeline function
def run_pipeline(
    json_path,
    output_file="movie_summaries.json",
    MAX_MOVIES=10,
    START_OFFSET=0,
    MAX_REVIEWS_PER_MOVIE=5,
    POS_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct",
    NEG_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
):
    data = load_data(json_path)

    imdb_ids = list(data.keys())[START_OFFSET:]
    total_movies = min(MAX_MOVIES, len(imdb_ids))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    ACCESS_TOKEN = "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"  # Replace with your own token

    print(f"[INFO] Loading POSITIVE model: {POS_MODEL_NAME}")
    positive_tokenizer = AutoTokenizer.from_pretrained(POS_MODEL_NAME, use_auth_token=ACCESS_TOKEN)
    positive_model = AutoModelForCausalLM.from_pretrained(
        POS_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=ACCESS_TOKEN
    )

    print(f"[INFO] Loading NEGATIVE model: {NEG_MODEL_NAME}")
    negative_tokenizer = AutoTokenizer.from_pretrained(NEG_MODEL_NAME, use_auth_token=ACCESS_TOKEN)
    negative_model = AutoModelForCausalLM.from_pretrained(
        NEG_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=ACCESS_TOKEN
    )

    results = []

    for idx, imdb_id in tqdm(enumerate(imdb_ids), desc="Processing Movies", total=total_movies):
        if idx >= total_movies:
            break

        reviews = data[imdb_id]
        selected_reviews = get_movie_reviews(reviews, max_reviews=MAX_REVIEWS_PER_MOVIE)

        if not selected_reviews:
            print(f"⚠️ No valid reviews found for movie {imdb_id}")
            continue

        movie_title = imdb_id
        if 'review_title' in reviews[0]:
            movie_title = reviews[0]['review_title'][:50]

        # Generate Positive Summary
        pos_prompt = build_positive_prompt(movie_title, selected_reviews)
        pos_response = summarize_with_llama(positive_model, positive_tokenizer, pos_prompt, device)
        positive_summary = parse_output(pos_response, keyword="Positive Summary:")

        # Generate Negative Summary
        neg_prompt = build_negative_prompt(movie_title, selected_reviews)
        neg_response = summarize_with_llama(negative_model, negative_tokenizer, neg_prompt, device)
        negative_summary = parse_output(neg_response, keyword="Negative Summary:")

        results.append({
            "imdb_id": imdb_id,
            "title": movie_title,
            "positive_summary": positive_summary,
            "negative_summary": negative_summary
        })

        del reviews, selected_reviews, pos_prompt, neg_prompt
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(output_file.replace(".json", ".csv"), index=False)

    print(f"✅ Summaries saved to {output_file} and CSV version")


if __name__ == "__main__":
    run_pipeline(
        json_path="/root/SummayGenAi/grouped_reviews.json",
        output_file="summaries.json",
        MAX_MOVIES=2,
        START_OFFSET=0,
        MAX_REVIEWS_PER_MOVIE=10,
        POS_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct",
        NEG_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    )
