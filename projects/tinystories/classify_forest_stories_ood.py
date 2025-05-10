import datasets
import os
import random
import pandas as pd
from collections import defaultdict

def load_forest_words(file_path):
    """Loads forest-related words from a file."""
    try:
        with open(file_path, "r") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return set(words)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return set()

def contains_common_forest_words(example, common_forest_words):
    """Check if story contains any common forest words."""
    story_text = example.get("story", "").lower()
    return any(word in story_text for word in common_forest_words)

def label_story(example, forest_words_set):
    """Labels a story as forest-related (1) or not (0)."""
    story_text = example.get("story", "")
    if not story_text:
        example["is_forest_story"] = 0
        return example

    # Check for presence of any forest word
    story_lower = story_text.lower()
    # Count how many unique forest words appear in the story
    forest_word_count = sum(1 for word in forest_words_set if word in story_lower)
    
    # Label as forest story if it contains at least 2 forest-related words
    # This makes the OOD detection more stringent
    example["is_forest_story"] = 1 if forest_word_count >= 2 else 0
    example["forest_word_count"] = forest_word_count
    return example

def balance_dataset(dataset):
    """Creates a balanced dataset with equal numbers of forest and non-forest stories."""
    forest_stories = []
    non_forest_stories = []
    
    # Separate stories into forest and non-forest
    for example in dataset:
        if example["is_forest_story"] == 1:
            forest_stories.append(example)
        else:
            non_forest_stories.append(example)
    
    # Find the smaller class size
    min_size = min(len(forest_stories), len(non_forest_stories))
    
    # Randomly sample from the larger class to match the smaller class size
    if len(forest_stories) > min_size:
        forest_stories = random.sample(forest_stories, min_size)
    if len(non_forest_stories) > min_size:
        non_forest_stories = random.sample(non_forest_stories, min_size)
    
    # Combine and shuffle
    balanced_stories = forest_stories + non_forest_stories
    random.shuffle(balanced_stories)
    
    return balanced_stories

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Determine the absolute path to forest words files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forest_words_ood_file = os.path.join(script_dir, "forest_words_ood.txt")
    common_forest_words_file = os.path.join(script_dir, "forest_words.txt")

    # Load both sets of forest words
    forest_words_ood = load_forest_words(forest_words_ood_file)
    common_forest_words = load_forest_words(common_forest_words_file)

    if not forest_words_ood or not common_forest_words:
        print("Failed to load forest words. Exiting.")
        return

    print(f"Loaded {len(forest_words_ood)} OOD forest words: {forest_words_ood}")
    print(f"Loaded {len(common_forest_words)} common forest words: {common_forest_words}")

    # Load the dataset
    print("Loading dataset 'delphi-suite/stories'...")
    try:
        # Using a larger subset for OOD detection
        dataset = datasets.load_dataset("delphi-suite/stories", split="train[:10%]")
        print(f"Dataset loaded. Initial number of stories: {len(dataset)}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # First, filter out stories containing common forest words
    print("Filtering out stories with common forest words...")
    filtered_dataset = [
        example for example in dataset 
        if not contains_common_forest_words(example, common_forest_words)
    ]
    print(f"Remaining stories after filtering: {len(filtered_dataset)}")

    # Label the filtered dataset using OOD forest words
    print("Labeling remaining stories with OOD forest words...")
    labeled_dataset = [label_story(example, forest_words_ood) for example in filtered_dataset]
    
    # Balance the dataset
    print("Creating balanced dataset...")
    balanced_stories = balance_dataset(labeled_dataset)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(balanced_stories)
    
    # Keep only the required columns in the same order as the original CSV
    df = df[["story", "is_forest_story"]]
    
    # Save to CSV in the same directory
    output_file = os.path.join(script_dir, "classified_forest_stories_ood_balanced.csv")
    df.to_csv(output_file, index=False)
    print(f"\nBalanced OOD dataset saved to: {output_file}")
    
    # Print statistics
    total_stories = len(df)
    forest_stories = df["is_forest_story"].sum()
    print(f"\n--- Balanced Dataset Stats ---")
    print(f"Total stories: {total_stories}")
    print(f"Forest-related stories (OOD): {forest_stories}")
    print(f"Non-forest-related stories: {total_stories - forest_stories}")

if __name__ == "__main__":
    main() 