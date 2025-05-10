import datasets
import os

def load_forest_words(file_path="forest_words.txt"):
    """Loads forest-related words from a file."""
    try:
        with open(file_path, "r") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return set(words)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return set()

def label_story(example, forest_words_set):
    """Labels a story as forest-related (1) or not (0)."""
    story_text = example.get("story", "")
    if not story_text:
        example["is_forest_story"] = 0
        return example

    # Simple check for presence of any forest word
    # More sophisticated matching (e.g., word boundaries) could be added
    if any(word in story_text.lower() for word in forest_words_set):
        example["is_forest_story"] = 1
    else:
        example["is_forest_story"] = 0
    return example

def main():
    # Determine the absolute path to forest_words.txt
    # Assumes forest_words.txt is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forest_words_file = os.path.join(script_dir, "forest_words.txt")

    forest_words = load_forest_words(forest_words_file)

    if not forest_words:
        print("No forest words loaded. Exiting.")
        return

    print(f"Loaded {len(forest_words)} forest-related words: {forest_words}")

    # Load the dataset
    print("Loading dataset 'delphi-suite/stories'...")
    try:
        # Using a small subset for faster processing during development/testing.
        # Remove/adjust 'train[:1%]' for the full dataset.
        dataset = datasets.load_dataset("delphi-suite/stories", split="train[:1%]")
        # To use the full dataset:
        # dataset = datasets.load_dataset("delphi-suite/stories", split="train")
        print(f"Dataset loaded. Number of stories: {len(dataset)}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Label the dataset
    print("Labeling stories...")
    # Use `partial` or a lambda if `label_story` needed more fixed arguments with `map`
    # For Hugging Face datasets, `map` passes the examples (dict) directly.
    # We pass `batched=False` as our function processes one example at a time.
    # If `forest_words` were very large and `label_story` complex, batching could be more efficient.
    dataset_with_labels = dataset.map(
        lambda example: label_story(example, forest_words_set=forest_words)
    )
    print("Labeling complete.")

    # Analyze the labels
    forest_story_count = sum(dataset_with_labels["is_forest_story"])
    non_forest_story_count = len(dataset_with_labels) - forest_story_count

    print(f"\\n--- Labeling Stats ---")
    print(f"Total stories processed: {len(dataset_with_labels)}")
    print(f"Number of forest-related stories: {forest_story_count}")
    print(f"Number of non-forest-related stories: {non_forest_story_count}")

    # You can now work with `dataset_with_labels`.
    # For example, to see a few examples:
    # print("\\n--- Example Labeled Stories ---")
    # for i in range(min(5, len(dataset_with_labels))):
    #     print(f"Story: {dataset_with_labels[i]['story'][:100]}...") # Print first 100 chars
    #     print(f"Is Forest Story: {dataset_with_labels[i]['is_forest_story']}\\n")

    # If you want to save the dataset:
    #print("\\nSaving labeled dataset...")
    #dataset_with_labels.save_to_disk("path_to_save_labeled_dataset_directory")
    #print("Dataset saved.")
    # Or to save as CSV:
    dataset_with_labels.to_csv("classified_forest_stories.csv")
    print("Dataset saved to CSV.")


if __name__ == "__main__":
    main() 