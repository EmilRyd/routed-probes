import pandas as pd
import numpy as np

# Read the original dataset
df = pd.read_csv('classified_forest_stories.csv')

# Separate forest and non-forest stories
forest_stories = df[df['is_forest_story'] == 1]
non_forest_stories = df[df['is_forest_story'] == 0]

# Sample 500 from each class
n_per_class = 500
forest_sample = forest_stories.sample(n=n_per_class, random_state=42)
non_forest_sample = non_forest_stories.sample(n=n_per_class, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([forest_sample, non_forest_sample])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV
balanced_df.to_csv('classified_forest_stories_1k_balanced.csv', index=False)

print(f"Created balanced dataset with {len(balanced_df)} stories")
print("Class distribution:")
print(balanced_df['is_forest_story'].value_counts()) 