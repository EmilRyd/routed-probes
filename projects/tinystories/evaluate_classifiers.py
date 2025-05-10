import torch
import torch.nn as nn
import pandas as pd
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

# Get the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class ActivationClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))

class ERAActivationClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))

def get_expanded_neuron_mask(model, layer_idx):
    """Get a boolean mask indicating which neurons are expanded in the MLP layer."""
    original_size = 2048
    mlp = model.blocks[layer_idx].mlp
    current_size = mlp.state_dict()['W_out'].shape[0]
    expanded_mask = torch.zeros(current_size, dtype=torch.bool)
    if current_size > original_size:
        expanded_mask[original_size:] = True
    return expanded_mask

def get_model_activations_batch(model, texts, layer_idx, expanded_mask=None, device=torch.device('cpu')):
    """Get activations from a specific layer for a batch of texts."""
    tokens = model.to_tokens(texts)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        
        if expanded_mask is not None:
            # For ERA model, get intermediate MLP activations (after first linear layer)
            activations = cache["post", layer_idx]  # Shape: [batch_size, seq_len, 2112]
            # Average across sequence length
            avg_activations = activations.mean(dim=1)  # [batch_size, d_mlp]
            # Apply mask to get expanded neurons
            expanded_mask = expanded_mask.to(avg_activations.device)
            avg_activations = avg_activations[:, expanded_mask]  # [batch_size, num_expanded]
        else:
            # For base model, get residual stream activations
            activations = cache["resid_post", layer_idx]  # [batch_size, seq_len, d_model]
            # Average across sequence length
            avg_activations = activations.mean(dim=1)  # [batch_size, d_model]
            
    return avg_activations

def evaluate_classifier(classifier, activations, labels, device):
    """Evaluate classifier and return metrics."""
    classifier.eval()
    with torch.no_grad():
        activations = activations.to(device)
        outputs = classifier(activations).squeeze()
        predictions = (outputs > 0.5).float().cpu().numpy()
        labels = np.array(labels)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_activations_for_dataset(model, stories, layer_idx, expanded_mask=None, batch_size=4, device=torch.device('cpu')):
    """Get activations for all stories in batches."""
    all_activations = []
    for i in tqdm(range(0, len(stories), batch_size)):
        batch_stories = stories[i:i + batch_size]
        batch_activations = get_model_activations_batch(
            model, batch_stories, layer_idx, expanded_mask, device
        )
        all_activations.append(batch_activations.cpu())
    return torch.cat(all_activations, dim=0)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(SCRIPT_DIR, "classified_forest_stories_1k_balanced.csv"))
    full_df = pd.read_csv(os.path.join(SCRIPT_DIR, "classified_forest_stories.csv"))
    ood_df = pd.read_csv(os.path.join(SCRIPT_DIR, "classified_forest_stories_ood_balanced.csv"))
    
    # Remove training data from full dataset to get unseen data
    train_stories = set(train_df['story'].tolist())
    unseen_df = full_df[~full_df['story'].isin(train_stories)]
    
    # Take 10% sample of unseen data
    unseen_df = unseen_df.sample(frac=0.1, random_state=42)
    print(f"Using {len(unseen_df)} unseen stories for evaluation")
    
    # Load base model for non-ERA classifier
    print("Loading base model...")
    base_model = HookedTransformer.from_pretrained(
        "roneneldan/TinyStories-28M",
        device=device
    )
    base_model.eval()
    
    # Load ERA model
    print("Loading ERA model...")
    era_model_path = "/workspace/routed-probes/models/models/full_runs/full_run_era_pre_ablation.pt"
    era_checkpoint = torch.load(era_model_path, map_location=device)
    
    # Create ERA model config
    era_cfg = {
        'd_model': 512,
        'n_layers': 8,
        'n_heads': 16,
        'd_head': 32,
        'd_mlp': 2112,
        'n_ctx': 2048,
        'tokenizer_name': 'gpt2',
        'd_vocab': 50257,
        'act_fn': 'gelu',
        'normalization_type': 'LN'
    }
    era_model = HookedTransformer(era_cfg)
    era_model.load_state_dict(era_checkpoint)
    era_model.to(device)
    era_model.eval()
    
    # Get middle layer index and expanded mask for ERA model
    middle_layer_idx = era_model.cfg.n_layers // 2
    print(f"\nUsing layer {middle_layer_idx} for both models")
    expanded_mask = get_expanded_neuron_mask(era_model, middle_layer_idx)
    
    # Load classifiers
    base_classifier = ActivationClassifier(512)  # d_model size for non-PCA classifier
    base_classifier.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "forest_classifier_not_pca.pt")))
    base_classifier.to(device)
    
    era_classifier = ERAActivationClassifier(64)  # number of expanded neurons
    era_classifier.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "forest_era_classifier.pt")))
    era_classifier.to(device)
    
    # Evaluate on unseen data
    print("\nEvaluating on unseen in-distribution data...")
    
    # Get activations for unseen data
    unseen_activations_base = get_activations_for_dataset(
        base_model, unseen_df['story'].tolist(), middle_layer_idx, 
        batch_size=4, device=device
    )
    unseen_activations_era = get_activations_for_dataset(
        era_model, unseen_df['story'].tolist(), middle_layer_idx, 
        expanded_mask=expanded_mask, batch_size=4, device=device
    )
    
    # Evaluate
    unseen_metrics_base = evaluate_classifier(
        base_classifier, unseen_activations_base, unseen_df['is_forest_story'].tolist(), device
    )
    unseen_metrics_era = evaluate_classifier(
        era_classifier, unseen_activations_era, unseen_df['is_forest_story'].tolist(), device
    )
    
    # Evaluate on OOD data
    print("\nEvaluating on out-of-distribution data...")
    
    # Get activations for OOD data
    ood_activations_base = get_activations_for_dataset(
        base_model, ood_df['story'].tolist(), middle_layer_idx,
        batch_size=4, device=device
    )
    ood_activations_era = get_activations_for_dataset(
        era_model, ood_df['story'].tolist(), middle_layer_idx,
        expanded_mask=expanded_mask, batch_size=4, device=device
    )
    
    # Evaluate
    ood_metrics_base = evaluate_classifier(
        base_classifier, ood_activations_base, ood_df['is_forest_story'].tolist(), device
    )
    ood_metrics_era = evaluate_classifier(
        era_classifier, ood_activations_era, ood_df['is_forest_story'].tolist(), device
    )
    
    # Print results
    print("\n=== Results ===")
    print("\nUnseen In-Distribution Data:")
    print("Base Classifier:")
    for metric, value in unseen_metrics_base.items():
        print(f"{metric}: {value:.4f}")
    print("\nERA Classifier:")
    for metric, value in unseen_metrics_era.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nOut-of-Distribution Data:")
    print("Base Classifier:")
    for metric, value in ood_metrics_base.items():
        print(f"{metric}: {value:.4f}")
    print("\nERA Classifier:")
    for metric, value in ood_metrics_era.items():
        print(f"{metric}: {value:.4f}")
    
    # Create bar plot
    labels = ['Unseen In-Distribution', 'Out-of-Distribution']
    base_accuracies = [unseen_metrics_base['accuracy'], ood_metrics_base['accuracy']]
    era_accuracies = [unseen_metrics_era['accuracy'], ood_metrics_era['accuracy']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_accuracies, width, label='Base Model', color='skyblue')
    rects2 = ax.bar(x + width/2, era_accuracies, width, label='ERA Model', color='lightgreen')
    
    # Add labels and title
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'accuracy_comparison.png'))
    print("\nPlot saved as accuracy_comparison.png")

if __name__ == "__main__":
    main() 