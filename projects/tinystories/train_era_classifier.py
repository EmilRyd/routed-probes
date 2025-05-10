import torch
import torch.nn as nn
import torch.backends.mps
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm
import wandb
import os

class PrecomputedActivationsDataset(Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

class ERAActivationClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))

def get_expanded_neuron_mask(model, layer_idx):
    """Get a boolean mask indicating which neurons are expanded in the MLP layer."""
    # In ERA models, the MLP layer is expanded in the middle dimension
    # Original size is typically 2048, expanded size is 2112
    original_size = 2048  # Standard size for TinyStories models
    
    # Get the MLP layer
    mlp = model.blocks[layer_idx].mlp
    
    # Print MLP dimensions
    print("\nMLP Layer Info:")
    print(f"Layer index: {layer_idx}")
    print(f"MLP state dict keys: {[k for k in mlp.state_dict().keys()]}")
    for name, param in mlp.state_dict().items():
        print(f"{name} shape: {param.shape}")
    
    # Get current size from the MLP weights - use the expanded middle dimension
    current_size = mlp.state_dict()['W_out'].shape[0]  # This is the expanded dimension (2112)
    print(f"Current MLP size: {current_size}")
    print(f"Original size: {original_size}")
    
    # Create a mask where True indicates expanded neurons (the last (current_size - original_size) neurons)
    expanded_mask = torch.zeros(current_size, dtype=torch.bool)
    if current_size > original_size:
        expanded_mask[original_size:] = True
    
    return expanded_mask

def get_model_activations(model, text, layer_idx, expanded_mask=None):
    """Get activations from a specific layer for the input text, optionally filtering for expanded neurons."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        # Get MLP activations from the specified layer
        activations = cache["mlp_out", layer_idx][0]  # Shape: [seq_len, d_mlp]
        
        # If expanded_mask is provided, only keep expanded neurons
        if expanded_mask is not None:
            activations = activations[:, expanded_mask]
            
        # Average across sequence length
        avg_activations = activations.mean(dim=0)  # Shape: [d_mlp] or [num_expanded]
    return avg_activations

def get_model_activations_batch(model, texts, layer_idx, expanded_mask=None):
    """Get activations from a specific layer for a batch of texts."""
    tokens = model.to_tokens(texts)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        # Get intermediate MLP activations (after first linear layer)
        activations = cache["post", layer_idx]  # Shape: [batch_size, seq_len, 2112]
        
        # Print activation shapes for debugging
        print(f"\nActivation shapes in get_model_activations_batch:")
        print(f"Initial activations shape: {activations.shape}")
        
        # If expanded_mask is provided, only keep expanded neurons
        if expanded_mask is not None:
            # Make sure expanded_mask is on the right device
            expanded_mask = expanded_mask.to(activations.device)
            # Only keep the expanded neurons
            activations = activations[:, :, expanded_mask]
            print(f"After masking shape: {activations.shape}")
            
        # Average across sequence length for each item in batch
        avg_activations = activations.mean(dim=1)  # Shape: [batch_size, num_expanded]
        print(f"Final shape: {avg_activations.shape}")
            
    return avg_activations

def precompute_activations(model, stories, layer_idx, expanded_mask=None, batch_size=4, device=torch.device('cpu')):
    """Precompute activations for all stories."""
    all_activations = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(stories), batch_size), desc="Precomputing activations"):
        batch_stories = stories[i:i + batch_size]
        batch_activations = get_model_activations_batch(model, batch_stories, layer_idx, expanded_mask)
        all_activations.append(batch_activations.cpu())  # Move to CPU to save memory
    
    return torch.cat(all_activations, dim=0)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear cache at start if using CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load the ERA model
    model_path = "/workspace/routed-probes/models/models/full_runs/full_run_era_pre_ablation.pt"  # Update this path as needed
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get the model config from the checkpoint or use a base config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # You'll need to specify the correct model architecture parameters here
        config = {
            'd_model': 512,  # Adjust these values based on your model
            'n_layers': 8,
            'n_heads': 16,
            'd_head': 32,
            'd_mlp': 2112,
            'n_ctx': 2048,
            'act_fn': 'gelu',
            'normalization_type': 'LN',
            'd_vocab': 50257,  # GPT-2 vocabulary size
            'tokenizer_name': 'gpt2',  # Use GPT-2 tokenizer
            'use_attn_result': True,
            'device': device,
        }
    
    # Ensure tokenizer configuration is present
    if 'tokenizer_name' not in config:
        config['tokenizer_name'] = 'gpt2'
    if 'd_vocab' not in config:
        config['d_vocab'] = 50257  # GPT-2 vocabulary size
    if 'device' not in config:
        config['device'] = device
    
    # Create model with config
    model = HookedTransformer(config)
    
    # Load the state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Get expanded neuron mask for the middle layer
    middle_layer_idx = model.cfg.n_layers // 2
    expanded_mask = get_expanded_neuron_mask(model, middle_layer_idx)
    num_expanded = expanded_mask.sum().item()
    print(f"Found {num_expanded} expanded neurons in layer {middle_layer_idx}")
    
    # Load and preprocess the dataset
    df = pd.read_csv("classified_forest_stories_1k_balanced.csv")
    stories = df["story"].tolist()
    labels = df["is_forest_story"].tolist()
    
    # Split the dataset
    stories_train, stories_val, labels_train, labels_val = train_test_split(
        stories, labels, test_size=0.2, random_state=42
    )
    
    # Check if precomputed activations exist
    activations_file = "precomputed_era_activations_1k.pt"
    if os.path.exists(activations_file):
        print("Loading precomputed activations...")
        saved_data = torch.load(activations_file)
        train_activations = saved_data['train_activations']
        val_activations = saved_data['val_activations']
    else:
        print("Precomputing activations...")
        # Compute activations for training and validation sets
        train_activations = precompute_activations(
            model, stories_train, middle_layer_idx, 
            expanded_mask=expanded_mask, batch_size=4, device=device
        )
        val_activations = precompute_activations(
            model, stories_val, middle_layer_idx, 
            expanded_mask=expanded_mask, batch_size=4, device=device
        )
        
        # Save activations
        torch.save({
            'train_activations': train_activations,
            'val_activations': val_activations
        }, activations_file)
        print(f"Saved activations to {activations_file}")
    
    # Inspect activations
    print("\nActivation Statistics:")
    print(f"Training activations shape: {train_activations.shape}")
    print(f"Validation activations shape: {val_activations.shape}")
    
    print("\nTraining activations stats:")
    print(f"Mean: {train_activations.mean().item():.4f}")
    print(f"Std: {train_activations.std().item():.4f}")
    print(f"Min: {train_activations.min().item():.4f}")
    print(f"Max: {train_activations.max().item():.4f}")
    print(f"Number of non-zero values: {(train_activations != 0).sum().item()}")
    
    # Check if activations are different between classes
    train_forest_mask = torch.tensor(labels_train, dtype=torch.bool)
    forest_acts = train_activations[train_forest_mask]
    non_forest_acts = train_activations[~train_forest_mask]
    
    print("\nForest vs Non-Forest activation stats:")
    print(f"Forest mean: {forest_acts.mean().item():.4f}")
    print(f"Non-forest mean: {non_forest_acts.mean().item():.4f}")
    print(f"Forest std: {forest_acts.std().item():.4f}")
    print(f"Non-forest std: {non_forest_acts.std().item():.4f}")
    
    # Check if any neurons are consistently different between classes
    forest_means = forest_acts.mean(dim=0)
    non_forest_means = non_forest_acts.mean(dim=0)
    neuron_diffs = (forest_means - non_forest_means).abs()
    print(f"\nNeuron differences between classes:")
    print(f"Max difference: {neuron_diffs.max().item():.4f}")
    print(f"Mean difference: {neuron_diffs.mean().item():.4f}")
    print(f"Number of neurons with diff > 0.1: {(neuron_diffs > 0.1).sum().item()}")
    
    # Clear cache after precomputing/loading activations
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Free the model since we don't need it anymore
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Create datasets with the activations
    train_dataset = PrecomputedActivationsDataset(train_activations, labels_train)
    val_dataset = PrecomputedActivationsDataset(val_activations, labels_val)
    
    # Create dataloaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize classifier
    input_dim = train_activations.shape[1]  # This should be num_expanded
    classifier = ERAActivationClassifier(input_dim)
    classifier.to(device)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    num_epochs = 100
    
    # Initialize wandb
    wandb.init(
        project="forest-probe-era",
        config={
            "architecture": "era-linear-probe",
            "dataset": "forest-stories-1k-balanced",
            "num_expanded_neurons": input_dim,
            "probe_layer": middle_layer_idx,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": 0.1,
            "optimizer": "Adam"
        }
    )
    
    # Log dataset sizes
    wandb.config.update({
        "train_size": len(stories_train),
        "val_size": len(stories_val),
    })
    
    # Training loop
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0
        
        for batch_activations, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_activations = batch_activations.to(device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_activations).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Clear cache after each backward pass
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Validation
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_activations, labels_batch in tqdm(val_loader, desc="Validation"):
                batch_activations = batch_activations.to(device)
                labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)
                
                outputs = classifier(batch_activations).squeeze()
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                total += labels_batch.size(0)
                correct += (predictions == labels_batch).sum().item()
                
                # Clear cache after each validation batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        
        val_accuracy = 100 * correct / total
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Save the trained classifier
    torch.save(classifier.state_dict(), "forest_era_classifier.pt")
    print("Classifier saved to forest_era_classifier.pt")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 