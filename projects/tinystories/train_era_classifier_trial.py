import torch
import torch.nn as nn
import torch.backends.mps
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
import numpy as np
from sklearn.model_selection import train_test_split
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
    # In ERA models, expanded neurons are initialized to 0 in the MLP layers
    # We can identify them by checking which neurons have weights initialized to 0
    mlp_weights = model.blocks[layer_idx].mlp.W_in.weight
    # A neuron is considered expanded if all its weights are 0
    expanded_mask = (mlp_weights == 0).all(dim=1)
    return expanded_mask

def get_model_activations_batch(model, texts, layer_idx, expanded_mask=None):
    """Get activations from a specific layer for a batch of texts."""
    tokens = model.to_tokens(texts)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        # Get MLP activations [batch_size, seq_len, d_mlp]
        activations = cache["mlp_out", layer_idx]
        
        # If expanded_mask is provided, only keep expanded neurons
        if expanded_mask is not None:
            activations = activations[:, :, expanded_mask]
            
        # Average across sequence length for each item in batch
        avg_activations = activations.mean(dim=1)  # Shape: [batch_size, d_mlp] or [batch_size, num_expanded]
    return avg_activations

def precompute_activations(model, stories, layer_idx, expanded_mask=None, batch_size=4):
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear MPS cache at start if using MPS
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # Load the ERA model
    model_path = "models/debugging_demix/demix_debug_era"  # Update this path as needed
    print(f"Loading model from {model_path}")
    model = HookedTransformer.from_pretrained(
        model_path,
        device=device
    )
    
    # Get expanded neuron mask for the middle layer
    middle_layer_idx = model.cfg.n_layers // 2
    expanded_mask = get_expanded_neuron_mask(model, middle_layer_idx)
    num_expanded = expanded_mask.sum().item()
    print(f"Found {num_expanded} expanded neurons in layer {middle_layer_idx}")
    
    # Load and preprocess the dataset
    df = pd.read_csv("classified_forest_stories_1k_balanced.csv")
    
    # Take just 20 examples for the trial run
    df_small = pd.concat([
        df[df['is_forest_story'] == 1].head(10),
        df[df['is_forest_story'] == 0].head(10)
    ])
    
    stories = df_small["story"].tolist()
    labels = df_small["is_forest_story"].tolist()
    print(f"Using {len(stories)} stories for trial run ({sum(labels)} forest, {len(labels)-sum(labels)} non-forest)")
    
    # Split the dataset - using more validation data since dataset is tiny
    stories_train, stories_val, labels_train, labels_val = train_test_split(
        stories, labels, test_size=0.4, random_state=42
    )
    
    print("Computing activations...")
    # Compute activations for training and validation sets
    train_activations = precompute_activations(
        model, stories_train, middle_layer_idx, 
        expanded_mask=expanded_mask, batch_size=2
    )
    val_activations = precompute_activations(
        model, stories_val, middle_layer_idx, 
        expanded_mask=expanded_mask, batch_size=2
    )
    
    print(f"Activation shapes: train {train_activations.shape}, val {val_activations.shape}")
    
    # Clear MPS cache after precomputing activations
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # Free the model since we don't need it anymore
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # Create datasets with the activations
    train_dataset = PrecomputedActivationsDataset(train_activations, labels_train)
    val_dataset = PrecomputedActivationsDataset(val_activations, labels_val)
    
    # Create dataloaders - using batch size of 2 since dataset is tiny
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize classifier
    input_dim = train_activations.shape[1]  # This should be num_expanded
    classifier = ERAActivationClassifier(input_dim)
    classifier.to(device)
    
    # Training setup - fewer epochs for trial
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    num_epochs = 10
    
    print("\nStarting training...")
    # Training loop
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0
        
        for batch_activations, labels_batch in train_loader:
            batch_activations = batch_activations.to(device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = classifier(batch_activations).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Clear MPS cache after each backward pass
            if device.type == "mps":
                torch.mps.empty_cache()
        
        # Validation
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_activations, labels_batch in val_loader:
                batch_activations = batch_activations.to(device)
                labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)
                
                outputs = classifier(batch_activations).squeeze()
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                total += labels_batch.size(0)
                correct += (predictions == labels_batch).sum().item()
                
                # Clear MPS cache after each validation batch
                if device.type == "mps":
                    torch.mps.empty_cache()
        
        val_accuracy = 100 * correct / total
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print()
    
    # Save the trained classifier
    torch.save(classifier.state_dict(), "forest_era_classifier_trial.pt")
    print("Trial classifier saved to forest_era_classifier_trial.pt")

if __name__ == "__main__":
    main() 