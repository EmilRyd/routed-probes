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
from projects.tinystories import shared_settings
import wandb
import os

# Configuration
USE_PCA = False  # Set to False to skip PCA dimensionality reduction
PCA_COMPONENTS = 64  # Number of components to use when PCA is enabled

class PrecomputedActivationsDataset(Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

class ActivationClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))

def get_model_activations(model, text, layer_idx):
    """Get activations from a specific layer for the input text."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        # Get activations from the specified layer
        activations = cache["resid_post", layer_idx][0]  # Shape: [seq_len, d_model]
        # Average across sequence length
        avg_activations = activations.mean(dim=0)  # Shape: [d_model]
    return avg_activations

def get_model_activations_batch(model, texts, layer_idx):
    """Get activations from a specific layer for a batch of texts."""
    tokens = model.to_tokens(texts)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        # Get activations from the specified layer [batch_size, seq_len, d_model]
        activations = cache["resid_post", layer_idx]
        # Average across sequence length for each item in batch
        avg_activations = activations.mean(dim=1)  # Shape: [batch_size, d_model]
    return avg_activations

def precompute_activations(model, stories, layer_idx, batch_size, device):
    """Precompute activations for all stories."""
    all_activations = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(stories), batch_size), desc="Precomputing activations"):
        batch_stories = stories[i:i + batch_size]
        batch_activations = get_model_activations_batch(model, batch_stories, layer_idx)
        all_activations.append(batch_activations.cpu())  # Move to CPU to save memory
    
    return torch.cat(all_activations, dim=0)

def main():
    # Set device to MPS if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear MPS cache at start
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load the base model
    cfg = shared_settings.cfg
    config = get_pretrained_model_config(cfg.transformer_lens_model_name, device=device)
    model = HookedTransformer.from_pretrained(cfg.transformer_lens_model_name, device=device)
    
    # We'll use the middle layer of the model
    middle_layer_idx = model.cfg.n_layers // 2
    d_model = model.cfg.d_model  # Store this before deleting model
    
    # Load and preprocess the dataset
    df = pd.read_csv("classified_forest_stories_1k_balanced.csv")
    stories = df["story"].tolist()
    labels = df["is_forest_story"].tolist()
    
    # Split the dataset
    stories_train, stories_val, labels_train, labels_val = train_test_split(
        stories, labels, test_size=0.2, random_state=42
    )
    
    # Check if precomputed activations exist
    activations_file = f"precomputed_activations_1k{'_pca' if USE_PCA else ''}.pt"
    pca_explained_variance = None
    if os.path.exists(activations_file):
        print("Loading precomputed activations...")
        saved_data = torch.load(activations_file)
        train_activations = saved_data['train_activations']
        val_activations = saved_data['val_activations']
        if 'pca_explained_variance' in saved_data:
            pca_explained_variance = saved_data['pca_explained_variance']
    else:
        print("Precomputing activations...")
        # Compute activations for training and validation sets
        batch_size = 4  # Can use larger batch size for precomputation
        train_activations = precompute_activations(model, stories_train, middle_layer_idx, batch_size, device)
        val_activations = precompute_activations(model, stories_val, middle_layer_idx, batch_size, device)
        
        if USE_PCA:
            # Apply PCA to reduce dimensionality
            print(f"Reducing dimensionality from {d_model} to {PCA_COMPONENTS} using PCA...")
            
            # Convert to numpy for PCA
            train_activations_np = train_activations.cpu().numpy()
            val_activations_np = val_activations.cpu().numpy()
            
            # Fit PCA on training data and transform both training and validation
            pca = PCA(n_components=PCA_COMPONENTS)
            train_activations_pca = pca.fit_transform(train_activations_np)
            val_activations_pca = pca.transform(val_activations_np)
            
            # Convert back to torch tensors
            train_activations = torch.tensor(train_activations_pca, dtype=torch.float32)
            val_activations = torch.tensor(val_activations_pca, dtype=torch.float32)
            
            pca_explained_variance = pca.explained_variance_ratio_.sum()
            print(f"Explained variance ratio: {pca_explained_variance:.3f}")
        
        # Save activations
        torch.save({
            'train_activations': train_activations,
            'val_activations': val_activations,
            'pca_explained_variance': pca_explained_variance
        }, activations_file)
        print(f"Saved activations to {activations_file}")
    
    # Clear MPS cache after precomputing/loading activations
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # Free the base model since we don't need it anymore
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # Create datasets
    train_dataset = PrecomputedActivationsDataset(train_activations, labels_train)
    val_dataset = PrecomputedActivationsDataset(val_activations, labels_val)
    
    # Create dataloaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize classifier
    input_dim = PCA_COMPONENTS if USE_PCA else d_model
    classifier = ActivationClassifier(input_dim)
    classifier.to(device)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    num_epochs = 100
    
    # Initialize wandb
    wandb.init(
        project="forest-probe",
        config={
            "architecture": "linear-probe",
            "dataset": "forest-stories-1k-balanced",
            "input_dim": d_model,
            "use_pca": USE_PCA,
            "pca_components": PCA_COMPONENTS if USE_PCA else None,
            "pca_explained_variance": pca_explained_variance,
            "probe_layer": middle_layer_idx,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": 0.01,
            "optimizer": "Adam",
            "model_name": cfg.transformer_lens_model_name,
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
            
            # Clear MPS cache after each backward pass
            if device.type == "mps":
                torch.mps.empty_cache()
        
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
                
                # Clear MPS cache after each validation batch
                if device.type == "mps":
                    torch.mps.empty_cache()
        
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
    torch.save(classifier.state_dict(), "forest_classifier_not_pca.pt")
    print("Classifier saved to forest_classifier_not_pca.pt")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 