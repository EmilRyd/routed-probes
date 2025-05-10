import sys
print(f"Running with Python executable: {sys.executable}")

import os
import traceback # Added for more detailed error reporting

'''print(f"Current Working Directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

try:
    import torch
    print(f"Successfully imported torch. Version: {torch.__version__}, Path: {torch.__file__}")
except ModuleNotFoundError as e:
    print(f"Failed to import torch. Error: {e}")
    print("Traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred while trying to import torch: {e}")
    print("Traceback:")
    traceback.print_exc()'''

import shared_settings
import tinystories_era
import torch

# It's assumed that shared_settings.py and tinystories_era.py are accessible in the Python path.
# This is typically true if this script is in the same directory (projects/tinystories/)
# and run from the workspace root or the projects/tinystories directory.

def main():
    """
    Configures and runs a base model training experiment.
    """
    # 1. Load SharedExperimentConfig
    # We'll use the global 'cfg' instance from shared_settings.py.
    # You can customize this further by modifying shared_settings.py or creating a new
    # SharedExperimentConfig instance here if needed.
    shared_config = shared_settings.cfg

    # 2. Configure RunTypeConfig for a base model
    # This configuration is based on the example for a base model in README_verbose.md.
    # Key parameters for base model training:
    # - pretrained_model_to_load=None (train from scratch)
    # - expand_model=False
    # - use_gradient_routing=False
    # - num_steps_coherence_finetuning=0
    # - num_steps_forget_set_retraining=0
    # - l1_coeff=0

    # You may want to adjust the number of training steps.
    num_total_training_steps = 20000  # Default value, adjust as per your requirements
                                    # The original README example used 'era_steps', which was 20k
                                    # in some bulk run configurations.

    base_model_run_config = shared_settings.RunTypeConfig(
        label="base_model_script_generated",  # Descriptive label for this run
        pretrained_model_to_load=None,    # Train from scratch
        anneal_gradient_mask_weights=False, # Not relevant for standard base model
        mask_weight_increase_steps=0,     # Not relevant for standard base model
        expand_model=False,               # Do not expand model dimensions
        use_gradient_routing=False,       # No gradient routing
        # The following forget_data parameters are from the README's base_model_cfg example;
        # their impact might be minimal if no unlearning objective is active.
        forget_data_labeling_percentage=1.0,
        drop_labeled_forget_data=False,
        drop_unlabeled_forget_data=False,
        sort_forget_data_by_label=False,
        num_steps_era_training=num_total_training_steps, # Total training steps
        num_steps_coherence_finetuning=0, # No specific coherence finetuning phase
        num_steps_forget_set_retraining=0, # No specific forget set retraining phase
        l1_coeff=0,                       # No L1 regularization
    )

    # 3. Configure a minimal ERAConfig
    # For base model training where expand_model=False and use_gradient_routing=False,
    # most ERAConfig parameters are not used. A default instance should be sufficient.
    minimal_era_config = shared_settings.ERAConfig(
        layers_to_mask=[],  # No layers to mask for base model
        to_expand={},  # No expansion for base model
        masking_scheme="no_routing",  # Use no_routing for base model
        masking_type="demix",  # Use demix as the masking type
        expanded_vs_original_dim_learning_rates={
            "expanded_dim_lr_target": 1.0,
            "original_dim_lr_target": 1.0,
            "expanded_dim_lr_off_target": 1.0,
            "original_dim_lr_off_target": 1.0
        },  # Set all learning rates to 1.0 since we're not using expansion
        include_conditional_bias_term=False,  # No conditional bias term needed
    )
    # If specific ERAConfig defaults need to be overridden even for a base model (e.g.,
    # to ensure certain ERA mechanisms are explicitly disabled if they have active defaults),
    # you could set them here. For example:
    # minimal_era_config.masking_type = "none" # Or an equivalent if applicable.
    # However, typically, these are conditional on expand_model or use_gradient_routing.

    # 4. Call the training function
    print("--- Starting Base Model Training ---")
    print(f"Shared Config Used: {shared_config.transformer_lens_model_name}")
    print(f"Run Type Config Label: {base_model_run_config.label}")
    print(f"Total Training Steps: {base_model_run_config.num_steps_era_training}")
    print("ERA Config: Using default ERAConfig (most params inactive for base model).")
    print("------------------------------------")

    # Call the training function with all required parameters
    tinystories_era.do_era_training_run(
        experiment_cfg=shared_config,
        run_type_cfg=base_model_run_config,
        era_cfg=minimal_era_config,
        random_shuffle_seed=42,  # Fixed seed for reproducibility
        num_validation_stories=1000,  # Number of stories to use for validation
        num_stories_to_retrain=[1000],  # Number of stories to use for retraining
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_save_dir="models",  # Using our local models directory
        model_save_name=f"base_model_{base_model_run_config.label}",
        overwrite_model_saves=True,  # Allow overwriting existing models
        dry_run=False  # Set to True to test configuration without training
    )

    print("--- Base Model Training Finished ---")

if __name__ == "__main__":
    # This script can be extended to parse command-line arguments for parameters like
    # num_total_training_steps, label, model_save_dir, etc., for more flexibility.
    # For now, modify the variables directly in the script if needed.
    main()