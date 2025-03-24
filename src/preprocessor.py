import h5py
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Define the tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def preprocess_time_series(data, decimals: int = 3) -> str:
    """
    Preprocesses a time-series dataset using the LLMTIME scheme, automatically selecting an alpha
    to ensure values lie within the 0-10 range.

    Args:
        data (np.ndarray): Time-series data of shape (T, 2), where T = number of time steps.
        decimals (int): Number of decimal places to round to. (can be a hyperparameter to be tuned)

    Returns:
        str: Preprocessed string suitable for tokenization.
    """
    data = np.array(data, dtype=np.float32)  # Ensures numerical stability

    # Compute alpha as the 99th percentile divided by 10
    p99 = np.percentile(data, 99)  # Find the 99th percentile
    alpha = p99 / 10 if p99 > 0 else 1  # Ensure alpha is never zero

    # Scale and round
    scaled_data = np.round(data / alpha, decimals)

    # Convert to string format (LLMTIME style)
    formatted_sequence = ";".join(
        ",".join(f"{value:.{decimals}f}" for value in timestep)
        for timestep in scaled_data
    )

    return formatted_sequence


def preprocess_all_time_series(trajectories, decimals: int = 3):
    """
    Preprocesses all time-series data using the LLMTIME scheme.

    Args:
        trajectories (np.ndarray): Time-series data of shape (num_systems, T, 2).
        decimals (int): Number of decimal places to round to.
        test_size (float): Fraction of data to use for validation.

    Returns:
        tuple[list[str], list[str]]: Preprocessed training and validation sequences.
    """
    all_sequences = []
    
    num_systems = trajectories.shape[0]
    
    for system_id in range(num_systems):
        # Extract prey and predator for one system
        prey = trajectories[system_id, :, 0]
        predator = trajectories[system_id, :, 1]
        data = np.column_stack((prey, predator))
        
        # Ensure numerical stability
        data = np.array(data, dtype=np.float32)
        
        # Compute alpha using the 99th percentile divided by 10
        p99 = np.percentile(data, 99)
        alpha = p99 / 10 if p99 > 0 else 1
        
        # Scale and round
        scaled_data = np.round(data / alpha, decimals)
        
        # Convert to LLMTIME-style text sequence
        formatted_sequence = ";".join(
            ",".join(f"{value:.{decimals}f}" for value in timestep)
            for timestep in scaled_data
        )
        
        all_sequences.append(formatted_sequence)
    

    # First 6 systems as test data
    test_texts = all_sequences[:6]
    
    # Remaining systems for training and validation
    remaining_sequences = all_sequences[6:]
    
    # Split remaining sequences into training and validation sets
    train_texts, val_texts = train_test_split(
        remaining_sequences, test_size=0.2, random_state=42
    )
    
    return train_texts, val_texts, test_texts



def tokenize_sequence(sequence: str):
    """
    Tokenize the preprocessed sequence using Qwen's tokenizer.

    Args:
        sequence (str): Preprocessed sequence.

    Returns:
        list[int]: Tokenized sequence.
    """
    tokens = tokenizer(sequence, return_tensors="pt")["input_ids"].tolist()[0]
    return tokens



if __name__ == "__main__":
    # Load example data
    with h5py.File("data/lotka_volterra_data.h5", "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
    
    # Example system
    system_id = 0
    prey = trajectories[system_id, :, 0]
    predator = trajectories[system_id, :, 1]
    data = np.column_stack((prey, predator))
    print("\nInput data from system_id=0:")
    print(data)
    
    # Preprocess the data
    preprocessed_sequence = preprocess_time_series(data)
    print("\nPreprocessed Sequence from system_id=0:")
    print(preprocessed_sequence)
    
    # Tokenize the sequence
    tokenized_sequence = tokenize_sequence(preprocessed_sequence)
    print("\nTokenized Sequence from system_id=0:")
    print(tokenized_sequence)

    train_texts, val_texts, test_texts = preprocess_all_time_series(trajectories)
    print("\nExample train sequence:")
    print(train_texts[0])
    print("\nExample validation sequence:")
    print(val_texts[0])
    print(f"\nShape of train_texts: {np.shape(train_texts)}, shape of val_texts: {np.shape(val_texts)}, shape of test_texts: {np.shape(test_texts)}")
    