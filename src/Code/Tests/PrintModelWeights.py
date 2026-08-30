import torch
import collections

def print_pth_file_shapes(file_path):
    """
    Loads a .pth file (state_dict) and prints the shape of each parameter.

    Args:
        file_path (str): The path to the .pth file.
    """
    try:
        # Load the state_dict from the .pth file
        # map_location='cpu' ensures it works even without a GPU
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

        # A .pth file might store just the state_dict or a dictionary
        # containing other info (like epoch number, optimizer state, etc.)
        # We need to find the actual state_dict if it's nested
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        # Also check for 'model_state_dict' as a common key
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif not isinstance(checkpoint, collections.OrderedDict):
            print(f"The file format is not a standard state_dict or checkpoint dictionary. Printing the loaded object directly:")
            print(checkpoint)
            return

        print(f"--- Contents of {file_path} ---")
        for key, value in state_dict.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: (Not a tensor, type: {type(value)})")
        print("---------------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure the file path is correct and it is a valid PyTorch file.")

# Example usage:
# Replace 'path/to/your/model.pth' with your actual file path
# print_pth_file_shapes('path/to/your/model.pth')

print_pth_file_shapes("WorkingCNNVisionWeights2.pth")