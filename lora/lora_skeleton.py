import torch
import torch.nn as nn

# print PATH

# LoRA implementation
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        # The original projection layer is saved and its weights and bias (if any) are frozen (won't be updated during backpropagation).
        # Goal: keep base model unchanged and train only the LoRA matrices.
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_dim = original_linear.in_features # input dimension of the layer
        out_dim = original_linear.out_features # output dimension of the layer
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device)) # shape: (r, in_dim)
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device)) # shape: (out_dim, r)
        # A, B are two trainable matrices. Total number of trainable parameters: r * (in_dim + out_dim)
        
        # Initialise A with He initialization (ensures the variance of activations remains stable during training)
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x) # original projection output
        lora_out = (x @ self.A.T) @ self.B.T # LoRA output
        return base_out + lora_out * (self.alpha / self.r) # Scaled output
        # Goal: Add the LoRA output to the frozen base output (train only LoRA matrices)



# This function processes input data into tokenized sequences, handles chunking, and ensures proper length.
def process_sequences(texts, tokenizer, max_length=512, stride=256):
    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]

            # If chunk is shorter than max_length, pad it with the pad token.
            # This ensures all input sequences have the same length → Enables batching
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )

            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)