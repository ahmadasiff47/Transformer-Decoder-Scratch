import torch
import time
import psutil
import os
from model_block import TransformerDecoderBlock

def get_process_memory():
    """Calculates the current memory usage of this script in Megabytes (MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_performance_test():
    print("=" * 50)
    print("         MODEL PERFORMANCE BENCHMARK         ")
    print("=" * 50)

    # Initialize model parameters
    d_model = 256
    ffn_dim = 1024
    device = "cpu"
    
    print(f"Instantiating target model on {device.upper()}...")
    model = TransformerDecoderBlock(d_model=d_model, ffn_dim=ffn_dim).to(device)
    
    # Define test configurations: (Batch Size, Sequence Length)
    scenarios = [
        (1, 128),    # Light load: single prompt
        (4, 512),    # Medium load: multiple users
        (8, 1024)    # Heavy load: processing long context
    ]
    
    for batch, seq_len in scenarios:
        print(f"\nEvaluating: Batch Size = {batch} | Sequence Length = {seq_len}")
        
        # Track memory before execution
        initial_mem = get_process_memory()
        
        # Create input tensor
        inputs = torch.randn(batch, seq_len, d_model).to(device)
        
        # Warmup pass to discard initialization overhead
        _ = model(inputs)
        
        # Track start time
        start_time = time.time()
        
        # Run multiple iterations to calculate average latency
        iterations = 10
        for _ in range(iterations):
            _ = model(inputs)
            
        # Track end metrics
        end_time = time.time()
        final_mem = get_process_memory()
        
        # Compute final results
        avg_latency = (end_time - start_time) / iterations
        memory_allocated = max(0.0, final_mem - initial_mem)
        
        print(f"  -> Average Latency : {avg_latency:.4f} seconds")
        print(f"  -> RAM Allocated    : {memory_allocated:.2f} MB")

    print("\n" + "=" * 50)
    print("             BENCHMARK COMPLETED             ")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_performance_test()