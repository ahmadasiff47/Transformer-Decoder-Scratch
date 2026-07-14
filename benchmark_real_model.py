import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_process_memory():
    """Calculates the current memory usage of this script in Megabytes (MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_real_model_benchmark():
    print("=" * 60)
    print("        REAL-WORLD MODEL BENCHMARK (SMALLEST LLM)       ")
    print("=" * 60)

    device = "cpu"
    # Using a very lightweight model (Qwen 0.5B) so it runs fast on Mac CPU
    model_name = "Qwen/Qwen2.5-0.5B"
    
    initial_mem = get_process_memory()
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model weight (this might take a minute on first run)...")
    start_load_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    end_load_time = time.time()
    
    post_load_mem = get_process_memory()
    
    print(f"\nModel Loaded Successfully!")
    print(f"  -> Loading Time     : {end_load_time - start_load_time:.2f} seconds")
    print(f"  -> Model Memory Size: {post_load_mem - initial_mem:.2f} MB")
    
    # Let's test text generation speed
    prompt = "Artificial Intelligence is"
    print(f"\nWarmup and testing text generation with prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Measure token generation speed
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    end_gen = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    gen_time = end_gen - start_gen
    tokens_per_sec = total_tokens / gen_time
    
    print(f"\nGenerated Output:\n'{generated_text}'")
    print("-" * 60)
    print(f"Performance Metrics:")
    print(f"  -> Tokens Generated : {total_tokens}")
    print(f"  -> Generation Time  : {gen_time:.2f} seconds")
    print(f"  -> Generation Speed : {tokens_per_sec:.2f} tokens/sec")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_real_model_benchmark()