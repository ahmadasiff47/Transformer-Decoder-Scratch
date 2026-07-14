Building a Transformer Decoder Block from Scratch & LLM Benchmarking 🚀

Hey there! This is my project for Week 1 of the AI Internship. I wanted to really understand how modern Language Models work under the hood, so I built a standard Transformer Decoder block entirely from scratch using PyTorch. To wrap things up, I also ran some performance benchmarks on my local hardware using a real-world pre-trained LLM.

 What's Inside the Code?
model_block.py: This is where the magic happens. I coded the Causal Self-Attention mechanism (including Query, Key, Value projections and the causal mask), Layer Normalization, Residual Connections, and Feed-Forward Networks from scratch.
benchmark.py: A clean script I wrote to test how my custom block handles different types of workloads (Light, Medium, and Heavy) to check its scalability.
benchmark_real_model.py: A pipeline using Hugging Face to test actual text generation speed on my local machine using Alibaba's `Qwen2.5-0.5B` model.

 Performance Highlights (Local CPU Benchmark)
I tested the text generation speed on my Apple Silicon CPU, and the results were actually pretty awesome:
Custom Block Latency (Heavy Workload):** ~0.051 seconds per block.
Local LLM Generation Speed (Qwen2.5-0.5B):** **24.34 tokens per second** ⚡ (Super fast for local CPU inference!)

Built with code and curiosity as part of my AI Engineering journey.
