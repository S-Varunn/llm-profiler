"""
Simple Qwen inference using only HuggingFace Transformers.
No profiler — just model loading, tokenization, and generation.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-4B"
PROMPT = "Explain what a neural network is in one sentence."
MAX_NEW_TOKENS = 50

# =============================================================================
# Load model and tokenizer
# =============================================================================

print(f"Loading model: {MODEL_NAME}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

print(f"Model loaded on {device}")

# =============================================================================
# Generate
# =============================================================================

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

print(f"\nPrompt: {PROMPT}")
print(f"Max new tokens: {MAX_NEW_TOKENS}\n")

start = time.perf_counter()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

elapsed = time.perf_counter() - start

# =============================================================================
# Decode and print results
# =============================================================================

generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Generated text:\n{output_text}\n")
print(f"Tokens generated: {len(generated_ids)}")
print(f"Total time: {elapsed:.3f}s")
print(f"Tokens/sec: {len(generated_ids) / elapsed:.2f}")
