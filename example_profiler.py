"""
Example usage of the LLM Inference Profiler.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_inference_profiler import LLMProfiler

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-4B"
PROMPT = "Explain what a neural network is in one sentence."
MAX_NEW_TOKENS = 50
OUTPUT_FILE = "profile_output.json"

print(f"Loading model: {MODEL_NAME}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

profiler = LLMProfiler(model=model, tokenizer=tokenizer)

output_text = profiler.generate(
    prompt=PROMPT,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
)

profiler.summary()

profiler.save(OUTPUT_FILE)

results = profiler.results()
