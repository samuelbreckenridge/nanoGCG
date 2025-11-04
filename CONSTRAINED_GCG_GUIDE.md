# Constrained GCG Usage Guide

This guide explains how to use the constrained GCG implementation with your allowable string set.

## Overview

The constrained GCG implementation projects the adversarial suffix to the nearest allowable string every `k` iterations. This ensures the final suffix comes from your predefined set while maintaining attack effectiveness.

## Step 1: Prepare Your Allowable Strings

Create a text file with your allowable strings (one per line):

```bash
# Example: allowable_strings.txt
string number one goes here
another string here
yet another allowable string
...
```

## Step 2: Preprocess the Strings

This computes embeddings and tokenizes all strings for fast search:

```bash
python -m nanogcg.preprocess_allowable_strings \
    --strings_file /path/to/your/allowable_strings.txt \
    --output_dir ./preprocessed_strings \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --batch_size 32
```

**Important**: Use the same model that you'll use for the attack!

This creates:
- `preprocessed_strings/strings.txt` - Original strings
- `preprocessed_strings/tokenized.npy` - Tokenized version
- `preprocessed_strings/embeddings.npy` - Mean-pooled embeddings (~4 bytes per dimension per string)
- `preprocessed_strings/metadata.txt` - Metadata

For 1M strings with embedding dim ~4096, expect ~16GB of embedding storage.

## Step 3: Run Constrained GCG

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanogcg.constrained_gcg import ConstrainedGCG, ConstrainedGCGConfig

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Configure
config = ConstrainedGCGConfig(
    num_steps=250,
    search_width=512,
    topk=256,
    # Constrained GCG parameters
    projection_frequency=10,  # Project every 10 steps
    distance_metric="hybrid",  # "embedding", "loss", or "hybrid" (recommended)
    faiss_k=100,  # Retrieve top-100 candidates
    use_loss_refinement=True,
    preprocessed_dir="./preprocessed_strings",
    project_final_result=True,
)

# Run attack
gcg = ConstrainedGCG(model, tokenizer, config)
result = gcg.run(
    messages="Your prompt here",
    target="Target completion here"
)

print(f"Best loss: {result.best_loss}")
print(f"Best string: {result.best_string}")
```

## Configuration Parameters

### Distance Metrics

- **`embedding`** (fastest): Uses cosine similarity in embedding space
  - Pros: Very fast with FAISS
  - Cons: May not align perfectly with attack objective

- **`hybrid`** (recommended): FAISS retrieves top-k, then evaluates loss
  - Pros: Good balance of speed and effectiveness
  - Cons: Moderate computational cost

- **`loss`** (most accurate): Directly evaluates loss on candidates
  - Pros: Most aligned with attack objective
  - Cons: Slowest, evaluates many candidates

### Key Parameters

- `projection_frequency`: How often to project (default: 10)
  - Lower = more constraint enforcement, potentially slower convergence
  - Higher = more freedom to optimize, may drift from allowable set

- `faiss_k`: Number of candidates to retrieve (default: 100)
  - For "hybrid": retrieve this many, evaluate loss on all
  - For "embedding": only top-1 is used
  - For "loss": retrieves 5x this many

- `project_final_result`: Whether to project the final output (default: True)
  - Ensures final result is always from allowable set

## Using FAISS for Speed (Optional but Recommended)

Install FAISS for much faster search over large string sets:

```bash
# CPU version
pip install faiss-cpu

# GPU version (if you have CUDA)
pip install faiss-gpu
```

Without FAISS, the implementation falls back to brute-force numpy search (slower but functional).

## Example: Quick Test

See `examples/constrained_gcg_example.py` for a complete example with comparison to standard GCG.

```bash
python examples/constrained_gcg_example.py \
    --preprocessed_dir ./preprocessed_strings \
    --projection_frequency 10 \
    --distance_metric hybrid \
    --num_steps 100
```

## Performance Considerations

For 1M strings:

1. **Preprocessing**: One-time cost, ~30-60 minutes depending on hardware
2. **Memory**: ~16GB for embeddings (float32, dim=4096)
3. **Search speed** (with FAISS):
   - Exact search: ~10-50ms per query
   - Approximate search (IVF): ~1-5ms per query
4. **Projection overhead**: Every `projection_frequency` steps, adds 1-5 seconds (depending on `faiss_k` and loss evaluation)

## Tips

1. **Start small**: Test with a smaller subset (10k strings) to validate your approach
2. **Tune `projection_frequency`**: Higher values (20-50) give more freedom, lower values (5-10) enforce constraints more strictly
3. **Monitor loss**: Compare constrained vs unconstrained loss to assess constraint cost
4. **Use GPU**: Both for the model and FAISS (faiss-gpu) for best performance