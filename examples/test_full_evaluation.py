"""Test script to evaluate timing for evaluating all allowable transactions.

This script helps determine if direct loss evaluation over all 800k transactions
is faster than running multiple GCG trials.

Usage:
    python examples/test_full_evaluation.py \
        --preprocessed_dir ./preprocessed \
        --model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --message "What is the balance of account 12345?" \
        --target "The balance is $0" \
        --test_size 1000 \
        --batch_size 64
"""

import argparse
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanogcg.constrained_gcg import AllowableStringSet


def main():
    parser = argparse.ArgumentParser(description="Test full evaluation timing")
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        required=True,
        help="Directory with preprocessed allowable strings"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model to use"
    )
    parser.add_argument(
        "--message",
        type=str,
        default="What is the total?",
        help="User message (will append {optim_str} at the end)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="The answer is 42",
        help="Target generation"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of transactions to test for timing estimation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--run_full",
        action="store_true",
        help="If set, run evaluation on ALL transactions (not just test set)"
    )
    parser.add_argument(
        "--use_prefix_cache",
        action="store_true",
        default=True,
        help="Use prefix caching to speed up evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    print("="*80)
    print("TESTING FULL EVALUATION TIMING")
    print("="*80)

    # Load model and tokenizer
    print(f"\n[1/5] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("✓ Model loaded")

    # Initialize allowable string set
    print(f"\n[2/5] Loading allowable strings from {args.preprocessed_dir}")
    string_set = AllowableStringSet(
        args.preprocessed_dir,
        model,
        tokenizer,
        use_faiss=True,
    )
    n_total = len(string_set.tokenized)
    print(f"✓ Loaded {n_total:,} allowable strings")

    # Prepare prompt components
    print(f"\n[3/5] Preparing prompt")
    messages = [{"role": "user", "content": args.message + "{optim_str}"}]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "")
    before_str, after_str = template.split("{optim_str}")
    target = " " + args.target  # Add space before target

    # Tokenize
    before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(args.device)
    after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)
    target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)

    # Embed
    embedding_layer = model.get_input_embeddings()
    before_embeds = embedding_layer(before_ids)
    after_embeds = embedding_layer(after_ids)
    target_embeds = embedding_layer(target_ids)

    print(f"  Before: {before_ids.shape[1]} tokens")
    print(f"  After: {after_ids.shape[1]} tokens")
    print(f"  Target: {target_ids.shape[1]} tokens")

    # Compute prefix cache if enabled
    prefix_cache = None
    if args.use_prefix_cache:
        print(f"\n[4/5] Computing prefix cache")
        with torch.no_grad():
            output = model(inputs_embeds=before_embeds, use_cache=True)
            prefix_cache = output.past_key_values
        print("✓ Prefix cache computed")
    else:
        print(f"\n[4/5] Skipping prefix cache (disabled)")

    # Test evaluation on subset
    print(f"\n[5/5] Testing evaluation on {args.test_size:,} transactions")
    print(f"  Batch size: {args.batch_size}")

    # Sample random indices
    test_indices = np.random.choice(n_total, size=min(args.test_size, n_total), replace=False)

    print(f"\n  Starting evaluation...")
    start_time = time.time()

    best_idx, best_loss = string_set.find_nearest_by_loss(
        test_indices,
        before_embeds,
        after_embeds,
        target_embeds,
        target_ids,
        prefix_cache=prefix_cache,
        batch_size=args.batch_size,
        minimize_target_prob=False,
    )

    elapsed = time.time() - start_time

    print(f"  ✓ Evaluated {len(test_indices):,} transactions in {elapsed:.2f}s")
    print(f"  Time per transaction: {elapsed/len(test_indices)*1000:.2f}ms")
    print(f"  Time per batch ({args.batch_size}): {elapsed/(len(test_indices)/args.batch_size):.3f}s")

    # Extrapolate to full dataset
    print(f"\n{'='*80}")
    print("TIMING ESTIMATES")
    print(f"{'='*80}")

    estimated_total_time = elapsed * (n_total / len(test_indices))
    print(f"Estimated time for all {n_total:,} transactions:")
    print(f"  {estimated_total_time:.1f}s = {estimated_total_time/60:.1f} min = {estimated_total_time/3600:.2f} hrs")

    print(f"\nBest from test set:")
    print(f"  Index: {best_idx}")
    print(f"  Loss: {best_loss:.4f}")
    best_transaction = tokenizer.decode(string_set.tokenized[best_idx])
    print(f"  Transaction: {best_transaction[:200]}...")

    # Optionally run on full dataset
    if args.run_full:
        print(f"\n{'='*80}")
        print(f"RUNNING FULL EVALUATION ({n_total:,} transactions)")
        print(f"{'='*80}")

        user_input = input(f"\nThis will take approximately {estimated_total_time/60:.1f} minutes. Continue? [y/N]: ")
        if user_input.lower() != 'y':
            print("Aborted.")
            return

        all_indices = np.arange(n_total)

        print(f"\nEvaluating all {n_total:,} transactions...")
        start_time = time.time()

        best_idx_full, best_loss_full = string_set.find_nearest_by_loss(
            all_indices,
            before_embeds,
            after_embeds,
            target_embeds,
            target_ids,
            prefix_cache=prefix_cache,
            batch_size=args.batch_size,
            minimize_target_prob=False,
        )

        elapsed_full = time.time() - start_time

        print(f"\n{'='*80}")
        print("FULL EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Time: {elapsed_full:.1f}s = {elapsed_full/60:.1f} min")
        print(f"Best index: {best_idx_full}")
        print(f"Best loss: {best_loss_full:.4f}")

        best_transaction_full = tokenizer.decode(string_set.tokenized[best_idx_full])
        print(f"\nBest transaction:")
        print(f"  {best_transaction_full}")

        # Test the result
        print(f"\n{'='*80}")
        print("TESTING BEST TRANSACTION")
        print(f"{'='*80}")

        full_prompt = args.message + best_transaction_full
        test_messages = [{"role": "user", "content": full_prompt}]
        test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(test_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Prompt: {test_prompt[:500]}...")
        print(f"\nGenerated: {generated}")

    else:
        print(f"\nTo run full evaluation, use --run_full flag")
        print(f"Example:")
        print(f"  python {__file__} --preprocessed_dir {args.preprocessed_dir} \\")
        print(f"    --model_name {args.model_name} --run_full")


if __name__ == "__main__":
    main()