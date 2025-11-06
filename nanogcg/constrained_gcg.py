"""Constrained GCG: GCG with projection to an allowable set of strings.

This module implements a constrained version of the GCG attack where the adversarial
suffix is periodically projected to the nearest string from a predefined allowable set.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import transformers

from nanogcg.gcg import GCG, GCGConfig, GCGResult, AttackBuffer
from nanogcg.utils import find_executable_batch_size

logger = logging.getLogger("nanogcg")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu). "
        "Falling back to slower brute-force search."
    )


@dataclass
class ConstrainedGCGConfig(GCGConfig):
    """Configuration for constrained GCG with allowable string set.

    All GCGConfig parameters are inherited. Additional parameters:

    Args:
        projection_frequency: Project to nearest allowable string every k iterations
        distance_metric: How to measure distance to allowable strings
            - "embedding": Pure embedding-based (fastest)
            - "loss": Pure loss-based (most accurate, slowest)
            - "hybrid": Embedding to filter + loss to refine (recommended)
        faiss_k: Number of candidates to retrieve from FAISS/brute-force search
        use_loss_refinement: Whether to evaluate loss on retrieved candidates (for hybrid mode)
        preprocessed_dir: Path to directory with preprocessed allowable strings
            Should contain: strings.txt, tokenized.npy, embeddings.npy
        project_final_result: Whether to project the final result before returning
    """
    projection_frequency: int = 10
    distance_metric: Literal["embedding", "loss", "hybrid"] = "hybrid"
    faiss_k: int = 100
    use_loss_refinement: bool = True
    preprocessed_dir: str = None
    project_final_result: bool = True


class AllowableStringSet:
    """Manages the set of allowable strings with efficient nearest neighbor search."""

    def __init__(
        self,
        preprocessed_dir: str,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        use_faiss: bool = True,
    ):
        """Initialize the allowable string set.

        Args:
            preprocessed_dir: Directory containing preprocessed data
                (strings.txt, tokenized.npy, embeddings.npy)
            model: The transformer model (used for embedding and loss computation)
            tokenizer: The model's tokenizer
            use_faiss: Whether to use FAISS for fast search (if available)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.embedding_layer = model.get_input_embeddings()

        preprocessed_dir = Path(preprocessed_dir)

        # Load tokenized strings first (this is the source of truth for count)
        logger.info(f"Loading allowable strings from {preprocessed_dir}")
        tokenized_data = np.load(preprocessed_dir / "tokenized.npy", allow_pickle=True)
        self.tokenized = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_data]

        # Get actual number of strings from tokenized data
        n_strings = len(tokenized_data)

        # Don't decode strings - not needed for the attack, only tokenized versions are used
        # This saves significant time
        self.strings = None  # We'll use n_strings for counts instead

        # Load embeddings using memmap (matches how we saved it in preprocessing)
        embeddings_path = preprocessed_dir / "embeddings.npy"
        embed_dim = self.embedding_layer.weight.shape[1]

        self.embeddings = np.memmap(
            embeddings_path,
            dtype=np.float32,
            mode='r',
            shape=(n_strings, embed_dim)
        )
        logger.info(f"Loaded {n_strings} allowable strings with embeddings shape {self.embeddings.shape}")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / (norms + 1e-8)

        # Build or load search index
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        if self.use_faiss:
            faiss_index_path = preprocessed_dir / "faiss.index"
            if faiss_index_path.exists():
                logger.info(f"Loading FAISS index from {faiss_index_path}")
                self.index = faiss.read_index(str(faiss_index_path))
                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
            else:
                logger.warning(f"FAISS index not found at {faiss_index_path}, building from scratch...")
                self._build_faiss_index()
        else:
            logger.info("Using brute-force search (install FAISS for faster search)")
            self.index = None

    def _build_faiss_index(self):
        """Build FAISS index for fast approximate nearest neighbor search."""
        logger.info("Building FAISS index...")
        embed_dim = self.embeddings_normalized.shape[1]
        n_strings = len(self.tokenized)

        # Use IVF index for all dataset sizes
        # Adjust number of clusters based on dataset size
        if n_strings < 1000:
            nlist = max(1, n_strings // 100)  # Very small datasets
        elif n_strings < 10000:
            nlist = max(10, n_strings // 100)  # Small datasets
        elif n_strings < 100000:
            nlist = max(50, n_strings // 1000)  # Medium datasets
        else:
            nlist = 100  # Large datasets (1M strings)

        logger.info(f"Building IVF index with {nlist} clusters for {n_strings} strings")

        quantizer = faiss.IndexFlatIP(embed_dim)
        self.index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(self.embeddings_normalized)
        self.index.add(self.embeddings_normalized)

        # Set number of clusters to probe during search
        # More probes = more accurate but slower
        self.index.nprobe = min(10, max(1, nlist // 10))

        logger.info(f"FAISS IVF index built with {self.index.ntotal} vectors, nprobe={self.index.nprobe}")

    def _embed_tokens(self, token_ids: Tensor) -> np.ndarray:
        """Compute mean-pooled embedding for a token sequence.

        Args:
            token_ids: Tensor of shape (1, seq_len) or (seq_len,)

        Returns:
            Normalized embedding of shape (1, embed_dim)
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        with torch.no_grad():
            embeds = self.embedding_layer(token_ids.to(self.device))  # (1, seq_len, embed_dim)
            mean_embed = embeds.mean(dim=1)  # (1, embed_dim)

            # Normalize for cosine similarity
            mean_embed = mean_embed / (mean_embed.norm(dim=1, keepdim=True) + 1e-8)

            return mean_embed.cpu().numpy().astype(np.float32)

    def find_nearest_by_embedding(self, query_ids: Tensor, k: int = 100) -> tuple:
        """Find k nearest allowable strings by embedding similarity.

        Args:
            query_ids: Token IDs of query sequence, shape (1, seq_len) or (seq_len,)
            k: Number of nearest neighbors to retrieve

        Returns:
            Tuple of (indices, distances) where:
                - indices: np.ndarray of shape (k,) with indices of nearest strings
                - distances: np.ndarray of shape (k,) with cosine similarities
        """
        # Compute query embedding
        query_embed = self._embed_tokens(query_ids)

        k = min(k, len(self.tokenized))  # Can't retrieve more than total

        if self.use_faiss:
            # FAISS search (returns similarities for METRIC_INNER_PRODUCT)
            distances, indices = self.index.search(query_embed, k)
            return indices[0], distances[0]
        else:
            # Brute-force search using numpy
            similarities = (self.embeddings_normalized @ query_embed.T).squeeze()
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
            return top_k_indices, similarities[top_k_indices]

    def find_nearest_by_loss(
        self,
        candidate_indices: np.ndarray,
        before_embeds: Tensor,
        after_embeds: Tensor,
        target_embeds: Tensor,
        target_ids: Tensor,
        prefix_cache=None,
        batch_size: int = 32,
        minimize_target_prob: bool = False,
    ) -> tuple:
        """Evaluate loss for candidate strings and return best.

        Args:
            candidate_indices: Indices of candidate strings to evaluate
            before_embeds: Embeddings before the optimized string
            after_embeds: Embeddings after the optimized string
            target_embeds: Embeddings of the target string
            target_ids: Token IDs of the target
            prefix_cache: Optional KV cache for prefix
            batch_size: Batch size for loss evaluation

        Returns:
            Tuple of (best_index, best_loss)
        """
        n_candidates = len(candidate_indices)
        all_losses = []

        # Prepare candidates
        candidate_ids_list = [self.tokenized[idx] for idx in candidate_indices]

        # Pad to same length
        max_len = max(len(ids) for ids in candidate_ids_list)
        padded_candidates = []
        for ids in candidate_ids_list:
            if len(ids) < max_len:
                # Pad with tokenizer's pad token (or 0 if not available)
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded = torch.cat([ids, torch.full((max_len - len(ids),), pad_id, dtype=torch.long)])
            else:
                padded = ids
            padded_candidates.append(padded)

        candidate_ids_tensor = torch.stack(padded_candidates).to(self.device)

        # Compute losses in batches
        prefix_cache_batch = []
        for i in range(0, n_candidates, batch_size):
            batch_ids = candidate_ids_tensor[i:i + batch_size]
            current_batch_size = batch_ids.shape[0]

            with torch.no_grad():
                # Embed the candidate suffixes
                optim_embeds = self.embedding_layer(batch_ids)

                # Construct full input
                if prefix_cache:
                    if not prefix_cache_batch or current_batch_size != batch_size:
                        prefix_cache_batch = [
                            [x.expand(current_batch_size, -1, -1, -1) for x in prefix_cache[i]]
                            for i in range(len(prefix_cache))
                        ]
                    input_embeds = torch.cat([
                        optim_embeds,
                        after_embeds.repeat(current_batch_size, 1, 1),
                        target_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    outputs = self.model(
                        inputs_embeds=input_embeds,
                        past_key_values=prefix_cache_batch,
                        use_cache=True
                    )
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(current_batch_size, 1, 1),
                        optim_embeds,
                        after_embeds.repeat(current_batch_size, 1, 1),
                        target_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    outputs = self.model(inputs_embeds=input_embeds)

                logits = outputs.logits

                # Compute loss
                shift = input_embeds.shape[1] - target_ids.shape[1]
                shift_logits = logits[..., shift - 1 : -1, :].contiguous()
                shift_labels = target_ids.repeat(current_batch_size, 1)

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none"
                )
                loss = loss.view(current_batch_size, -1).mean(dim=-1)

                # If minimize_target_prob is True, negate the loss
                if minimize_target_prob:
                    loss = -loss

                all_losses.append(loss)

                # Clean up to prevent memory accumulation
                del outputs
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        all_losses = torch.cat(all_losses)
        best_idx = all_losses.argmin().item()
        best_loss = all_losses[best_idx].item()

        return candidate_indices[best_idx], best_loss

    def get_tokenized_string(self, index: int) -> Tensor:
        """Get tokenized string at given index."""
        return self.tokenized[index].to(self.device)


class ConstrainedGCG(GCG):
    """GCG with periodic projection to allowable string set."""

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: ConstrainedGCGConfig,
    ):
        """Initialize Constrained GCG.

        Args:
            model: The transformer model
            tokenizer: The model's tokenizer
            config: Configuration including both GCG params and constraint params
        """
        super().__init__(model, tokenizer, config)
        self.config: ConstrainedGCGConfig = config

        if config.preprocessed_dir is None:
            raise ValueError("preprocessed_dir must be specified in ConstrainedGCGConfig")

        # Initialize allowable string set
        logger.info("Initializing allowable string set...")
        self.string_set = AllowableStringSet(
            config.preprocessed_dir,
            model,
            tokenizer,
            use_faiss=FAISS_AVAILABLE,
        )
        logger.info("Allowable string set initialized.")

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        """Run constrained GCG optimization.

        This extends the base GCG.run() method by adding periodic projection
        to the allowable string set.

        Args:
            messages: Conversation messages or string
            target: Target generation string

        Returns:
            GCGResult with losses and optimized strings (projected to allowable set)
        """
        # Call parent's setup (everything before the optimization loop)
        # We'll need to replicate the parent's run method with projection added

        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            from transformers import set_seed
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        import copy
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        ]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize probe sampling if needed
        if config.probe_sampling_config:
            # Same as parent class
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer
            draft_before_ids = self.draft_tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            draft_after_ids = self.draft_tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            self.draft_target_ids = self.draft_tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

            self.draft_before_embeds, self.draft_after_embeds, self.draft_target_embeds = [
                self.draft_embedding_layer(ids)
                for ids in (draft_before_ids, draft_after_ids, self.draft_target_ids)
            ]

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(inputs_embeds=self.draft_before_embeds, use_cache=True)
                    self.draft_prefix_cache = output.past_key_values

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        # Main optimization loop with projection
        for step in tqdm(range(config.num_steps)):
            # Standard GCG step: compute gradient and sample candidates
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                from nanogcg.gcg import sample_ids_from_grad, filter_ids

                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                if self.config.probe_sampling_config is None:
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)
                    current_loss = loss.min().item()
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                else:
                    current_loss, optim_ids = find_executable_batch_size(
                        self._compute_candidates_loss_probe_sampling, batch_size
                    )(input_embeds, sampled_ids)

                # Update the buffer
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            # PROJECT TO ALLOWABLE SET every k iterations
            if (step + 1) % config.projection_frequency == 0:
                logger.info(f"Step {step + 1}: Projecting to allowable set...")
                optim_ids = self._project_to_allowable_set(optim_ids)

                # Re-evaluate loss after projection
                projected_loss = self._evaluate_loss(optim_ids)
                buffer.add(projected_loss, optim_ids)

                logger.info(f"  Projected loss: {projected_loss:.4f}")

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        # Final projection (if enabled)
        if config.project_final_result:
            logger.info("Projecting final result to allowable set...")
            best_ids = buffer.get_best_ids()
            best_ids = self._project_to_allowable_set(best_ids)
            best_loss = self._evaluate_loss(best_ids)
            best_str = tokenizer.batch_decode(best_ids)[0]

            result = GCGResult(
                best_loss=best_loss,
                best_string=best_str,
                losses=losses,
                strings=optim_strings,
            )
        else:
            min_loss_index = losses.index(min(losses))
            result = GCGResult(
                best_loss=losses[min_loss_index],
                best_string=optim_strings[min_loss_index],
                losses=losses,
                strings=optim_strings,
            )

        return result

    def _project_to_allowable_set(self, current_ids: Tensor) -> Tensor:
        """Project current suffix to nearest allowable string.

        Args:
            current_ids: Current suffix token IDs, shape (1, seq_len)

        Returns:
            Token IDs of nearest allowable string, shape (1, seq_len)
        """
        config = self.config

        if config.distance_metric == "embedding":
            # Pure embedding-based: fast but may not align with loss
            indices, _ = self.string_set.find_nearest_by_embedding(current_ids, k=1)
            best_ids = self.string_set.get_tokenized_string(indices[0])
            return best_ids.unsqueeze(0)

        elif config.distance_metric == "hybrid":
            # Hybrid: retrieve top-k by embedding, refine by loss
            indices, _ = self.string_set.find_nearest_by_embedding(current_ids, k=config.faiss_k)

            if config.use_loss_refinement:
                best_idx, _ = self.string_set.find_nearest_by_loss(
                    indices,
                    self.before_embeds,
                    self.after_embeds,
                    self.target_embeds,
                    self.target_ids,
                    prefix_cache=self.prefix_cache,
                    batch_size=config.batch_size if config.batch_size else 32,
                    minimize_target_prob=config.minimize_target_prob,
                )
                best_ids = self.string_set.get_tokenized_string(best_idx)
            else:
                # Just use the top embedding match
                best_ids = self.string_set.get_tokenized_string(indices[0])

            return best_ids.unsqueeze(0)

        elif config.distance_metric == "loss":
            # Pure loss-based: most accurate but slowest
            # Retrieve more candidates and evaluate all
            k = min(config.faiss_k * 5, len(self.string_set.tokenized))
            indices, _ = self.string_set.find_nearest_by_embedding(current_ids, k=k)

            best_idx, _ = self.string_set.find_nearest_by_loss(
                indices,
                self.before_embeds,
                self.after_embeds,
                self.target_embeds,
                self.target_ids,
                prefix_cache=self.prefix_cache,
                batch_size=config.batch_size if config.batch_size else 32,
                minimize_target_prob=config.minimize_target_prob,
            )
            best_ids = self.string_set.get_tokenized_string(best_idx)
            return best_ids.unsqueeze(0)

        else:
            raise ValueError(f"Unknown distance metric: {config.distance_metric}")

    def _evaluate_loss(self, optim_ids: Tensor) -> float:
        """Evaluate loss for a single suffix.

        Args:
            optim_ids: Token IDs of suffix, shape (1, seq_len)

        Returns:
            Loss value
        """
        embedding_layer = self.embedding_layer

        if self.prefix_cache:
            input_embeds = torch.cat([
                embedding_layer(optim_ids),
                self.after_embeds,
                self.target_embeds,
            ], dim=1)
            prefix_cache_batch = [[x for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]
            outputs = self.model(
                inputs_embeds=input_embeds,
                past_key_values=prefix_cache_batch,
                use_cache=True
            )
        else:
            input_embeds = torch.cat([
                self.before_embeds,
                embedding_layer(optim_ids),
                self.after_embeds,
                self.target_embeds,
            ], dim=1)
            outputs = self.model(inputs_embeds=input_embeds)

        logits = outputs.logits
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()
        shift_labels = self.target_ids

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # If minimize_target_prob is True, negate the loss for consistency
        if self.config.minimize_target_prob:
            loss = -loss

        return loss.item()


def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[ConstrainedGCGConfig] = None,
) -> GCGResult:
    """Run constrained GCG with allowable string set.

    Args:
        model: The model to use for optimization
        tokenizer: The model's tokenizer
        messages: The conversation to use for optimization
        target: The target generation
        config: The constrained GCG configuration

    Returns:
        GCGResult with optimized strings from the allowable set
    """
    if config is None:
        raise ValueError("ConstrainedGCGConfig must be provided")

    import logging
    logger.setLevel(getattr(logging, config.verbosity))

    gcg = ConstrainedGCG(model, tokenizer, config)
    result = gcg.run(messages, target)
    return result