"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Find the index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    last_pos = int(real_positions[-1].item())                 # scalar index

    # Keep the strong baseline feature: last real token of the final layer.
    final_vec = hidden_states[-1, last_pos]  # (hidden_dim,)

    # Add a compact description of late-layer representation dynamics without
    # concatenating full hidden vectors from extra layers.
    prev1_vec = hidden_states[-2, last_pos]
    prev2_vec = hidden_states[-3, last_pos]
    prev4_vec = hidden_states[-5, last_pos]
    prev8_vec = hidden_states[-9, last_pos]

    eps = final_vec.new_tensor(1e-6)
    n_real = float(real_positions.numel())
    seq_frac = final_vec.new_tensor(n_real / float(attention_mask.numel()))

    final_norm = torch.linalg.vector_norm(final_vec)
    prev1_norm = torch.linalg.vector_norm(prev1_vec)
    prev4_norm = torch.linalg.vector_norm(prev4_vec)

    dynamics = torch.stack(
        [
            seq_frac,
            final_norm,
            prev1_norm,
            prev4_norm,
            final_norm / (prev1_norm + eps),
            F.cosine_similarity(final_vec, prev1_vec, dim=0, eps=1e-8),
            F.cosine_similarity(final_vec, prev4_vec, dim=0, eps=1e-8),
            F.cosine_similarity(final_vec, prev8_vec, dim=0, eps=1e-8),
            torch.linalg.vector_norm(final_vec - prev1_vec),
            torch.linalg.vector_norm(final_vec - prev4_vec),
            torch.linalg.vector_norm(prev1_vec - prev2_vec),
        ]
    )

    feature = torch.cat([final_vec, dynamics], dim=0)

    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
