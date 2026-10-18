import numpy as np

from diffusion_policy.samplers.aiet_erlenmeyer_flask_sampler import AIETErlenmeyerFlaskSampler


class AIETWujiTreeChunkSampler(AIETErlenmeyerFlaskSampler):
    """
    Samples of avatar_intelligence's tree_to_zarr.py (aiet_wuji_tree_cond_bc_*): every row is one complete sample --
    its obs (the start node's state features, split into the obs groups, and its edge_quality label) and its whole
    action chunk, stored in the row as 'action_chunk' [N, horizon, action_dim] (the edge's target, then the best path's
    below it). The rows are NOT consecutive time steps: an episode is just a block of one tree's samples (thousands of
    rows each instead of one 4-row episode per sample -- the per-episode overhead). So the action is the row's own
    chunk, not a window of action_rel_indices rows (whose length must still be the chunk's: the model's horizon), and
    every row is a valid sample (common_indices: restrict_to_valid_future false, pad_after 0).
    """

    CHUNK_KEY = "action_chunk"

    def get_action_trajectory(self, ep_idx, key):
        return self.get_key_sample(self.CHUNK_KEY, ep_idx)[0]  # (horizon, action_dim)

    def build_action_chunk(self, train_idx):
        """The vectorized GPU preload's batched get_action_trajectory: [N] episode-relative rows -> [N, horizon, action_dim]."""
        return np.asarray(self.indices.get_sequence_by_train_indices_and_key(np.asarray(train_idx), self.CHUNK_KEY), dtype=np.float32)
