from diffusion_policy.common.replay_buffer import ReplayBuffer

class ReplayBufferLoader:
    """
    Loads all replay buffers so that they're available in a singleton
    """

    def __init__(self,
        rb_paths: dict
            ):
        self.rb_paths = rb_paths

        # load replay buffers
        self.rbs = {}
        for key, val in enumerate(self.rb_paths):
            # this will load directly from disk, and not into RAM. There's no noticeable slowdown. You really don't want to load into RAM, so that we don't save the entire dataset into each checkpoint during pickling
            self.rbs[key] = ReplayBuffer.create_from_path(val)

            print("Replay buffer nb datapoints: ", self.rbs[key].n_steps) 
            print("Replay buffer nb episodes: ", self.rbs[key].n_episodes) 