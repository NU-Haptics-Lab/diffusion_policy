from diffusion_policy.common.replay_buffer import ReplayBuffer

class ReplayBufferLoader:
    """
    Loads all replay buffers so that they're available in a singleton
    """

    def __init__(self,
        rb_paths: dict,
        do_loading: bool = True,
        modes = {}, # read/write/all: 'r', 'w', 'a'
            ):
        self.rb_paths = rb_paths
        self.modes = modes
        
        # setting for actually loading the replay buffer or just creating the structure. Used during inference.
        self.do_loading = do_loading
        
    def setup(self):

        # load replay buffers
        self.rbs = {}
        for key, val in self.rb_paths.items():
            # this will load directly from disk, and not into RAM. There's no noticeable slowdown. You really don't want to load into RAM, so that we don't save the entire dataset into each checkpoint during pickling

            if self.do_loading:
                if key not in self.modes:
                    raise ValueError("mode not specified for replay buffer {}".format(key))
                
                rb = ReplayBuffer.create_from_path(val, mode=self.modes[key])
                self.rbs[key] = rb
                
                print(key + ": replay buffer nb datapoints: ", self.rbs[key].n_steps) 
                print(key + ": replay buffer nb episodes: ", self.rbs[key].n_episodes) 
            else:
                rb = None
                self.rbs[key] = rb
                
            
    def __getitem__(self, key):
        # key check
        if key not in self.rbs:
            print("WARNING: {} not in replay buffer loader!".format(key))
            return None

        return self.rbs[key]
    
    def __len__(self):
        return len(self.rbs)
    
    def update(self, other):
        """
        add all of other's replay buffers to my replay buffer dict
        """
        assert(isinstance(other, ReplayBufferLoader))
        self.rbs.update(other.rbs)