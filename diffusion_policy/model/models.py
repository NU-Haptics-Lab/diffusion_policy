from torch import nn



class Models(nn.Module):
    """
    A container class for however many models are being used. Inherit from nn.Module so checkpointing works properly
    """
    
    def __init__(self,
            models: dict
        ):
        # must init nn.Module
        super().__init__()
        
        # 
        self.models = nn.ModuleDict()

        for key in models:
            self.models[key] = models[key]
        
    # def eval(self):
    #     for idx, model in self.models.items():
    #         model.eval()
            
    # def train(self):
    #     for idx, model in self.models.items():
    #         model.train()
            
    def __getitem__(self, key):
        return self.models[key]
    
    def step(self):
        for idx, model in self.models.items():
            model.step()
            
    def reset(self):
        for idx, model in self.models.items():
            model.reset()
            
            
# if __name__ == "__main__":
#     models = {"a": object(), "b": object()}
    
#     m = Models(models)
#     pass