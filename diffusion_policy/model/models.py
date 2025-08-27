



class Models:
    """
    A container class for however many models are being used.
    """
    
    def __init__(self,
            models: dict
        ):
        self.models = models
        
    def eval(self):
        for idx, model in self.models.items():
            model.eval()
            
    def train(self):
        for idx, model in self.models.items():
            model.train()
            
    def __getitem__(self, idx):
        return self.models[idx]
    
    def step(self):
        for idx, model in self.models.items():
            model.step()