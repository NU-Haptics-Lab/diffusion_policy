



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