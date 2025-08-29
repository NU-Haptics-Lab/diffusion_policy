

class Losses:
    """
    simple container for storing losses
    """
    def __init__(self,
                 losses: dict
                 ):
        self.losses = losses
        
    def backward(self):
        for key, val in self.losses.items():
            val.backward()