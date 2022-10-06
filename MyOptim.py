import sys

class MyOptim:
    def __init__(self,model_parameters,learning_rate=0.01):
        self.model_parameters = model_parameters
        self.lr = learning_rate
        
    def step(self):
        for p in self.model_parameters:
            p.value = p.value - p.grad * self.lr