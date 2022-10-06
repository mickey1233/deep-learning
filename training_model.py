import os
import sys
import random
import MyVar



#from matplotlib.pyplot import MultipleLocator
##目前使用relu##
def activation_function(value):
    return value if value > 0 else 0

def unnecessary_activation_function(value):
    return value

###神經元的sigma(x*w)
class Perceptron:
    def __init__(self,weights,necessary_activation_function): #weights為input數量
        self.weights = [MyVar.MyVar(random.uniform(-1,1)) for _ in range(weights)]
        self.bias = MyVar.MyVar(0)
        self.act_function = necessary_activation_function
           
            
    def __call__(self,x):
        total = 0.0 #x值為float 所以total為float
        total = sum([x * y for x,y in zip(x,self.weights)]) #計算sigma(x*w)
        total+=self.bias #加上bias
        if self.act_function:
            total = total.relu()
        return total
    
    def parameters(self):
        return self.weights + [self.bias]

#得到每層中所有神經元的sigma(x*w)   
class Layer:
    def __init__(self,feature_input_num,feature_output_num,necessary_activation_function):
        #為一個list 每個element為Perceptron的output value
        self.Perceptron = [Perceptron(feature_input_num,necessary_activation_function) for _ in range(feature_output_num)]
        
    def __call__(self,x):
        feature_output = [p(x) for p in self.Perceptron] #為每一層layer中所有Perceptron的value。iter self.Perceptron,並做p(x),呼叫 class Perceptron 然後__call__()
        return feature_output 
   
    def parameters(self):
        return [param for p in self.Perceptron for param in p.parameters()]
#得到每個layer的input_number
class MLP:
    def __init__(self,input_num,final_output_num):
        self.layer = []
        while True:
            self.output_num = input_num
            if self.output_num > final_output_num: 
                self.output_num //= 2
                self.layer.append(Layer(input_num,self.output_num,True))
            else: 
                self.layer.append(Layer(input_num,final_output_num,False))
                break
            input_num //= 2
        
    def __call__(self,x):
        for layer in self.layer:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer1 in self.layer for param in layer1.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
if __name__ == "__main__":
    x = [0.5 for _ in range(64)]
    mlp = MLP(64,10)
    y = mlp(x)
    print(len(mlp.parameters()))
    
        
    

   
   
   
    
        
        
        
        

        
        