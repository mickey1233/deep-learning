import math
import random
import sys
import MyVar
import matplotlib.pyplot as plt
class Loss:
    def __call__(self,train_x1,train_x2):     
        return (sum([(x1 - x2) ** 2 for x1, x2 in zip(train_x1,train_x2)])).sqrt()
    
if __name__ == "__main__":
    y1 = [MyVar.MyVar(1.0) if  idx == 2 else MyVar.MyVar(0.0) for idx in range(10)]
    y2 = [MyVar.MyVar(1.0) if  idx == 3 else MyVar.MyVar(0.0) for idx in range(10)]
    loss = Loss()
    diff = loss(y1,y2)
    diff.backward()
    MyVar.DFS(diff)
    print(diff)

        
    