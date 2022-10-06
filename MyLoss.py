import os
import sys
import math
import MyVar

class MyLoss:
	def __init__(self):
		pass

	#y1: mlp_output, y2: expected_output
	def __call__(self,y1,y2):
		total = 0.0
		for (yy1,yy2) in zip(y1,y2):
			total += (yy1-yy2)**2
		total = total / len(y1)
		total = total.sqrt()
		return total

class CrossEntropyLoss:
	def __init__(self):
		pass

	def softmax(self,y_list):
		y_exp = [y.exp() for y in y_list]
		y_exp_sum = sum((ye for ye in y_exp))
		y_exp_softmax = [ ye/y_exp_sum for ye in y_exp ]
		return y_exp_softmax

	def log_softmax(self,y_list):
		y_softmax = self.softmax(y_list)
		y_log_softmax = [ ye.ln() for ye in y_softmax ]
		return y_log_softmax

	#y1: good y2: pred
	def __call__(self, y1, y2):
		assert len(y1) == len(y2), 'Length of y1 should be the same as length of y2{}'.format(len(y1),len(y2))

		y_log_softmax = self.log_softmax(y2)
		result = sum( ( yy1* yy2 for yy1, yy2 in zip(y1,y_log_softmax) ) )
		result /= len(y1)

		return -result

if __name__ == '__main__':

	y1 = [MyVar.MyVar(1.0) if idx == 2 else MyVar.MyVar(0.0) for idx in range(10)]
	y2 = [MyVar.MyVar(1.0) if idx == 3 else MyVar.MyVar(0.0) for idx in range(10)]

	loss = MyLoss()

	diff = loss(y1,y2)

	diff.backward()

	MyVar.DFS(diff)


