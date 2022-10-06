import os
import sys
import math

class MyVar:
	def __init__(self,value,prev=()):
		self.value = value
		self.grad = 0
		self.grad_calculate = None
		self.prev = prev

	def __add__(self,other):
		if not isinstance(other, MyVar):
			other = MyVar(other)

		out = MyVar(self.value + other.value, (self, other))

		def _grad_calculate():
			self.grad +=  1 * out.grad
			other.grad += 1 * out.grad

		out.grad_calculate = _grad_calculate

		return out

	def __mul__(self,other):
		if not isinstance(other, MyVar):
			other = MyVar(other)

		out = MyVar(self.value * other.value, (self, other))

		def _grad_calculate():
			self.grad += other.value * out.grad
			other.grad += self.value * out.grad

		out.grad_calculate = _grad_calculate

		return out

	def __pow__(self, other):
		out = MyVar(self.value**other, (self,))

		def _grad_calculate():
			self.grad += (other * self.value**(other-1)) * out.grad

		out.grad_calculate = _grad_calculate

		return out

	def relu(self):
		if self.value < 0:
			out = MyVar(0, (self,))
		else:
			out = MyVar(self.value, (self,))
            
		def _grad_calculate():
			if out.value > 0:
				self.grad += 1 * out.grad
			else:
				self.grad += 0 * out.grad

		out.grad_calculate = _grad_calculate

		return out

	def exp(self):
		out = MyVar(math.exp(self.value),(self,))

		def _grad_calculate():
			self.grad += math.exp(self.value) * out.grad

		out.grad_calculate = _grad_calculate

		return out

	def ln(self):
		out = MyVar(math.log(self.value),(self,))

		def _grad_calculate():
			self.grad += 1/self.value * out.grad

		out.grad_calculate = _grad_calculate

		return out

	def sqrt(self):
		return self**0.5

	def backward(self):
		backward_list = []
		visited = set()

		def build_backward_list(v):
			if v not in visited:
				visited.add(v)
				for child in v.prev:
					build_backward_list(child)
				backward_list.append(v)

		build_backward_list(self)

		self.grad = 1
		for v in reversed(backward_list):
			if v.grad_calculate != None:
				v.grad_calculate()

	def __neg__(self): # -self
		return self * -1

	def __radd__(self, other): # other + self
		return self + other

	def __sub__(self, other): # self - other
		return self + (-other)

	def __rsub__(self, other): # other - self
		return other + (-self)

	def __rmul__(self, other): # other * self
		return self * other

	def __truediv__(self, other): # self / other
		return self * other**-1

	def __rtruediv__(self, other): # other / self
		return other * self**-1

	def __str__(self):
		return 'MyVar(data={}, prev={}, grad={})'.format(self.value,self.prev,self.grad)

def DFS(myvar):
	if len(myvar.prev) == 0:
		print(myvar)
		return 0
	else:
		indent = 0
		for xxxxx in myvar.prev:
			indent = max(indent,DFS(xxxxx)+1)
		print(' '*indent,myvar)
		return indent


if __name__ == '__main__':
	a = MyVar(10)
	b = MyVar(20)
	c = a/b
	d = c**-1
	e = d.exp()
	#f = e.ln()
	g = e + a
	#print(g)
	q = g + b
	#DFS(q)

	q.backward()

	DFS(q)


