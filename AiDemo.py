import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])

Y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

class NeuralNetwork:
    def __init__(self,x,y):
        self.input = x
        self.weights1 =np.random.rand(self.input.shape[1],4)
        self.weights2= np.random.rand(4,1)
        self.y=y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input,self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))

    def backdrop(self):
       # Sai số layer cuối
        delta2 = (self.y - self.output) * sigmoid_derivative(self.output)   # (4,1)

        # Lan truyền về layer 1
        delta1 = np.dot(delta2, self.weights2.T) * sigmoid_derivative(self.layer1)  # (4,4)

        # Gradient đúng
        d_weight2 = np.dot(self.layer1.T, delta2)   # (4,1)
        d_weight1 = np.dot(self.input.T, delta1)    # (2,4)

        # Update
        self.weights1 += d_weight1
        self.weights2 += d_weight2
nn = NeuralNetwork(X,Y)

for i in range(10000):
    nn.feedforward()
    nn.backdrop()
    if i%1000 ==0:
        print(f"Epoch{i}: Loss {np.mean(np.square(Y-nn.output))}")

nn.feedforward()
print("\nKết quả dự đoán: ")
print(nn.output)