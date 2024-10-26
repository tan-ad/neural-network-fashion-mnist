import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# fake dataset
def create_data(num_points, num_classes):
    X = np.zeros((num_points*num_classes, 2))
    y = np.zeros(num_points*num_classes, dtype='uint8')
    for class_number in range(num_classes):
        ix = range(num_points*class_number,num_points*(class_number+1))
        r = np.linspace(0.0, 1, num_points) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, num_points) + np.random.randn(num_points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense():
    def __init__(self, num_inputs, num_neurons): 

        self.weights = 0.01*np.random.randn(num_inputs, num_neurons) # array of random numbers with mean 0 variance 1 in shape parameters 
        self.biases = np.zeros((1, num_neurons)) # first parameter is shape


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) # probabilities
        self.output = norm_values

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_prediction, y_true):
        num_samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7) # clip limits the upper and lower bound of values. if exceeded, it will set value to the bound.

        if len(y_true.shape) == 1: # one dimensional, not one hot encoded
            correct_confidences = y_prediction_clipped[range(num_samples), y_true]
        elif len(y_true.shape) == 2: # 2d, uses one hot encoding
            correct_confidences = np.sum(y_prediction_clipped*y_true, axis=1) # each value is multiplied by the corresponding value in the same position in the other matrix. e.g. A[0][0]*B[0][0]. then sum each row

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def main():
    X, y = create_data(100, 3) # two input features, 100 points, 3 classes/output possibilities

    layer1 = Layer_Dense(2, 3) # second parameter is arbitrary, just number of neurons we choose to have in second layer
    activation1 = Activation_ReLU()

    layer2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    layer1.forward(X) 
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output) # array of probabilities for each input point (3*100 = 300)

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    # print(activation2.output[:5])
    print(loss)
    predictions = np.argmax(layer2.output, axis=1)
    accuracy = np.mean(predictions == y)
    print(accuracy)



if(__name__=="__main__"):
    main()