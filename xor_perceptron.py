import numpy as np

def unitStep(v):
    return 1 if v >= 0 else 0

# Perceptron function
def perceptron(w, x, b):
    v = np.dot(w, x) + b
    return unitStep(v)

# XOR network logic
def xor_logic(X, y, w_and, b_and, w_or, b_or, w_not, b_not, w_final, b_final, epochs, learning_rate):
    for epoch in range(epochs):
        for i in range(len(X)):
            x = X[i]
            y_true = y[i]

            y1 = perceptron(w_and, x, b_and)  
            y2 = perceptron(w_or, x, b_or)  
            y3 = perceptron(w_not, np.array([y1]), b_not)

            final_input = np.array([y2, y3])  
            final_output = perceptron(w_final, final_input, b_final)

            error = y_true - final_output

            if error != 0:
                w_final += learning_rate * error * final_input
                b_final += learning_rate * error

                w_or += learning_rate * error * np.array(x)
                b_or += learning_rate * error

                w_and += learning_rate * error * np.array(x)
                b_and += learning_rate * error

                w_not += learning_rate * error * np.array([y1])  
                b_not += learning_rate * error

    return w_and, b_and, w_or, b_or, w_not, b_not, w_final, b_final

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y = np.array([0, 1, 1, 0]) 

w_and = np.array([2.0, -1.0])
b_and = 2.0
w_or = np.array([2.0, -1.0])
b_or = -1.0
w_not = np.array([1.0])
b_not = -1.0
w_final = np.array([1.0, 1.0])
b_final = -1.0

epochs = 100
learning_rate = 0.1

w_and, b_and, w_or, b_or, w_not, b_not, w_final, b_final = xor_logic(
    X, y, w_and, b_and, w_or, b_or, w_not, b_not, w_final, b_final, epochs, learning_rate
)

for i in range(len(X)):
    x = X[i]
    
    y1 = perceptron(w_and, x, b_and)  
    y2 = perceptron(w_or, x, b_or)   
    y3 = perceptron(w_not, np.array([y1]), b_not)  
    final_input = np.array([y2, y3])  
    y_pred = perceptron(w_final, final_input, b_final)  
    
    print(f"Input: {x}, Predicted Output: {y_pred}, True Output: {y[i]}")
