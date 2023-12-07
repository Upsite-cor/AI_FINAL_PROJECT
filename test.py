import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)  # To make repeatable
LEARNING_RATE = 0.01
EPOCHS = 20

def read_data():
    # Your data goes here
    filedata = np.genfromtxt('./yellow-plastic-data2.txt', delimiter='')

    np.random.shuffle(filedata)
    data = filedata[0:20]
    print('\n data test first 20 elements: \n', data)
    return data

def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for i in range(neuron_count):
        for j in range(1, (input_count + 1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    print("hidden_layer_w: ", weights)
    return weights

# Declare matrices and vectors representing the neurons.
hidden_layer_w = layer_w(25, 4)
hidden_layer_y = np.zeros(25)
hidden_layer_error = np.zeros(25)

output_layer_w = layer_w(1, 25)
output_layer_y = np.zeros(1)
output_layer_error = np.zeros(1)

chart_x = []
chart_y_train = []

def show_learning(epoch_no, train_loss, predicted_r):
    global chart_x
    global chart_y_train
    print('epoch no:', epoch_no, ', train_loss:', '%6.4f' % train_loss, ', predicted r:', '%6.4f' % predicted_r)
    chart_x.append(epoch_no + 1)
    chart_y_train.append(train_loss)


def plot_learning(chart_y_test):
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x):
    global hidden_layer_y
    global output_layer_y
    # Activation function for hidden layer
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w, x)
        hidden_layer_y[i] = sigmoid(z)
        #hidden_layer_y[i] = np.tanh(z)

    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))
    # Activation function for output layer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = z  # No activation for regression

def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error
    # Backpropagate error for output neuron
    error_prime = output_layer_y - y_truth  # Loss derivative for regression
    output_layer_error[0] = error_prime
    for i, y in enumerate(hidden_layer_y):
        #derivative = 1.0 - y ** 2  # tanh derivative
        derivative = y*(1-y) #derivative of sigmoid
        weighted_error = output_layer_w[0][i + 1] * output_layer_error[0]
        hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w

    # Update hidden layer weights
    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= x * LEARNING_RATE * error

    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

    # Update output layer weights
    output_layer_w[0][1:] -= hidden_output_array[1:] * LEARNING_RATE * output_layer_error[0]

# Network training loop
data = read_data()
index_list = list(range(len(data))) #the range of the columns 
#print(index_list)


chart_x = []
chart_y_train = []
chart_y_test = []

def show_learning(epoch_no, train_loss, predicted_r, y_true):
    global chart_x, chart_y_train
    print('epoch no:', epoch_no)
    print('train_loss:', train_loss)
    print('predicted r:', predicted_r)
    print('actual r:', y_true)   
    chart_x.append(epoch_no + 1)
    chart_y_train.append(train_loss)

def plot_learning(chart_y_test):
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()

def final_weights():
   with open('./finalweights.txt', 'w') as file:
    file.write("FINAL WEIGHTS FOR W NEURON:\n")
    for i, w in enumerate(hidden_layer_w):
        file.write(str(w) + '\n')

    file.write("FINAL WEIGHTS FOR F NEURON:\n")
    file.write(str(output_layer_w[0][1:]))
    # All write operations should be within the 'with' block


best_loss = float('inf')
PATIENCE=5
counter = 0



# Network training loop
for i in range(EPOCHS):
    np.random.shuffle(index_list)
    total_loss = 0

    # Training loop
    for j in index_list:
        
        x = np.concatenate((np.array([1.0]), data[j][:4]))  # Input angles and biased 1 input
        
        y_true = data[j][4]  # Red output value
        print(y_true)
        forward_pass(x)
        error = output_layer_y[0] - y_true
        total_loss += abs(error)

        backward_pass(y_true)
        adjust_weights(x)

    # Calculate mean training loss
    mean_loss = total_loss / len(index_list)
    predicted_r = output_layer_y[0]  # Predicted red (r) value
    show_learning(i, mean_loss, predicted_r, y_true)

    # Calculate test error after each epoch
    correct_test_results = 0
    total_test_loss = 0

    for j in range(len(data)):  # Use test data or separate test set if available
        x_test = np.concatenate((np.array([1.0]), data[j][:4]))
        y_true_test = data[j][4]

        forward_pass(x_test)
        error_test = output_layer_y[0] - y_true_test
        total_test_loss += abs(error_test)

    mean_test_loss = total_test_loss / len(data)
    chart_y_test.append(mean_test_loss)


    if i == EPOCHS - 1:
        final_weights()

    if mean_test_loss < best_loss:
        best_loss = mean_test_loss
        count = 0
    else:
        counter+=1
    
    if counter>=PATIENCE:
        print(f"Early stopping: No improvement in test loss for {PATIENCE} epochs. Stopping training.")
        break  # Break the training loop if the patience limit is reached
    

# Plot learning curve
plot_learning(chart_y_test)