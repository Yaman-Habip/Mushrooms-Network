#Import various needed modules
from tensorflow import keras as tf_k #The tensorflow API of keras
from sklearn.model_selection import train_test_split #Use to split data into training and testing sets
from csv import reader #Use to read CSV files

#Read data from file into a list
mushrooms = []
with open("/Users/yamanhabip/PycharmProjects/Mushrooms network/data.csv", mode = "r") as file:

    to_read = reader(file)
    next(to_read)

    for line in to_read:
        mushrooms.append(line)

#Turn strings in list into numerical values
replacements = {}
num = 0

for i in range(len(mushrooms)):
    for j in range(len(mushrooms[i])):
        if mushrooms[i][j] in replacements:
            mushrooms[i][j] = replacements[mushrooms[i][j]]
        else:
            replacements[mushrooms[i][j]] = num
            mushrooms[i][j] = replacements[mushrooms[i][j]]
            num += 1

#Split data into inputs and outputs
input_data = []
output_data = []

for i in mushrooms:
    input_data.append(i[:-1])
    if i[-1:][0] == 16:
        output_data.append(1)
    else:
        output_data.append(0)


#Split data into training and testing data
input_training, input_testing, output_training, output_testing = train_test_split(input_data, output_data, test_size = 0.2)

#Create a network model
network = tf_k.Sequential()

#Add 2 hidden layers to the network, each with four nodes and relu activation
network.add(tf_k.layers.Dense(4, activation = "relu", name = "hidden1"))
network.add(tf_k.layers.Dense(4, activation = "relu", name = "hidden2"))

#Add an output node with sigmoid activation
network.add(tf_k.layers.Dense(1, activation = "sigmoid", name = "output"))

#Put the model together with a gradient descent optimizer, the goal of the training being accuracy,
#and the loss function being a binary loss function because we have a binary of edible or inedible
network.compile(optimizer = "adam", metrics = ["accuracy"], loss = tf_k.losses.binary_crossentropy)

#Train the network with out training data and 25 epochs through the data
network.fit(input_training, output_training, epochs = 20, verbose = 2)

#Test the model with the testing data
network.evaluate(input_testing, output_testing, verbose = 2)