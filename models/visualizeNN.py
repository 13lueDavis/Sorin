# This Libraray is modified based the work by Milo Spencer-Harper and Oli Blum, https://stackoverflow.com/a/37366154/10404826
# On top of that, I added support for showing weights (linewidth, colors, etc.)
# Contributor: Jianzheng Liu
# Contact: jzliu.100@gmail.com

import matplotlib as mpl
from matplotlib import pyplot
from math import cos, sin, atan
from palettable.tableau import Tableau_10
from time import localtime, strftime
import numpy as np
import webcolors
import sys
import glob
import json

from keras.models import model_from_json

LINECOLORS = ['#F45531', '#20CE99']
NODECOLORS = ['#999999', '#555555']
maxWeight = 1

COLOR = '#ffffff'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['figure.facecolor'] = '1A1A1D'
mpl.rcParams['axes.facecolor'] = '1A1A1D'

class Neuron():
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name

    def draw(self, neuron_radius, id=-1):
        circle = pyplot.Circle((self.x, self.y), zorder=4, radius=neuron_radius, color=NODECOLORS[0], fill=webcolors.hex_to_rgb(NODECOLORS[1]))
        pyplot.gca().add_patch(circle)
        if self.name is None:
            pyplot.gca().text(self.x, self.y-0.15, str(id), size=10, ha='center', zorder=5)
        else:
            if self.name in ['BUY', 'SELL']:
                pyplot.gca().text(self.x, self.y+0.8, self.name, size=10, ha='center', zorder=5)
            else:
                pyplot.gca().text(self.x, self.y-1-0.4*((id+1)%2), self.name, size=10, ha='center', zorder=5)

class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, node_names):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, node_names)

    def __intialise_neurons(self, number_of_neurons, node_names):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y, node_names[iteration] if node_names is not None else None)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight=0.4, textoverlaphandler=None):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        # assign colors to lines depending on the sign of the weight
        color=LINECOLORS[0]
        if weight > 0: color=LINECOLORS[1]

        # assign different linewidths to lines depending on the size of the weight
        abs_weight = abs(weight)
        if abs_weight > 0.5:
            linewidth = 10*abs_weight
        elif abs_weight > 0.8:
            linewidth =  100*abs_weight
        else:
            linewidth = abs_weight
        linewidth *= 2/maxWeight

        # draw the weights and adjust the labels of weights to avoid overlapping
        if abs_weight > 0.5:
            # while loop to determine the optimal locaton for text lables to avoid overlapping
            index_step = 2
            num_segments = 10
            txt_x_pos = neuron1.x - x_adjustment+index_step*(neuron2.x-neuron1.x+2*x_adjustment)/num_segments
            txt_y_pos = neuron1.y - y_adjustment+index_step*(neuron2.y-neuron1.y+2*y_adjustment)/num_segments
            while ((not textoverlaphandler.getspace([txt_x_pos-0.5, txt_y_pos-0.5, txt_x_pos+0.5, txt_y_pos+0.5])) and index_step < num_segments):
                index_step = index_step + 1
                txt_x_pos = neuron1.x - x_adjustment+index_step*(neuron2.x-neuron1.x+2*x_adjustment)/num_segments
                txt_y_pos = neuron1.y - y_adjustment+index_step*(neuron2.y-neuron1.y+2*y_adjustment)/num_segments

            # print("Label positions: ", "{:.2f}".format(txt_x_pos), "{:.2f}".format(txt_y_pos), "{:3.2f}".format(weight))
            a=pyplot.gca().text(txt_x_pos, txt_y_pos, "{:3.2f}".format(weight), size=8, ha='center')
            a.set_bbox(dict(facecolor='white', alpha=0))
            # print(a.get_bbox_patch().get_height())

        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=linewidth, color=color)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0, weights=None, textoverlaphandler=None):
        j=0 # index for neurons in this layer
        for neuron in self.neurons:
            i=0 # index for neurons in previous layer
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weights[i,j], textoverlaphandler)
                    i=i+1
            neuron.draw( self.neuron_radius, id=j+1 )
            j=j+1

        # write Text
        # x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        # if layerType == 0:
        #     pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        # elif layerType == -1:
        #     pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        # else:
        #     pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

# A class to handle Text Overlapping
# The idea is to first create a grid space, if a grid is already occupied, then
# the grid is not available for text labels.
class TextOverlappingHandler():
    # initialize the class with the width and height of the plot area
    def __init__(self, width, height, grid_size=0.2):
        self.grid_size = grid_size
        self.cells = np.ones((int(np.ceil(width / grid_size)), int(np.ceil(height / grid_size))), dtype=bool)

    # input test_coordinates(bottom left and top right),
    # getspace will tell you whether a text label can be put in the test coordinates
    def getspace(self, test_coordinates):
        x_left_pos = int(np.floor(test_coordinates[0]/self.grid_size))
        y_botttom_pos = int(np.floor(test_coordinates[1]/self.grid_size))
        x_right_pos = int(np.floor(test_coordinates[2]/self.grid_size))
        y_top_pos = int(np.floor(test_coordinates[3]/self.grid_size))
        if self.cells[x_left_pos, y_botttom_pos] and self.cells[x_left_pos, y_top_pos] \
        and self.cells[x_right_pos, y_top_pos] and self.cells[x_right_pos, y_botttom_pos]:
            for i in range(x_left_pos, x_right_pos):
                for j in range(y_botttom_pos, y_top_pos):
                    self.cells[i, j] = False

            return True
        else:
            return False

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, node_names):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, node_names)
        self.layers.append(layer)

    def draw(self, weights_list, saveFilename):
        fig2 = pyplot.figure(2, figsize=(9, 8))
        # vertical_distance_between_layers and horizontal_distance_between_neurons are the same with the variables of the same name in layer class
        vertical_distance_between_layers = 6
        horizontal_distance_between_neurons = 2
        overlaphandler = TextOverlappingHandler(\
            self.number_of_neurons_in_widest_layer*horizontal_distance_between_neurons,\
            len(self.layers)*vertical_distance_between_layers, grid_size=0.2 )

        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == 0:
                layer.draw( layerType=0 )
            elif i == len(self.layers)-1:
                layer.draw( layerType=-1, weights=weights_list[i-1], textoverlaphandler=overlaphandler)
            else:
                layer.draw( layerType=i, weights=weights_list[i-1], textoverlaphandler=overlaphandler)


        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.tight_layout()
        pyplot.savefig(saveFilename, dpi=300, facecolor='#1A1A1D', bbox_inches="tight")
        pyplot.show()

class DrawNN():
    # para: neural_network is an array of the number of neurons
    # from input layer to output layer, e.g., a neural network of 5 nerons in the input layer,
    # 10 neurons in the hidden layer 1 and 1 neuron in the output layer is [5, 10, 1]
    # para: weights_list (optional) is the output weights list of a neural network which can be obtained via classifier.coefs_
    def __init__( self, neural_network, weights_list=None, inputNames=None, outputNames=None):
        self.neural_network = neural_network
        self.weights_list = weights_list
        self.input_names = inputNames
        self.output_names = outputNames
        # if weights_list is none, then create a uniform list to fill the weights_list
        if weights_list is None:
            weights_list=[]
            for first, second in zip(neural_network, neural_network[1:]):
                tempArr = np.ones((first, second))*0.4
                weights_list.append(tempArr)
            self.weights_list = weights_list

    def draw( self, saveFilename ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for i,l in enumerate(self.neural_network):
            network.add_layer(l, self.input_names if i==0 else (self.output_names if i==len(self.neural_network)-1 else None))
        network.draw(self.weights_list, saveFilename)


def main():
    global maxWeight
    modelName = sys.argv[1]

    modelFilename = "./"+modelName+".json"
    paramsFilename = './'+modelName+'/params.json'
    weightsFilename = './'+modelName+'.h5'
    saveFilename = './visualizations/'+modelName+'.vis.png'

    try:
        model_json_file = open(modelFilename, 'r')
        params_json_file = open(paramsFilename, 'r')
    except Exception:
        print('No model with name: '+modelName)
        modelName = glob.glob('./*/model.json')[0].split('/')[1]
        print('Trying to load model: '+modelName+' instead...')
        modelFilename = "./"+modelName+"/model.json"
        paramsFilename = './'+modelName+'/params.json'
        weightsFilename = './'+modelName+'/weights.h5'
        saveFilename = './'+modelName+'/visualization.png'
        try:
            model_json_file = open(modelFilename, 'r')
            params_json_file = open(paramsFilename, 'r')
        except Exception:
            raise Exception('No models found...')


    loaded_model_json = model_json_file.read()
    model_json_file.close()
    loaded_params_json = params_json_file.read()
    params_json_file.close()

    params = json.loads(loaded_params_json)
    model = model_from_json(loaded_model_json)
    model.load_weights(weightsFilename)

    inputNames = [indicator['TYPE'] for indicator in params['indicators']]
    outputNames = ['SELL', 'BUY']

    architecture = [np.squeeze(model.layers[0].input_shape)[1]]
    weights = []
    maxWeight = 0.
    for layer in model.layers:
        architecture.append(np.squeeze(layer.output_shape)[1])
        weights.append(np.squeeze(layer.get_weights()[0]))
        maxWeight = max(maxWeight, np.max(abs(np.array(layer.get_weights()[0]))))

    network = DrawNN(architecture, weights, inputNames, outputNames)
    network.draw(saveFilename)

    return

if __name__ == "__main__":
    main()
