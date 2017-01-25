#!/usr/bin/python3

import sys

if(len(sys.argv) != 6):
	print("Usage:", sys.argv[0], "input_image output_prefix depth width epochs")
	print("e.g. image-learner.py images/monalisa.jpg monalisa 25 8 50")
	sys.exit()


import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import lasagne
import theano
import theano.tensor as T

import pylab

import time

input_image = sys.argv[1]
output_prefix = sys.argv[2]
depth = int(sys.argv[3]) # number of hidden layers
width = int(sys.argv[4]) # width of the hidden layers
epochs = int(sys.argv[5])

im = Image.open(input_image)

plt.imshow(np.asarray(im))
pylab.axis('off')

im = np.asarray(im)

shape = im.shape;
 
# positions will hold the positions of all pixels
positions = np.empty([shape[0]*shape[1], 2], dtype=np.float32)

# rgb_values will hold the rgb values of all pixels
rgb_values = np.empty([shape[0]*shape[1], 3], dtype=np.float32)

for x in range(0, shape[0]):
    for y in range(0, shape[1]):
        rgb_values[x*shape[1]+y,:] = np.asarray(im)[x,y][0:3]
        positions[x*shape[1]+y,0] = float(x)/shape[0]
        positions[x*shape[1]+y,1] = float(y)/shape[1]
        
class MultiplicativeGatingLayer(lasagne.layers.MergeLayer):
  
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape
    
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]
    
    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]
    
def highway_dense(incoming, Wh=lasagne.init.Orthogonal(), bh=lasagne.init.Constant(0.0),
                  Wt=lasagne.init.Orthogonal(), bt=lasagne.init.Constant(-4.0),
                  nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    # regular layer
    l_h = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh,
                               nonlinearity=nonlinearity)
    # gate layer
    l_t = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                               nonlinearity=T.nnet.sigmoid)
    
    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)
    
    input_data = T.matrix('X')
positions_data = T.matrix('P')
output_data = T.matrix('y')

from lasagne.nonlinearities import leaky_rectify, very_leaky_rectify, softmax, sigmoid, tanh, linear

positions_layer = lasagne.layers.InputLayer((None, 2), positions_data);

nl = very_leaky_rectify
init = lasagne.init.GlorotUniform()

last_hidden_layer = lasagne.layers.DenseLayer(positions_layer,
                                    width, nonlinearity=nl,
                                    W=init)

for x in range(1, depth):
    #last_hidden_layer = lasagne.layers.DenseLayer(
    #                                last_hidden_layer,
    #                                last_hidden_layer,
    #                                width, nonlinearity=nl,
    #                                W=init
    #                                )
    last_hidden_layer = highway_dense(last_hidden_layer, Wh=lasagne.init.Orthogonal(), bh=lasagne.init.Constant(0.0),
                  Wt=lasagne.init.Orthogonal(), bt=lasagne.init.Constant(-4.0),
                  nonlinearity=nl)


layer_last = lasagne.layers.DenseLayer(last_hidden_layer, 3)
                                     
reconstructed_output1 = lasagne.layers.get_output(layer_last)

loss = lasagne.objectives.squared_error(reconstructed_output1, output_data) 

loss = loss.mean()# + 1e-3 * lasagne.regularization.regularize_network_params(layer_last, lasagne.regularization.l1)

params = lasagne.layers.get_all_params(layer_last, trainable=True)
lr = theano.shared(np.float32(0.000001))
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lr)

train_fn = theano.function([positions_data, output_data], loss, updates=updates)
loss = 0

losses = []

def iterate_minibatches(pos, targets, batchsize, shuffle=False):
    assert len(pos) == len(targets)
    if shuffle:
        indices = np.arange(len(pos))
        np.random.shuffle(indices)
    for start_idx in range(0, len(pos) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield pos[excerpt], targets[excerpt]
        
        
def save_reconstruction(iteration, width, depth):
    start = time.time()
    print("saving to " + output_prefix + "_epoch_%04d_width_%03d_depth_%03d" % (iteration, width, depth)+".jpg")  
    reconstruct = lasagne.layers.get_output(layer_last, deterministic=True)
    reconstruct_fn = theano.function([positions_data], reconstruct)
    reconstructed = np.asarray(im).astype('float64')
    coords = np.zeros((reconstructed.shape[0]*reconstructed.shape[1],2), dtype=np.float32)
    
    for x in range(0, reconstructed.shape[0]):
        for y in range(0, reconstructed.shape[1]):
            coords[y*reconstructed.shape[0]+x] = np.array([
                           [float(x)/shape[0]],
                           [float(y)/shape[1]]
                      ]).T
    
   
    r = reconstruct_fn(coords)
    
    for x in range(0, reconstructed.shape[0]):
        for y in range(0, reconstructed.shape[1]):
            reconstructed[x,y,0:3] = r[y*reconstructed.shape[0]+x]

            if reconstructed[x,y,0] > 255:
                 reconstructed[x,y,0] = 255
            if reconstructed[x,y,1] > 255:
                 reconstructed[x,y,1] = 255
            if reconstructed[x,y,2] > 255:
                 reconstructed[x,y,2] = 255

    j = Image.fromarray(reconstructed.astype('uint8'))                                       
    j.save((output_prefix + "_epoch_%04d_width_%03d_depth_%03d" % (iteration, width, depth))+".jpg")
    
    
minibatchsize = 150

epochCounter = 0

for epoch in range(epochs):
    start = time.time()
    loss = 0
    loss_val = 0;
    for batch in iterate_minibatches(positions, rgb_values, minibatchsize, shuffle=True):
        pos, targets = batch
        loss += train_fn(pos, targets)
    losses.append(loss)
    print("epoch " + str(epochCounter) + " loss: " + str(loss/1000000) + " lr: " 
          + str(lr.eval()) 
          + " (" + "%.3f" % (time.time() - start) + "s)")
    
    #if epoch != 0 and epoch % 50 == 0:
    save_reconstruction(epochCounter, width, depth)
    epochCounter += 1
        
    if epoch != 0 and epoch % 25 == 0:
        lr = lr*0.1
 
