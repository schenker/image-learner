# Learning to reconstruct an image from pixel coordinates
Train a neural network to map from x and y coordinate of each pixel to the pixels r,g,b values.
The network has 2 inputs (x and y), several fully connected (highway) layers and three outputs (r, g and b).

See implementation in [image-learner.ipynb](image-learner.ipynb)

## Examples

### Input:
![example1](/images/monalisa.jpg?raw=true)
### Sample Outputs:
#### after 12 epochs with a network of width 6 and depth 35
![example1](/images/monalisa_epoch_0012_width_006_depth_035.jpg?raw=true)
#### after 520 epochs with a network of width 150 and depth 5
![example1](/images/monalisa_epoch_0520_width_150_depth_005.jpg?raw=true)

### Input:
![example1](/images/stifte.jpg?raw=true)
### Sample Outputs:
#### after 4 epochs with a network of width 6 and depth 25
![example1](/images/stifte_epoch_0004_width_006_depth_025.jpg?raw=true)
#### after 14 epochs with a network of width 6 and depth 25
![example1](/images/stifte_epoch_0014_width_006_depth_025.jpg?raw=true)

### Input:
![example1](/images/tuebingen.jpg?raw=true)
### Sample Outputs:
#### after 2 epochs with a network of width 50 and depth 20
![example1](/images/tuebingen_epoch_0002_width_050_depth_020.jpg?raw=true)
#### after 142 epochs with a network of width 50 and depth 20
![example1](/images/tuebingen_epoch_0142_width_050_depth_020.jpg?raw=true)

### Input:
![example1](/images/bunt.jpg?raw=true)
### Sample Outputs:
#### after 7 epochs with a network of width 6 and depth 25
![example1](/images/bunt_epoch_0007_width_006_depth_025.jpg?raw=true))
#### after 45 epochs with a network of width 6 and depth 25
![example1](/images/bunt_epoch_0045_width_006_depth_025.jpg?raw=true)

### Input:
![example1](/images/fluorescence.jpg?raw=true)
### Sample Outputs:
#### after 8 epochs with a network of width 70 and depth 5
![example1](/images/fluorescence_epoch_0008_width_070_depth_005.jpg?raw=true)
#### after 75 epochs with a network of width 70 and depth 5
![example1](/images/fluorescence_epoch_0075_width_070_depth_005.jpg?raw=true)
