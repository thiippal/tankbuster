from tankbuster.cnn import CNNArchitecture
import h5py

arch = "MiniVGGNetFC"  # Network architecture
model = CNNArchitecture.select(arch, None, None, 3, 3)

# print model.summary()

for layer in model.layers:  # Loop over layers
    if layer.name.startswith("conv"):  # Get all convolutional layers
        orig = model.get_layer(layer.name)  # Get original layer
        layer.set_weights(orig.get_weights())  # Retain original weights
    if layer.name.startswith("dense"):  # Get dense layers
        orig = model.get_layer(layer.name)
        W, b = orig.get_weights()
        n_filter, previous_filter, ax1, ax2 = layer.get_weights()[0].shape
        new_W = W.reshape((previous_filter, ax1, ax2, n_filter))
        new_W = new_W.transpose((3, 0, 1, 2))
        new_W = new_W[:, :, ::-1, ::-1]
        layer.set_weights([new_W, b])

model.load_weights('tankbuster/engine/weights.h5')

# Load weights manually?
# wf = h5py.File('tankbuster/engine/weights.h5', 'r')

# weights = {}

# for layer, group in wf.items():
  #  for parameter_name in group.keys():
   #     if 'W' in parameter_name:
    #        weights[layer] = group[parameter_name]








