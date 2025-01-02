import pennylane as qml


from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

n_epochs = 30   # Number of optimization epochs
n_layers = 5    # Number of random layers
n_train = 50    # Size of the train dataset
n_test = 50     # Size of the test dataset

SAVE_PATH = "/content/"  # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce dataset size
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# Normalize pixel values within 0 and 1
train_images = train_images / 255
test_images = test_images / 255

# Add extra dimension for convolution channels
train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

dev1 = qml.device("lightning.qubit", wires = 4)
dev2 = qml.device("default.qubit", wires = 4)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

print(rand_params)
@qml.qnode(dev1)
def circuit1(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)), seed = None)


    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]


# qml.draw_mpl(circuit1, level=None, decimal = 1)([1,2,3,4])
# qml.draw_mpl(circuit1, level=None, decimal = 1)([2,3,4,5])

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for i in range(4):
      cir = circuit1
      for j in range(0, 28, 2):
          for k in range(0, 28, 2):
              # Process a squared 2x2 region of the image with a quantum circuit
              q_results = cir(
                  [
                      image[j, k, 0],
                      image[j, k + 1, 0],
                      image[j + 1, k, 0],
                      image[j + 1, k + 1, 0]
                  ]
              )
              # Assign expectation values to different channels of the output pixel (j/2, k/2)

                  # out[j // 2, k // 2, c] = q_results[c]
              for c in range(4):
                out[j // 2, k // 2, i] += q_results[c]
              # out[j // 2, k // 2, i] /= 4
    return out

if PREPROCESS == True:
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print("\r{}/{}".format(idx + 1, n_train), end = '')
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images):
        print("\r{}/{}".format(idx + 1, n_test), end = '')
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)

    # # Save pre-processed images
    # np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    # np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


# # Load pre-processed images
# q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
# q_test_images = np.load(SAVE_PATH + "q_test_images.npy")

n_samples = 4
n_channels = 4
fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
for k in range(n_samples):
    axes[0, 0].set_ylabel("Input")
    if k != 0:
        axes[0, k].yaxis.set_visible(False)
    axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

    # Plot all output channels
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
        if k != 0:
            axes[c, k].yaxis.set_visible(False)
        axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

plt.tight_layout()
plt.show()