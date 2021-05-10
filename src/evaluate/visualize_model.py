import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def build_activation_model(mdl):
    layer_outputs = [layer.output for layer in mdl.layers]
    
    # Construct model outputing all layer activations
    return Model(inputs=mdl.inputs, outputs=layer_outputs)

def show_activations(img, activation_mdl, layer_num, size=5):

    # get activation outputs
    activation = activation_mdl.predict(img)[layer_num]

    n_features = activation.shape[-1]
    
    #colswitch = {1: 1, 2: 2, 3: 2, 4: 2}
    #ncols = colswitch.get(n_features, 3)
    ncols = 4
    nrows = 1 + (n_features - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size*ncols, size*nrows))
    axes = np.ravel(axes)

    fig.suptitle(f"Activations", fontsize = 20)
    
    for i in range(n_features):
        axes[i].imshow(activation[0, :, :, i], cmap='gray')
        
    for i in range(n_features, nrows*ncols):
        axes[i].axis('off')
        
def show_probs(img, activation_mdl):
    activation = activation_mdl.predict(img)[-1]
    n_features = activation.shape[-1]
    plt.bar([str(i) for i in range(n_features)], activation[0,:]**2)


def build_feature_extractor(mdl, layer_number):
    layer = mdl.layers[layer_number]
    feature_extractor = Model(inputs=mdl.inputs, outputs=layer.output)
    return feature_extractor

def compute_loss(input_image, filter_index, model):
    activation = model(input_image)
    filter_activation = activation[..., filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, model):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, model)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_config(cfg_shape):
    # We start from a random binary configuration
    #cfg = tf.random.uniform(cfg_shape, maxval=2, dtype=tf.int32)
    cfg = tf.random.uniform(cfg_shape, maxval=1, dtype=tf.float32)
    return tf.cast(cfg, dtype=tf.float32)

def visualize_filter(filter_index, cfg_shape, model):
    # run gradient ascent
    iterations = 1000
    learning_rate = 10.0
    img = initialize_config(cfg_shape)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, model)

    return loss, img[0].numpy()

# Compute image inputs that maximize per-filter activations
def show_filters(nfilt, cfg_shape, feature_extractor):
    all_imgs = [] 
    for filter_index in range(nfilt):
        loss, img = visualize_filter(filter_index, cfg_shape=cfg_shape, model=feature_extractor)
        all_imgs.append(img)
    
    ncols = 4
    nrows = (len(all_imgs) - 1) // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axs = axs.flatten()

    for i in range(len(all_imgs)):
        axs[i].matshow(all_imgs[i])
    
    for i in range(len(all_imgs), nrows*ncols):
        axs[i].axis('off')
