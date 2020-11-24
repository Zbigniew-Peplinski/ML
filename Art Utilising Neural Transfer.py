# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:11:19 2020

@author: z
"""

import scipy.io
import scipy.misc
import tensorflow as tf

content_image = scipy.misc.imread("images/louvre.jpg")
style_image = scipy.misc.imread("images/monet_800600.jpg")

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.transpose(a_C)
    a_G_unrolled = tf.transpose(a_G)
    
    # compute the cost with tensorflow (≈1 line)
    J_content = (1/ (4* n_H * n_W * n_C)) * tf.reduce_sum(tf.pow((a_G_unrolled - a_C_unrolled), 2))
    ### END CODE HERE ###
    
    return J_content

def gram_matrix(A):
    """
    A -- matrix of shape (n_C, n_H*n_W)
    
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A, tf.transpose(A))
    ### END CODE HERE ###
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_H*n_W, n_C) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1./(4 * n_C**2 * (n_H*n_W)**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))
    
    ### END CODE HERE ###
    
    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    J -- total cost as defined by the formula below
    """
    
    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style
    ### END CODE HERE ###
    
    return J

i = 0
"""
Prepares all the daata and initializes tf
"""
while i<1:  
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    content_image = scipy.misc.imread("images/louvre_small.jpg")
    content_image = reshape_and_normalize_image(content_image)
    style_image = scipy.misc.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image)
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2'] 
    a_C = sess.run(out) 
    a_G = out
    J_content = compute_content_cost(a_C, a_G) 
    sess.run(model['input'].assign(style_image))
    
    J_style = compute_style_cost(model, STYLE_LAYERS)
    J = total_cost(J_content, J_style, alpha = 10, beta = 40)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    i += 1


def model_nn(sess, input_image, num_iterations = 200):

    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):

        sess.run(train_step)
        generated_image = sess.run(model['input'])

        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

model_nn(sess, generated_image)
