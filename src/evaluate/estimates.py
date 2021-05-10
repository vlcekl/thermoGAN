import numpy as np
import tensorflow as tf

def nll_multinom(y, neff, pr):
    """Calculate multinomial loglikelihood"""
    
    nll  = tf.reduce_sum(neff * y * tf.math.log(pr), axis = -1)
    nll += tf.math.lgamma(neff + 1)
    nll -= tf.reduce_sum(tf.math.lgamma(neff * y + 1))
    nll += neff * (1 - tf.reduce_sum(pr))

    return -tf.reduce_mean(nll)

def optimize_multinom(y_true, epochs = 10):
    """Find parameters of a multinomial distribution maximizing likelhood"""
    
    learning_rate = 1e-4
    # Collect the history of parameters
    neffs, prs = [], []
    
    neff = tf.Variable(10.0, name='n_eff', dtype=tf.float32)
    p_mean = np.mean(y_true, axis=0)
    pr = tf.constant(p_mean, name='probs', dtype=tf.float32)

    for epoch in range(epochs):
        
        with tf.GradientTape() as t:
            mll = nll_multinom(y_true, neff, pr)

        # Use GradientTape to calculate the gradients with respect to W and b
        dneff, dpr = t.gradient(mll, [neff, pr])
        
        #dneff = tf.math.l2_normalize(dneff)

        # Subtract the gradient scaled by the learning rate
        neff.assign_sub(learning_rate * dneff)
        #pr.assign_sub(learning_rate * dpr)
        
        neffs.append(neff.numpy())
        prs.append(pr.numpy())

        current_loss = nll_multinom(y_true, neff, pr)

    #print(current_loss.numpy(), neffs[-1], prs[-1], np.sum(prs[-1]))

    return neffs, prs

def get_estimates(data, labels, epochs=20):
    params = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)
        dd = data[idx[0], :]**2  # probabilities
        #print("class: ", c)
        neffs, prs = optimize_multinom(dd, epochs=epochs)
        params[c] = {'neff': neffs, 'pr': prs}  

    return params
