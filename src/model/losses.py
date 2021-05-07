import tensorflow as tf

def trace_loss(r = 1e-3):

    def inner_trace_loss(y_true, y_pred):
        """
        Calculate the ratio of between and within-class covariance traces
        """
        
        # Split data into groups based on their classes
        group_ids, group_idxs = tf.unique(tf.reshape(y_true, (-1, )))
        
        # Function to calculate a within-class covariance matrix on a spehre
        # for a given group_id
        def fn(group_id):
            g_idxs = tf.where(tf.equal(group_id, group_idxs))
            g_idxs = tf.reshape(g_idxs, (-1, ))
            X = tf.gather(y_pred, g_idxs)
            #X_mean = X - tf.reduce_mean(X, axis=0)
            X_mean = tf.math.acos(X*tf.reduce_mean(X, axis=0))
            m = tf.cast(tf.shape(X_mean)[1], tf.float32)
            return (1 / (m - 1)) * tf.matmul(tf.transpose(X_mean), X_mean)

        # Get covariance matrices for each group (within-class)
        covs_t = tf.map_fn(lambda gid: fn(gid), group_ids, dtype=tf.float32)
        
        # compute within scatters
        Sw = tf.reduce_sum(covs_t, axis=0) 

        # compute total scatter
        Xt_bar = tf.math.acos(y_pred*tf.reduce_mean(y_pred, axis=0))
        m = tf.cast(tf.shape(Xt_bar)[1], tf.float32)
        St = (1 / (m - 1)) * tf.matmul(tf.transpose(Xt_bar), Xt_bar)

        # compute between scatter
        Sb = St - Sw

        trSx = tf.map_fn(lambda c: tf.linalg.trace(c), covs_t, dtype=tf.float32)
        trSb = tf.linalg.trace(Sb)
        trSw = (tf.reduce_mean(trSx) + r)/tf.cast(len(covs_t), dtype=tf.float32)
        
        return -tf.math.divide(trSb, trSw)

    return inner_trace_loss

def variance_loss(r = 1e-3):

    def inner_variance_loss(y_true, y_pred):
        """
        Minimize the ratio of between- and within-class variances
        """
        
        # Split data into groups based on their classes
        group_ids, group_idxs = tf.unique(tf.reshape(y_true, (-1, )))
        
        # Function to calculate a within-cluster variance on a sphere
        def fn(group_id):
            g_idxs = tf.where(tf.equal(group_id, group_idxs))
            g_idxs = tf.reshape(g_idxs, (-1, ))
            X = tf.gather(y_pred, g_idxs)
            X_mean = tf.reduce_mean(X, axis=0)
            cb = tf.tensordot(X, X_mean, axes=[[1], [0]])
            s2 = tf.math.square(tf.math.acos(cb))
            return tf.reduce_sum(s2)

        # Calculate within-cluster sum of squares for each group
        Vws = tf.map_fn(lambda gid: fn(gid), group_ids, dtype=tf.float32)
        Vw = tf.reduce_sum(Vws + r)
        
        # Calculate total sum of squares
        X_mean = tf.reduce_mean(y_pred, axis=0)
        cb = tf.tensordot(y_pred, X_mean, axes=[[1], [0]])
        s2 = tf.math.square(tf.math.acos(cb))
        Vt = tf.reduce_sum(s2)

        Vb = Vt - Vw
        return -Vb
        #return -tf.math.divide((Vt - Vw), Vw)
        #return -(Vt - Vw)

    return inner_variance_loss
