# =============
# Critic DNN
# =============

import tensorflow as tf

# network parameters: hidden layers
n_hidden_1 = 400
n_hidden_2 = 300

# weight parameter init
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# bias weight
def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)

# Critic DNN network
class CriticNetwork(object):
    """
    Input: state, action
    Output: Q(s, a) Q-value
    The action must be obtained from the ouput of the Actor Network        
    """
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]
        
        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                     tf.multiply(self.target_network_params[i], 1. - self.tau))\
              for i in range(len(self.target_network_params))]
        
        # network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predicted_q_value, self.out))))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Input -> Hidden Layer
        w1 = weight_variable([self.s_dim, n_hidden_1])
        b1 = bias_variable([n_hidden_1])

        # Hidden Layer -> Hidden Layer + Action
        w2 = weight_variable([n_hidden_1, n_hidden_2])
        w2a = weight_variable([self.a_dim, n_hidden_2])
        b2 = bias_variable([n_hidden_2])
        
        # Hidden Layer -> Output Q-value
        w3 = weight_variable([n_hidden_2, 1])
        b3 = bias_variable([1])

        
        # 1st hidden layer, option: Softmax, ReLU, tanh or sigmoid
        h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
         
        # 2nd hidden layer, option: Softmax, ReLU, tanh or sigmoid
        # Action inserted here.
        h2 = tf.nn.relu(tf.matmul(h1, w2) + tf.matmul(action, w2a) + b2)
        
        out = tf.matmul(h2, w3) + b3

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action    
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action    
        })
        
    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })        
        
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
