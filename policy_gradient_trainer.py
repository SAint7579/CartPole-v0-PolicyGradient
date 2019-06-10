import gym
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Declaring values for the network
num_inp = 4
num_hidden = 4
num_outputs = 1
lr = 0.01

#Creating the He initializer object
initializer = tf.contrib.layers.variance_scaling_initializer()
#Placeholder for the observation
X = tf.placeholder(tf.float32, shape= [None,num_inp], name= "OBSERVATIONS")

#Creating the neural network
hidden_layer = tf.layers.dense(X,num_hidden,activation = tf.nn.elu, kernel_initializer = initializer)
logits = tf.layers.dense(hidden_layer,num_outputs)
output = tf.nn.sigmoid(logits, name="OP_PROBABILITY")

#The output form the network's feed forward
prob = tf.concat(axis = 1, values=[output, 1-output])   
action = tf.multinomial(prob,num_samples = 1)           
#Creating the optimizer
y = 1.0 - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)
optimizer = tf.train.AdamOptimizer(lr)

#Extracting the gradient
gradient_and_vars = optimizer.compute_gradients(cross_entropy)


gradients = []              #Values to multiply with the discounted rewards
gradient_placeholders = []  #Placehoder for the multiplied gradients for the training opp
grads_and_vars_feed = []    #Array for applying gradients with the varaibles to the training opps


for grads,variable in gradient_and_vars:
    gradients.append(grads)
    gradient_placeholder = tf.placeholder(tf.float32,shape=grads.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder,variable))

#Creating the training tensor
training_op = optimizer.apply_gradients(grads_and_vars_feed)

#Initializer and saver functions
init = tf.global_variables_initializer()
saver = tf.train.Saver()
def helper_discount_rewards(rewards, discount_rate):
    '''
    Takes in rewards and applies discount rate
    '''
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Normalizes the rewards using Standard Scaling
    '''
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards,discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


#Creating the CartPole environment form the gym lib
env = gym.make('CartPole-v0')
game_rounds = 10
game_steps = 1000
num_iter = 1000
discount_rate = 0.95 #For policy gradients


with tf.Session() as sess:
    writer = tf.summary.FileWriter("./summary",sess.graph)
    sess.run(init)
    #For the training epochs
    for iteration in range(num_iter):
        print("On iteration ",iteration)
        #To store all gradients and rewards from the iteration
        all_rewards = []
        all_gradients = []
        for game in range(game_rounds):
            #To store gradient and rewards form one game instance
            current_rewards = []
            current_gradients = []
            observation = env.reset()

            for steps in range(game_steps):
                #To get the action and gradient values
                action_val, gradient_val = sess.run([action,gradients],feed_dict = {X : observation.reshape(1,num_inp)})
                #Executing the action
                observations, reward, done, info = env.step(action_val[0][0])  #Format of multinomial's output

                #Updating current rewards and gradients
                current_rewards.append(reward)
                current_gradients.append(gradient_val)
                if done:
                    break
            #Updating all rewards and gradients
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        #Discounting and normalizing all the rewards
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}

        #Multiplying the discounted rewards with the gradients
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            #Creating the feed dictionary of gradients for training the agent 
            feed_dict[gradient_placeholder] = mean_gradients

        #Training the model's weights
        sess.run(training_op , feed_dict = feed_dict)
        print("SAVING THE GRAPHS AND THE SESSION")

        #Saving the graph and model
        meta_graph_def = tf.train.export_meta_graph("./models/policy-model.meta")
        saver.save(sess,"./models/policy-model")
    writer.close()


#RUNNING THE TRAINED MODEL

env = gym.make('CartPole-v0')

observations = env.reset()
with tf.Session() as sess:
    #Loading the model
    new_saver = tf.train.import_meta_graph("./models/policy-model.meta")
    new_saver.restore(sess,"./models/policy-model")
    #Executing the model for 500 steps
    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: observations.reshape(1, num_inp)})
        observations, reward, done, info = env.step(action_val[0][0])
        if done:
            break
        
                




    
