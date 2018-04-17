import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import random

from lib import auxilary as aux
from lib import buffer as buf
from lib import gameproc as gmp
from lib import myargparser as marg

#number of games to play
gl_n_games = 10000
#globalnumber of iterations
numberOfIterations = 1
#completed game flag
gl_is_done = False
#get batch os necessary size by batch_size element
batch_index = 0
#batch size
gl_batch_size_short = 1
gl_batch_size_normal = 8
#reward
gl_reward = 0
#gamma
gl_gamma = 0.99
#learning rate
gl_learning_rate = 0.01

#if no negative rewards expected use following at gl_is_donw=True
#TODO: consider the filter for negative reward
gl_neg_reward = -10


def model_iteration(env, model, trainingElement, buffer, n_actions):
    global gl_is_done
    global gl_reward
    global gl_gamma
    global numberOfIterations
    global gl_batch_size_short
    global gl_batch_size_normal
    global batch_index
    global action_sample

    do_fit_and_sample = False
    save_model_flag = False

    #initialize action vector randomly once at start-up
    if (numberOfIterations < 4):
        action_sample = env.action_space.sample()

    another_frame, reward, gl_is_done, _ = env.step(action_sample)
    gl_reward = aux.transform_reward(reward)

    if(gl_is_done == True):
        gl_reward = gl_neg_reward

    print('Game reward immediate ' + str(gl_reward))

    #resize the image to predefined dimensions of the network input
    another_frame = aux.grayscale_img(another_frame)
    another_frame = aux.downsample_img(another_frame)

    #add new frame to the last trainingElemenet which is a current state
    trainingElement.training_set_new(another_frame)
    #append new elements to the global ring buffer
    #buffer.append([another_frame, action_sample, reward, gl_is_done])
    buffer.append(trainingElement.training_set)

    #get epsilon
    epsi = aux.epsilon_get_linear(numberOfIterations, 100000)


    if (random.random() < epsi and numberOfIterations % 4 != 0 ):#get a new random action from time to time
        action_sample = env.action_space.sample()
    #it is import to fit the batch every 4 iterations. Therefore no "elif" here

    if numberOfIterations % 4 == 0:
        do_fit_and_sample = True
        batch_size = gl_batch_size_normal

    '''
        if (numberOfIterations < 32 ):
            if(numberOfIterations % 4 == 0):#make an action on current state based on current model for each 4 new frames
                do_fit_and_sample = True
                batch_size = gl_batch_size_short
        else:
            if(numberOfIterations % 32 == 0):#make an action on current state based on current model when we exceed 32 samples with batches of 32

                do_fit_and_sample = True
                batch_size = gl_batch_size_normal
    '''
    if (numberOfIterations % 10000 == 0):
        save_model_flag = True

    if (do_fit_and_sample == True):
        np_state_array = np.array(trainingElement.get_training_set_wUpdate())#get the temporary array of elements as numpy array. Flush the initial trainingElemnt array with wUpdate

        #array aligned with model inputs
        np_state_array_padded = [np.expand_dims(np_state_array, axis=0), np.expand_dims(np.ones(n_actions),axis=0)]

        #pad the elements to get (None, x, y, z) and (None, k) format
        #action_vect = model.max_action_vect_on_model( np_state_array_padded )
#CHANGE
        action_vect = model.max_action_rawvalue_on_model(np_state_array_padded)

        #fit_batch(gp.model, rewards, buffer, np_state_array_padded , action_vect)
        #fit_batch(self, rewards, gamma, start_states_buffer, next_state_formatted, actions_vect, is_game_over)
        '''
        gp.fit_batch(gl_reward, gl_gamma, buffer.get_seq_of_items(numberOfIterations-batch_size, batch_size), np_state_array_padded, action_vect, gl_is_done)
        '''


        gp.fit_batch(gl_reward, gl_gamma, buffer.getitem(numberOfIterations-1),
                     np_state_array_padded, action_vect, gl_is_done, save_model=save_model_flag)

#CHANGE
        action_vect = aux.set_max_action_to_one(action_vect)

        action_sample = np.argmax(action_vect,axis=1)#get index of action

    #if (numberOfIterations % 10000 == 0):
     #   gp.clone_network()

    return reward

#parser
arg_parser = marg.InputArguments()
args = arg_parser.parser.parse_args()

#check inputs
if (args.lr <= 0.05 and args.lr >= 0.00025):
    gl_learning_rate = args.lr
if (args.ngames <= 1000000 and args.ngames > 1):
    gl_n_games = args.ngames
if args.load == True:
    arg_loadmodel = True
else:
    arg_loadmodel = False

#processing start
#env = gym.make('BreakoutDeterministic-v0')
env = gym.make('SpaceInvaders-v0')
#env = gym.make('BattleZone-v0')
frame = env.reset()  # type: object
env.render()


#grayscale the image
frame = aux.grayscale_img(frame)
frame = aux.downsample_img(frame)


#Show image with a 'frame' array
#imgplot = plt.imshow(frame)
#plt.show()

n_actions = 4
n_games_played = 0
print('Initialize input action vector size..' + str(n_actions))

#initialize the model
print('Initialize input data sizes..' + str(np.size(frame)))
gp = gmp.GameProcessor(np.size(frame, axis=0),
                       np.size(frame, axis=1),
                       4, gl_learning_rate,
                       'huber_loss',#can use 'mse' as loss function
                       arg_loadmodel)
print('Initialize game model..')
gp.game_model(n_actions)#16 actions


#init state_Frames
print('Initialize state_frames..')
state_frames = buf.trainingElement(frame)
#init RingBuffer
print('Initialize ring buffer..')
ring_buffer = buf.RingBuffer(32^5)

print('Iterating the model..')
#while not gl_is_done:
while not n_games_played == gl_n_games:
    print('Iteration '+str(numberOfIterations))
    game_reward_state = model_iteration(env,gp, state_frames, ring_buffer, n_actions)
    print('Game reward '+str(game_reward_state))
    #increment the global number of Iterations
    numberOfIterations = numberOfIterations + 1
    env.render()
    #time.sleep(0.10)

    if gl_is_done:
        n_games_played += 1
        gl_is_done = False
        env.reset()

print('Games played '+str(n_games_played))
'''
while not gl_is_done:
    frame, reward, gl_is_done, _ = env.step(env.action_space.sample())

    if(state_frames.training_set_ready == False):
        state_frames.training_set_new(frame)
    else:
        curTrainingSet = state_frames.get_training_set()
        imgplot = plt.imshow(curTrainingSet[0])
        plt.show()

    env.render()
    gl_is_done = False
    time.sleep(0.10)
'''
env.close()

