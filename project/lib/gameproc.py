import keras
import numpy as np
import auxilary as aux
import os.path

#Class GameProcessor
class GameProcessor:


    def __init__(self, Inx, InY, trainingsize, learningrate, lossfunction, loadmodel = True):
        self.Inx = Inx
        self.Iny = InY
        self.trainingsize = trainingsize
        self.learnrate = learningrate

        self.target_model_available = False

        if(lossfunction == 'huber_loss'):
            self.loss_function = self.huber_loss
        elif(lossfunction == 'mse'):
            self.loss_function = 'mse'

        if loadmodel:
            if os.path.isfile('current_network'):
                print ('Network file found')
                self.model = keras.models.load_model('current_network')

                self.model.loss = self.loss_function
                loaded_optimizer = keras.optimizers.RMSprop(self.learnrate, rho=0.95, epsilon=0.01)
                self.model.compile(loaded_optimizer, loss=self.loss_function)

                self.model_loaded = True
            else:
                print('Network file not found')
        else:
            print('New model will be created')
            self.model_loaded = False

    #loss function
    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error*error  / 2
        linear_term = abs(error) - 1/2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            use_linear_term = keras.backend.cast(use_linear_term, 'float32')
        return use_linear_term*linear_term + (1-use_linear_term)*quadratic_term

    ###define the model###
    def game_model(self, n_actions):

        if self.model_loaded:
            return


        #defines the input layer size
        INPUT_LAYER_SHAPE = (self.trainingsize, self.Inx, self.Iny)

        #Create keras layers input as
        #State
        frame_input = keras.layers.Input(INPUT_LAYER_SHAPE, name='frameset')
        #And actions
        actions_input = keras.layers.Input((n_actions,), name='actionmask')

        #Add normalizing layer
        normalized = keras.layers.Lambda(lambda x: x/255.0)(frame_input)

        #Hidden layer #1
        # 16 filters, 8 size, (4,4) stride, relu
        conv_1 = keras.layers.Conv2D(16, 8, strides=(4,4), activation='relu')(normalized)

        # Hidden layer #2
        # 32 filters, 4 size, (2,2) stride, relu
        conv_2 = keras.layers.Conv2D(32, 4, strides=(2, 2), activation='relu')(conv_1)

        #Flatten
        flattened = keras.layers.Flatten()(conv_2)

        #fully connected
        fconnected = keras.layers.Dense(256, activation='relu')(flattened)

        #output
        #n_actions - number of action inputs
        output = keras.layers.Dense(n_actions)(fconnected)

        #multiply now the actions_input input layer with network output
        #merged_output = keras.layers.merge([output, actions_input], mode='mul')# Merge method is deprecated
        merged_output = keras.layers.multiply([output, actions_input])

        self.model = keras.models.Model([frame_input, actions_input], output=merged_output)
        optimizer = keras.optimizers.RMSprop(self.learnrate, rho=0.95, epsilon=0.01)

        self.model.compile(optimizer, loss=self.loss_function)
        #self.model.compile(optimizer, loss='mse')

        #initialize copy of the model for keeping target_model network
        self.target_network = self.model

    ###~define the model###

    #TBD: get the the best action for current state
    def max_action_rawvalue_on_model(self, state):
        #actions = self.model.predict(state, 4, steps=1)
        actions = self.model.predict(state, 1)
        return actions
    def max_action_vect_on_model(self,state):
        #actions = self.model.predict(state, 4, steps=1)
        actions = self.model.predict(state, 1)
        return aux.set_max_action_to_one(actions)

    #to be called as             fit_batch(gp.model, rewards, buffer, np_state_array_padded , action_vect)
    def fit_batch(self, rewards, gamma, start_states_buffer, next_state_formatted, actions_vect, is_game_over, save_model):

        #target_network - periodically copied and saved based in the self.model
        if save_model:
            self.target_network = self.clone_network()
            self.target_model_available = True

        #predict next Q values based on the most recent state
        ##CHANGE to target_model
        if self.target_model_available:
        #next_Q_vector = self.model.predict(next_state_formatted)
            next_Q_vector = self.target_network.predict(next_state_formatted)
        else:
            next_Q_vector = self.model.predict(next_state_formatted)

        #if game is terminated, set Q-vector to 0 and don't do any action
        if (is_game_over == True):
            next_Q_vector.fill(0)

        #calculate Q-scalar based on the max arguement of the predicted Q-vector
        Q_value = rewards + gamma*np.amax(next_Q_vector, axis=1)

        #Q_value = Q_value[:, None]

        target_action_vector = actions_vect*Q_value


        #temp = np.take(start_states_buffer, np.size(start_states_buffer,axis=0), axis=1)
        tempsize = np.size(start_states_buffer,axis=0)
        #NEW
        start_states_buffer = np.reshape(start_states_buffer, (tempsize, self.Inx, self.Iny))
        padded = [np.expand_dims(start_states_buffer, axis=0), actions_vect]


        #OLD
        #reshaped = np.reshape(start_states_buffer, (tempsize,105,80))
        #padded = [np.expand_dims(reshaped, axis=0), actions_vect]

        #fit model with the
        #padded
        self.model.fit(padded, target_action_vector, nb_epoch=1, batch_size=32, verbose=0)

    def clone_network(self):
        #self.model.save('current_network')
        self.model.save('current_network', overwrite=True, include_optimizer=False)
        #return keras.models.load_model('current_network', custom_objects={'loss', self.huber_loss})
        #return keras.models.load_model('current_network', custom_objects={'self.huber_loss', self.huber_loss})

        new_model = keras.models.load_model('current_network')
        new_model.loss = self.loss_function
        new_optimizer = keras.optimizers.RMSprop(self.learnrate, rho=0.95, epsilon=0.01)
        new_model.compile(new_optimizer, loss=self.loss_function)

        #saving-loaading of custom loss function doesn't work so far

        return new_model
