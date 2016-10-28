# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
# import theano

# import the neural net stuff
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

# import other stuff
import gym
import random
import numpy as np

from rl.utils.memory import Memory

defaultSettings = {
    'memorySize' : 100000,
    'discountFactor' : 0.975,
    'learningRate' : 0.00025,
    'hiddenLayers' : [30,30,30],
    'bias' : True
}

defaultRunSettings = {
    'updateTargetNetwork' : 10000,
    'explorationRate' : 1,
    'miniBatchSize' : 36,
    'learnStart' : 36,
    'renderPerXEpochs' : 1,
    'shouldRender' : True,
    'experimentId' : None,
    'force' : True,
    'upload' : False
}

class Advantage:
    def __init__(
            self, 
            env, 
            memorySize = defaultSettings['memorySize'], 
            discountFactor = defaultSettings['discountFactor'], 
            learningRate = defaultSettings['learningRate'], 
            hiddenLayers = defaultSettings['hiddenLayers'], 
            bias = defaultSettings['bias']):
        self.env = env
        self.size_observation = len(env.observation_space.high)
        self.size_action = env.action_space.n
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.hiddenLayers = hiddenLayers
        self.bias = bias
        self.initNetworks(hiddenLayers)

    def run(self, 
            epochs, 
            steps, 
            api_key,
            updateTargetNetwork = defaultRunSettings['updateTargetNetwork'], 
            explorationRate = defaultRunSettings['explorationRate'], 
            miniBatchSize = defaultRunSettings['miniBatchSize'], 
            learnStart = defaultRunSettings['learnStart'], 
            renderPerXEpochs = defaultRunSettings['renderPerXEpochs'], 
            shouldRender = defaultRunSettings['shouldRender'], 
            experimentId = defaultRunSettings['experimentId'], 
            force = defaultRunSettings['force'], 
            upload = defaultRunSettings['upload']):

        last100Scores = [0] * 100
        last100ScoresIndex = 0
        last100Filled = False

        if experimentId != None:
            self.env.monitor.start('tmp/'+experimentId, force = force)

        for epoch in xrange(epochs):
            path = {}
            path["actions"] = []
            path["rewards"] = []
            path["states"] = []
            path["values"] = []
            path["isDone"] = []

            observation = self.env.reset()
            # number of timesteps
            totalReward = 0
            for t in xrange(steps):
                if epoch % renderPerXEpochs == 0 and shouldRender:
                    self.env.render()

                policyValues = self.runModel(self.policyModel, observation)
                # print policyValues
                action = self.selectActionByProbability(policyValues, 1e-8)
                # print "action: ",action

                path["states"].append(observation)
                path["actions"].append(action)
                path["values"].append(self.runModel(self.valueModel, observation)[0])

                newObservation, reward, done, info = self.env.step(action)

                path["rewards"].append(reward)
                path["isDone"].append(done)

                totalReward += reward

                observation = newObservation

                if done:
                    last100Scores[last100ScoresIndex] = totalReward
                    last100ScoresIndex += 1
                    if last100ScoresIndex >= 100:
                        last100Filled = True
                        last100ScoresIndex = 0
                    if not last100Filled:
                        print "Episode ",epoch," finished after {} timesteps".format(t+1)," with total reward",totalReward
                    else :
                        print "Episode ",epoch," finished after {} timesteps".format(t+1)," with total reward",totalReward," last 100 average: ",(sum(last100Scores)/len(last100Scores))
                    break

            self.learn(path, observation)

        self.env.monitor.close()
        if upload:
            gym.upload('/tmp/'+experimentId, api_key=api_key)
   
    def initNetworks(self, hiddenLayers):
        self.valueModel = self.createRegularizedModel(self.size_observation, 1, hiddenLayers, "relu", "linear", "mse", 0.1 , self.bias)
        self.policyModel = self.createRegularizedModel(self.size_observation, self.size_action, hiddenLayers, "relu", "softmax", "mse", 0.0001, self.bias)

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, outputType, loss, learningRate, bias):
        dropout = 0
        regularizationFactor = 0
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(inputs,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(inputs,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(inputs,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)-1):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(outputs, init='lecun_uniform', bias=bias))
            model.add(Activation(outputType))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1

    def backupNetwork(self, model, backup):
        weights = model.get_weights()
        backup.set_weights(weights)

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def runModel(self, model, state):
        predicted = model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def learn(self, path, lastState):
        X_batch_value = np.empty((0,self.size_observation), dtype = np.float64)
        Y_batch_value = np.empty((0,1), dtype = np.float64)

        X_batch_policy = np.empty((0,self.size_observation), dtype = np.float64)
        Y_batch_policy = np.empty((0,self.size_action), dtype = np.float64)

        actions = path["actions"]
        rewards = path["rewards"]
        states = path["states"]
        values = path["values"]
        isDone = path["isDone"]

        print "expected value:",values[0]
        actions.reverse()
        rewards.reverse()
        states.reverse()
        values.reverse()
        isDone.reverse()

        path["returns"] = []
        path["advantage"] = []

        R = 0.0
        if isDone[0] == False:
            R = self.runModel(self.valueModel, lastState)
        for (ai, ri, si, vi, di) in zip (actions, rewards, states, values, isDone):
            R = ri + self.discountFactor * R
            td = R - vi
            a = np.zeros(self.size_action)
            a[ai] = 1
            path["returns"] = R
            path["advantage"] = td

            X_batch_value = np.append(X_batch_value, np.array([si]), axis=0)
            Y_batch_value = np.append(Y_batch_value, np.array([[R]]), axis=0)

            policy = self.runModel(self.policyModel, si)
            # print policy
            # print "action:",ai
            # print "td:",td
            change = min(1.0, td / vi)
            change_others = -((self.size_action -1) * change)
            policy = [x+change_others for x in policy]
            policy[ai] += change-change_others
            # print policy

            X_batch_policy = np.append(X_batch_policy, np.array([si]), axis=0)
            Y_batch_policy = np.append(Y_batch_policy, np.array([policy]), axis=0)

        self.valueModel.fit(X_batch_value, Y_batch_value, batch_size = len(path["actions"]), nb_epoch=1, verbose = 0)
        self.policyModel.fit(X_batch_policy, Y_batch_policy, batch_size = len(path["actions"]), nb_epoch=1, verbose = 0)




        
