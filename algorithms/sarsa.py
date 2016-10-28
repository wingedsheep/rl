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
    'discountFactor' : 0.99,
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

class Sarsa:
    def __init__(
            self, 
            env,
            input_size = None,
            output_size = None,
            memorySize = defaultSettings['memorySize'], 
            discountFactor = defaultSettings['discountFactor'], 
            learningRate = defaultSettings['learningRate'], 
            hiddenLayers = defaultSettings['hiddenLayers'], 
            bias = defaultSettings['bias']):
        self.env = env
        if input_size == None:
        	self.input_size = len(env.observation_space.high)
        else :
        	self.input_size = input_size
        if output_size == None:
        	self.output_size = env.action_space.n
        else :
        	self.output_size = output_size
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.hiddenLayers = hiddenLayers
        self.bias = bias
        self.initNetworks(hiddenLayers)

    def run(self, 
            epochs, 
            steps, 
            api_key,
            monitor = True,
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

        stepCounter = 0

        if experimentId != None and monitor:
            self.env.monitor.start('tmp/'+experimentId, force = force)

        for epoch in xrange(epochs):
            observation = self.env.reset()

            qValues = self.getQValues(observation)
            action = self.selectAction(qValues, explorationRate)

            print explorationRate
            # number of timesteps
            totalReward = 0
            for t in xrange(steps):
                if epoch % renderPerXEpochs == 0 and shouldRender:
                    self.env.render()

                newObservation, reward, done, info = self.env.step(action)
                
                qValues = self.getQValues(observation)
                newAction = self.selectAction(qValues, explorationRate)

                totalReward += reward

                self.addMemory(observation, action, reward, newObservation, newAction, done)

                if stepCounter >= learnStart:
                    if stepCounter <= updateTargetNetwork:
                        self.learnOnMiniBatch(miniBatchSize, False)
                    else :
                        self.learnOnMiniBatch(miniBatchSize, True)

                observation = newObservation.copy()
                action = newAction

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

                stepCounter += 1
                if stepCounter % updateTargetNetwork == 0:
                    self.updateTargetNetwork()
                    print "updating target network"

            explorationRate *= 0.995
            # explorationRate -= (2.0/epochs)
            explorationRate = max (0.05, explorationRate)

        self.env.monitor.close()
        if upload:
            gym.upload('/tmp/'+experimentId, api_key=api_key)
   
    def initNetworks(self, hiddenLayers):
        self.model = self.createRegularizedModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate, self.bias)
        self.targetModel = self.createRegularizedModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate, self.bias)

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate, bias):
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
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
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
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, newAction, reward, isFinal):
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * qValuesNewState[newAction]

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

    def addMemory(self, state, action, reward, newState, newAction, isFinal):
        self.memory.addMemoryWithNewAction(state, action, reward, newState, newAction, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        miniBatch = self.memory.getMiniBatch(miniBatchSize)
        X_batch = np.empty((0,self.input_size), dtype = np.float64)
        Y_batch = np.empty((0,self.output_size), dtype = np.float64)
        for sample in miniBatch:
            isFinal = sample['isFinal']
            state = sample['state'].copy()
            action = sample['action']
            reward = sample['reward']
            newState = sample['newState'].copy()
            newAction = sample['newAction']

            qValues = self.getQValues(state)
            if useTargetNetwork:
                qValuesNewState = self.getTargetQValues(newState)
            else :
                qValuesNewState = self.getQValues(newState)
            targetValue = self.calculateTarget(qValuesNewState, newAction, reward, isFinal)

            X_batch = np.append(X_batch, np.array([state]), axis=0)
            Y_sample = qValues
            Y_sample[action] = targetValue
            Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            if isFinal:
            	X_batch = np.append(X_batch, np.array([newState]), axis=0)
            	Y_batch = np.append(Y_batch, np.array([[0] * self.output_size]), axis=0)
        self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)
