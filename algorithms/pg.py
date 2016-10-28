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

class PG:
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
        self.initNetworks()

    def run(self, 
            epochs,  
            steps, 
            api_key,
            rollouts_per_epoch = 100,
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
            paths = []
            for rollout in xrange(rollouts_per_epoch):
                path = {}
                path["actions"] = []
                path["rewards"] = []
                path["states"] = []
                path["isDone"] = []

                observation = self.env.reset()
                # number of timesteps
                totalReward = 0
                for t in xrange(steps):
                    policyValues = self.runModel(self.policyModel, observation)
                    action = self.selectActionByProbability(policyValues)
                    # action = self.selectActionByProbability(self.convertToProbabilities(policyValues))

                    path["states"].append(observation)
                    path["actions"].append(action)

                    newObservation, reward, done, info = self.env.step(action)

                    path["rewards"].append(reward)
                    path["isDone"].append(done)

                    totalReward += reward

                    observation = newObservation

                    if done:
                        break
                paths.append(path)

            self.learn(paths)

        self.env.monitor.close()
        if upload:
            gym.upload('/tmp/'+experimentId, api_key=api_key)

    def convertToProbabilities(self, policyValues):
        print policyValues
        probabilities = [self.sigmoid(x) for x in policyValues ]
        sumProbabilities = sum(probabilities)
        probabilities = [ x / sumProbabilities for x in probabilities ]
        print probabilities
        return probabilities

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
   
    def initNetworks(self):
        self.policyModel = self.createRegularizedModel(self.size_observation, self.size_action, self.hiddenLayers, "relu", "softmax", "mse", self.learningRate, self.bias)

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

    def selectActionByProbability(self, probabilities):
        sumprob = sum(probabilities)
        probabilities = [x / sumprob for x in probabilities]

        r = random.random()
        index = 0
        while(r >= 0 and index < len(probabilities)):
          r -= probabilities[index]
          index += 1
        return index - 1

    def learn(self, paths):
        batch_size = 0
        bestScore = -1e8

        totalRewards = []
        for path in paths:
            pathReward = sum(path["rewards"])
            bestScore = max(bestScore, pathReward)
            batch_size += len(path["rewards"])
            path["totalReward"] = pathReward
            totalRewards.append(pathReward)

        std = np.std(totalRewards)
        avg = sum(totalRewards) / len(totalRewards)

        decreaseProbability = 1.0 / (self.size_action - 1)

        # normalizationFactor = 1e-8
        # for path in paths:
            # normalizationFactor = max(normalizationFactor, abs(path["totalReward"] - baselineScore))

        baselineScore = avg
        print "Average score:",avg
        print "Best score:",bestScore

        X_batch_policy = np.empty((0,self.size_observation), dtype = np.float64)
        Y_batch_policy = np.empty((0,self.size_action), dtype = np.float64)

        for path in paths:
            # performance = (path["totalReward"] - baselineScore) / normalizationFactor
            # print "reward",path["totalReward"]
            performance = ((path["totalReward"] - avg) / std)
            # print "performance",performance

            if performance != 0:
                for (action, state) in zip (path["actions"], path["states"]):
                    actionVector = self.runModel(self.policyModel, state)
                    actionVector[action] += performance
                    # if performance > 0:
                    #     actionVector[action] += 1
                    # else:
                    #     actionVector[action] -= 1 
                        
                    X_batch_policy = np.append(X_batch_policy, np.array([state]), axis=0)
                    Y_batch_policy = np.append(Y_batch_policy, np.array([actionVector]), axis=0)

        self.policyModel.fit(X_batch_policy, Y_batch_policy, nb_epoch=1, verbose = 0)

        
    