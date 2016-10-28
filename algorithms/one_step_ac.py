# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano

# import the neural net stuff
from keras.models import Sequential, Model
from keras import optimizers
import keras
from keras import backend as K
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
    'learningRatePolicy' : 0.00025,
    'learningRateQValue' : 0.00025,
    'hiddenLayersPolicy' : [30,30,30],
    'hiddenLayersQValue' : [30,30,30],
    'memorySize': 100000,
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

class OneStepAC:
    def __init__(
            self, 
            env, 
            input_size = None,
            output_size = None,
            memorySize = defaultSettings['memorySize'], 
            discountFactor = defaultSettings['discountFactor'], 
            learningRatePolicy = defaultSettings['learningRatePolicy'], 
            learningRateQValue = defaultSettings['learningRateQValue'], 
            hiddenLayersPolicy = defaultSettings['hiddenLayersPolicy'], 
            hiddenLayersQValue = defaultSettings['hiddenLayersQValue'], 
            bias = defaultSettings['bias']):
        self.env = env
        if input_size == None:
            self.size_observation = len(env.observation_space.high)
        else :
            self.size_observation = input_size
        if output_size == None:
            self.size_action = env.action_space.n
        else :
            self.size_action = output_size
        self.discountFactor = discountFactor
        self.learningRatePolicy = learningRatePolicy
        self.learningRateQValue = learningRateQValue
        self.hiddenLayersPolicy = hiddenLayersPolicy
        self.hiddenLayersQValue = hiddenLayersQValue
        self.bias = bias
        self.memory = Memory(memorySize)
        self.initNetworks()

    def run(self, 
            epochs,  
            steps, 
            api_key,
            rollouts_per_epoch = 20,
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

        if not experimentId == None:
            self.env.monitor.start('tmp/'+experimentId, force = force)

        for epoch in xrange(epochs):
            I = 1
            observation = self.env.reset();
            for t in xrange(steps):
                policyValues = self.runModel(self.policyModel, observation)
                action = self.selectActionByProbability(policyValues)

                newObservation, reward, done, info = self.env.step(action)

                cost, grads = self.get_cost_grads(self.policyModel);
                print (theano.pp(grads[1][0]));

                if done:
                    delta = reward + self.discountFactor * self.runModel(self.valueModel, newObservation) - self.runModel(self.valueModel, observation)
                else :
                    delta = reward - self.runModel(self.valueModel, observation) # because the value for new obs is 0

        self.env.monitor.close()
        if upload:
            gym.upload('/tmp/'+experimentId, api_key=api_key)

    def get_gradients(self, model):
        """Return the gradient of every trainable weight in model

        Parameters
        -----------
        model : a keras model instance

        First, find all tensors which are trainable in the model. Surprisingly,
        `model.trainable_weights` will return tensors for which
        trainable=False has been set on their layer (last time I checked), hence the extra check.
        Next, get the gradients of the loss with respect to the weights.

        """
        weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
        optimizer = model.optimizer

        return optimizer.get_gradients(model.total_loss, weights)

    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    def get_trainable_params(self, model):
        params = []
        for layer in model.layers:
            params += keras.engine.training.collect_trainable_weights(layer)
        return params

    def get_cost_grads(self, model):
        """ Returns the cost and flattened gradients for the model """
        trainable_params = self.get_trainable_params(model)

        cost = model.model.total_loss
        grads = K.gradients(cost, trainable_params)

        return cost, grads

    def flatten_grads(self, grads):
        """ Flattens a set tensor variables (gradients) """
        x = np.empty(0)
        for g in grads:
            x = np.concatenate((x, g.reshape(-1)))

    def convertToProbabilities(self, policyValues):
        probabilities = [self.sigmoid(x) for x in policyValues ]
        sumProbabilities = sum(probabilities)
        probabilities = [ x / sumProbabilities for x in probabilities ]
        return probabilities

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
   
    def initNetworks(self):
        self.policyModel = self.createRegularizedModel(self.size_observation, self.size_action, self.hiddenLayersPolicy, "relu", "softmax", "mse", self.learningRatePolicy, self.bias)
        self.valueModel = self.createRegularizedModel(self.size_observation, 1, self.hiddenLayersQValue, "relu", "linear", "mse", self.learningRateQValue, self.bias)

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

    def getValue(self, state):
        return self.getMaxQ(self.runModel(self.qValueModel, state))

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

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

    def processPath(self, path):
        returns = []
        expectations = []
        advantages = []
        td_errors = []

        pathIsFinished = path["isDone"][len(path["isDone"]) -1]
        if not pathIsFinished:
            lastState = path["newStates"][len(path["newStates"]) -1]
            R = self.getMaxQ(self.runModel(self.qValueModel,lastState))
        else:
            R = 0     

        for reward, state, newState in zip(reversed(path["rewards"]), reversed(path["states"]), reversed(path["newStates"])):
            expectation = self.getValue(state)
            newExpectation = reward + self.discountFactor * self.getValue(newState)
            expectations.append(expectation)
            R = reward + self.discountFactor * R
            returns.append(R)
            advantages.append(R - expectation)
            td_error = newExpectation - expectation
            td_errors.append(td_error)

        returns.reverse()
        expectations.reverse()
        advantages.reverse()
        td_errors.reverse()

        path["returns"] = returns
        path["advantages"] = advantages
        path["expectations"] = expectations
        path["td_errors"] = td_errors

        # print map(int, path["expectations"])

        # print "returns: ",path["returns"]
        # print "expectations: ",path["expectations"]
        # print "advantages: ",path["advantages"]
        return path

    def learnPolicy(self, paths):
        totalRewards = []
        allAdvantages = []
        for path in paths:
            path = self.processPath(path)
            allAdvantages.extend(path["advantages"])
            totalRewards.append(sum(path["rewards"]))

        averageReward = sum(totalRewards) / len(totalRewards)
        print "average reward: ",averageReward 

        std = np.std(allAdvantages)
        avg = sum(allAdvantages) / len(allAdvantages)

        counter = 0
        for path in paths:
            # print map(int, path["expectations"])
            for (action, state, advantage, td_error) in zip (path["actions"], path["states"], path["advantages"], path["td_errors"]):
                X_batch_policy = np.empty((0,self.size_observation), dtype = np.float64)
                Y_batch_policy = np.empty((0,self.size_action), dtype = np.float64)

                counter += 1
                actionVector = self.runModel(self.policyModel, state)
                # performance = (advantage - avg) / std
                performance = td_error

                actionVector[action] += performance
                    
                X_batch_policy = np.append(X_batch_policy, np.array([state]), axis=0)
                Y_batch_policy = np.append(Y_batch_policy, np.array([actionVector]), axis=0)

                self.policyModel.fit(X_batch_policy, Y_batch_policy, batch_size = 1, nb_epoch=1, verbose = 0)

    def learn(self, paths):
        totalRewards = []
        allAdvantages = []
        for path in paths:
            path = self.processPath(path)
            allAdvantages.extend(path["advantages"])
            totalRewards.append(sum(path["rewards"]))

        averageReward = sum(totalRewards) / len(totalRewards)

        std = np.std(allAdvantages)
        avg = sum(allAdvantages) / len(allAdvantages)

        X_batch_policy = np.empty((0,self.size_observation), dtype = np.float64)
        Y_batch_policy = np.empty((0,self.size_action), dtype = np.float64)

        X_batch_Q = np.empty((0,self.size_observation), dtype = np.float64)
        Y_batch_Q = np.empty((0,self.size_action), dtype = np.float64)

        for path in paths:
            for (action, state, advantage, R, expectation) in zip (path["actions"], path["states"], path["advantages"], path["returns"], path["expectations"]):
                actionVector = self.runModel(self.policyModel, state)
                # performance = (advantage - avg) / std
                # performance = advantage / expectation
                performance = advantage / std

                qValues = self.runModel(self.qValueModel, state)
                qValues[action] = R

                X_batch_Q = np.append(X_batch_Q, np.array([state]), axis=0)
                Y_batch_Q = np.append(Y_batch_Q, np.array([qValues]), axis=0)

                if performance != 0:
                    actionVector[action] += performance
                        
                    X_batch_policy = np.append(X_batch_policy, np.array([state]), axis=0)
                    Y_batch_policy = np.append(Y_batch_policy, np.array([actionVector]), axis=0)

                self.qValueModel.fit(X_batch_Q, Y_batch_Q, nb_epoch=1, verbose = 0)
        self.policyModel.fit(X_batch_policy, Y_batch_policy, nb_epoch=1, verbose = 0)

    def calculateQTarget(self, qValuesNewState, reward, isFinal):
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def learnOnMiniBatch(self, miniBatchSize):
        miniBatch = self.memory.getMiniBatch(miniBatchSize)
        X_batch = np.empty((0,self.size_observation), dtype = np.float64)
        Y_batch = np.empty((0,self.size_action), dtype = np.float64)
        for sample in miniBatch:
            isFinal = sample['isFinal']
            state = sample['state'].copy()
            action = sample['action']
            reward = sample['reward']
            newState = sample['newState'].copy()

            qValues = self.runModel(self.qValueModel, state)
            qValuesNewState = self.runModel(self.qValueModel, newState)
            targetValue = self.calculateQTarget(qValuesNewState, reward, isFinal)

            X_batch = np.append(X_batch, np.array([state]), axis=0)
            Y_sample = qValues
            Y_sample[action] = targetValue
            Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            if isFinal:
                X_batch = np.append(X_batch, np.array([newState]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[0] * self.size_action]), axis=0)
        self.qValueModel.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

        
    