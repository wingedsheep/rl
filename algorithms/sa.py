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
from keras.models import model_from_json

# import other stuff
import random
import math
import numpy as np

class Agent:
    def __init__(self, network=None, fitness=None):
        self.network = network
        self.fitness = fitness
        self.nr_simulations = 0

    def setNetwork(self, network):
        self.network = network

    def setFitness(self, fitness):
        self.fitness = fitness

    def setNrSimulations(self, nr_simulations):
        self.nr_simulations = nr_simulations

    def __repr__(self):
        return 'Agent {0}'.format(self.fitness)

class ScoreCounter:
    def __init__(self):
        self.last100 = []
        self.i = 0

    def add(self, new):
        if len(self.last100) < 100:
            self.last100.append(new)
        else:
            if self.i == 100:
                self.i = 0
            self.last100[self.i] = new
            self.i += 1

    def average(self):
        return sum(self.last100)/len(self.last100)


class Connection:
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

class SA:
    def __init__(self, env, starting_temp = 300, final_temp = 0.1, max_change = 0.5, max_modifications= 1, hidden_layers =[30]):
        self.input_size = len(env.observation_space.high)
        self.output_size = env.action_space.n
        self.env = env
        self.starting_temp = starting_temp
        self.temp = starting_temp
        self.final_temp = final_temp
        self.max_change = max_change
        self.max_modifications = max_modifications
        self.score_counter = ScoreCounter()
        self.initAgent(hidden_layers)
   
    def initAgent(self, hiddenLayers):
        self.hiddenLayers = hiddenLayers
        agent = Agent()

        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu")
        agent.setNetwork(model)

        self.agent = agent

    def cloneAgent(self, agent):
        newAgent = Agent()
        newNetwork = self.createModel(self.input_size, self.output_size, self.hiddenLayers, "relu")
        newNetwork.set_weights(agent.network.get_weights())
        newAgent.setNetwork(newNetwork)
        return newAgent

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType):
        bias = False
        dropout = 0.1
        regularizationFactor = 0
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

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
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)-1):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=1, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def backupNetwork(self, model, backup):
        weights = model.get_weights()
        backup.set_weights(weights)

    # predict Q values for all the actions
    def getValues(self, state, agent):
        predicted = agent.network.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getMaxValue(self, values):
        return np.max(values)

    def getMaxIndex(self, values):
        return np.argmax(values)

    # select the action with the highest value
    def selectAction(self, qValues):
        action = self.getMaxIndex(qValues)
        return action

    # could be useful in stochastic environments
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

    def run_simulation(self, env, agent, steps, render = False):
        observation = env.reset()
        totalReward = 0
        for t in xrange(steps):
            if (render):
                env.render()
            values = self.getValues(observation, agent)
            action = self.selectAction(values)
            newObservation, reward, done, info = env.step(action)
            totalReward += reward
            observation = newObservation
            if done:
                break
        if (agent.nr_simulations == 0):
            agent.setFitness(totalReward)
            agent.setNrSimulations(1)
        else :
            old_fitness = agent.fitness
            nr_simulations = agent.nr_simulations
            new_fitness = (old_fitness * nr_simulations + totalReward) / (nr_simulations + 1)
            agent.setNrSimulations(nr_simulations + 1)
            agent.setFitness(new_fitness)
        self.score_counter.add(totalReward)
        return totalReward

    def createWeightLayerList(self):
        weightLayerList = []
        for i in xrange(len(self.hiddenLayers)):
            weightLayerList.append(i * 2)
        return weightLayerList

    def mutate(self, agent, mutationFactor, max_modifications):
        nr_modifications = random.randint(1, max_modifications)
        new_weights = agent.network.get_weights()
        for m in xrange(nr_modifications):
            i = random.sample(self.createWeightLayerList(),1)[0]
            layer = new_weights[i]
            j = random.sample(range(len(layer)),1)[0]
            neuronConnectionGroup = layer[j]
            k = random.sample(range(len(neuronConnectionGroup)), 1)[0]
            rand = (random.random() - 0.5) * mutationFactor
            new_weights[i][j][k] = neuronConnectionGroup[k] + rand
        agent.network.set_weights(new_weights)

    def tryAgent(self, agent, nr_episodes):
        total = 0
        for i in xrange(nr_episodes):
            total += self.run_simulation(self.env, agent, self.steps)
        return total / nr_episodes


    def calculateAlpha(self):
        return (self.final_temp/self.starting_temp)**(1.0/(self.epochs));

    def decrease_temp(self, alpha):
        self.temp *= alpha

    def initialize(self, nr_tries):
        bestAgent = self.cloneAgent(self.agent)
        self.tryAgent(bestAgent, self.rounds_per_epoch)
        for i in xrange(nr_tries -1):
            self.initAgent(self.hiddenLayers)
            self.tryAgent(self.agent, self.rounds_per_epoch)
            print self.agent.fitness
            if (self.agent.fitness > bestAgent.fitness):
                bestAgent = self.cloneAgent(self.agent)
                bestAgent.setFitness(self.agent.fitness)
        self.agent = bestAgent
        print "best agent fitness ",self.agent.fitness


    def run(self, steps, epochs, rounds_per_epoch = 75, score_target = 1e9, nr_initializations = 1):
        self.rounds_per_epoch = rounds_per_epoch
        self.steps = steps
        self.epochs = epochs
        self.score_target = score_target
        self.nr_initializations = nr_initializations

        alpha = self.calculateAlpha()
        self.initialize(nr_initializations)
        for e in xrange(self.epochs):
            newAgent = self.cloneAgent(self.agent)
            self.mutate(newAgent, self.max_change, self.max_modifications)
            self.tryAgent(newAgent, self.rounds_per_epoch)

            if (self.agent.fitness < newAgent.fitness):
                self.agent = newAgent
            else :
                rand = random.random()
                probability = math.exp((newAgent.fitness - self.agent.fitness)/ self.temp)
                print probability
                if rand < probability:
                    self.agent = newAgent
                else :
                    self.tryAgent(self.agent, 1)

            average = self.score_counter.average()
            print "New agent fitness: ",newAgent.fitness
            print "Epoch:",e," fitness:",self.agent.fitness," average: ",average

            if average >= self.score_target:
                break

            self.decrease_temp(alpha)

