import os
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas'

# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from ...algorithms.deepq import DeepQ

env = gym.make('LunarLander-v3')

epochs = 10000
steps = 200
updateTargetNetwork = 10000
explorationRate = 1
minibatch_size = 36
learnStart = 36
learningRate = 0.00025
discountFactor = 0.99
memorySize = 10000000

last100Scores = [0] * 100
last100ScoresIndex = 0
last100Filled = False

renderPerXEpochs = 25
shouldRender = True

deepQ = DeepQ(len(env.observation_space.high), env.action_space.n, memorySize, discountFactor, learningRate, learnStart)
# deepQ.initNetworks([6])
# deepQ.initNetworks([8, 5])
deepQ.initNetworks([30, 30, 30])

stepCounter = 0

# env.monitor.start('/tmp/CartPole-v0-4')
# number of reruns
for epoch in xrange(epochs):
    observation = env.reset()
    print explorationRate
    # number of timesteps
    totalReward = 0
    for t in xrange(steps):
        if epoch % renderPerXEpochs == 0 and shouldRender:
            env.render()
        qValues = deepQ.getQValues(observation)

        action = deepQ.selectAction(qValues, explorationRate)

        newObservation, reward, done, info = env.step(action)

        totalReward += reward

        deepQ.addMemory(observation, action, reward, newObservation, done)

        if stepCounter >= learnStart:
            if stepCounter <= updateTargetNetwork:
                deepQ.learnOnMiniBatch(minibatch_size, False)
            else :
                deepQ.learnOnMiniBatch(minibatch_size, True)

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

        stepCounter += 1
        if stepCounter % updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()
            print "updating target network"

    explorationRate *= 0.995
    # explorationRate -= (2.0/epochs)
    explorationRate = max (0.05, explorationRate)

deepQ.printNetwork()

# env.monitor.close()
# gym.upload('/tmp/wingedsheep-lunarlander-deepQLearning8', api_key='sk_GC4kfmRSQbyRvE55uTWMOw')