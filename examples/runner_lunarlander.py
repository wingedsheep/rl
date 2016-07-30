import os
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas'

# import the gym stuff
import gym
# import own classes
from rl.algorithms.deepq import DeepQ

env = gym.make('LunarLander-v2')

epochs = 10000
steps = 1000
updateTargetNetwork = 10000
explorationRate = 1
miniBatchSize = 36
learnStart = 36
learningRate = 0.00025
discountFactor = 0.99
memorySize = 10000000

last100Scores = [0] * 100
last100ScoresIndex = 0
last100Filled = False

renderPerXEpochs = 25
shouldRender = True

hiddenLayers = [30, 30, 30]
bias = True

experimentId = 'LunarLander-v2'
force = True
api_key = None
upload = False

deepQ = DeepQ(
    env = env, 
    memorySize = memorySize, 
    discountFactor = discountFactor, 
    learningRate = learningRate, 
    hiddenLayers = hiddenLayers,
    bias = bias
)

deepQ.run(
    epochs = epochs, 
    steps = steps, 
    updateTargetNetwork = updateTargetNetwork, 
    explorationRate = explorationRate, 
    miniBatchSize = miniBatchSize, 
    learnStart = learnStart,
    renderPerXEpochs = renderPerXEpochs, 
    shouldRender = shouldRender,
    experimentId = experimentId,
    force = force,
    upload = upload,
    api_key = api_key
)