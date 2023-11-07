import retro
import numpy as np
import cv2 
import neat
import pickle

env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2.state')

imgarray = []
xpos_end = 0

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness= 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False

while not done:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    imgarray = np.ndarray.flatten(ob)    
    nnOutput = net.activate(imgarray)

    
    ob, rew, done, info = env.step(nnOutput)
    
    xpos = info['x']
    xpos_end = info['endOfLevel']

    if xpos > xpos_max:
        fitness_current += 1
        xpos_max = xpos
        coutner = 0
    else:
        coutner += 1

    if xpos_end == 1:
        fitness_current += 100000
        print("Ive Been Activated we have a winner")
        done = True

    #fitness_current += rew

    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
            
    if done or coutner == 300:
        done = True