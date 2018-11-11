#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:24:59 2018

author: magalidrumare
copyright Udemy Artificial Intelligence for Business. 

"""

# Importing the librairies 
import numpy as np 

# Setting the parameters gamma and alpha for Q_learning 

# temporal difference 
gamma = 0.75

# learning rate 
alpha = 0.9 


## PART 1 - DEFINING THE ENVIRONMENT 
# defining the state 
# creation of a dictionnary A is the key and 0 is the value
location_to_state = {'A': 0, 
                     'B': 1,
                     'C': 2, 
                     'D': 3, 
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# defining the action 
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# defining the reward 
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])


## PART 2 - BUILDING THE AI SOLUTION WITH Q_LEARNING 

#initializing the Q_value 
Q = np.array(np.zeros([12,12]))


# Implementing the Q_Learning process 

# for t>=1 to 1000

# 1.select a random state from 12 possible states 

for i in range(1000):
    # 12 is excluded
    # random function from an int 
    current_state = np.random.randint(0,12)

# 2. play a random action that can lead to the next step with a reward = 1
    playable_actions = []
    for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
    # random function from a list       

# 3. reach the next state and get the reward 
    next_state = np.random.choice(playable_actions)
            
# 4. compute the temporal difference
    # next_state at t+1 is the samething that the next_action 
    # TD = R(st,at) + gamma*maxQ(st+1,a)-Q(st,at)
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]

# 5. update the Q value by applying the Bellman equation 
    #Qt(s,a)= Qt-1(s,a)+ alpha*TD
    # next_state at t+1 is the samething that the next_action 
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
 
    
# PART 3_ GOING INTO PRODUCTION 
    
# inverse the mapping from state to location 
state_to_location = {state: location for location, state in location_to_state.items()}
    
# making the function that will return the optimal route 
def route(starting_location, ending_location): 
        
       
        
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        # convert location into state = index of row 
        starting_state = location_to_state[starting_location]
         # need to take the highest Q value of the state
        next_state = np.argmax(Q[starting_state,])
        # index of the row we have to go => need to be converted in location A, B,C,D....
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# Print the final route 
print('Route:')
route('E','G')

        
    
    
    
