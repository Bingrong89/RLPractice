#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import time
import cv2 
from tqdm import tqdm
from datetime import date
import numpy as np
import gym
from gym import envs
from collections import namedtuple
import random as rdm
import copy
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
dqn2->dqn3
use torch float 16, and convert images to 84x84, decreases memory consumption per image tensor by ~10%.
"""
"""
dqn3->dqn4
if continuous NOOP for 30turns, sub noop with random non-noop action instead
this is implemented inside replay_memory.store_transition
which means this condition is only imposed on the training data

also switched the optimizer from rmsprop to adadelta, because i can

#time to double to epsilon decay schedule control
"""



def sizecheck(x1,f,p,s):
    return ( ((x1-f+ (2*p))/s)+1      )

def output_final_size(inputdim,convlayers): #expect a list of conv2d layers
    #get the final feature map spatial size
    outputdim = inputdim
    for convlayer in convlayers:
        outputdim = int( sizecheck(outputdim,convlayer.kernel_size[0],convlayer.padding[0],convlayer.stride[0]))
    return outputdim 

class Modelz(nn.Module):
    def __init__(self,env):
        super(Modelz,self).__init__()
        self.conv1 = nn.Conv2d(3,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        
        inputheight = 84#env.observation_space.shape[0]
        inputwidth = 84#env.observation_space.shape[1]
        
        ffmap_ht = output_final_size(inputheight,[self.conv1,self.conv2,self.conv3])
        ffmap_wdth = output_final_size(inputwidth,[self.conv1,self.conv2,self.conv3])
        
        self.linear1 = nn.Linear(ffmap_ht*ffmap_wdth*self.conv3.out_channels,512)
        self.linear2 = nn.Linear(512,env.action_space.n)
        
    def forward(self,x):

        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))

        output = F.relu(self.linear1(output.view(output.shape[0],-1)))
        output = self.linear2(output)

        return output           
    
class Replay_Memory():
    def __init__(self,capacity):
        self.capacity = capacity
        self.data = [None] * capacity #a list to store transitions as namedtuple Ex
        self.counter = 0
        self.full = 0
        self.noop_counter = 0
        self.noop_limit = 30
    
    def store_transition(self,transition): #transition arg shd alrd be as Ex
        if isinstance(transition,Ex) == False:
            raise Exception("Replay_Memory.store_transition input arg should be in format Ex(namedtuple)")
            
        if self.counter == self.capacity:
            self.counter = 0
            self.full +=1
        if transition.at == 0:
            self.noop_counter +=1
        elif transition.at !=0:
            self.noop_counter = 0
        if self.noop_counter == self.noop_limit:
            transition2 = Ex(transition.st,rdm.randint(1,6),transition.rt,transition.st2,transition.done)
            self.data[self.counter] = transition2
            self.counter +=1
        else:       
            self.data[self.counter] = transition
            self.counter += 1
    
    def reset_counterfull(self):
        self.counter =0
        self.full = 0
    
    def sample_minibatch(self,batchsize):
        if self.full ==0:
            return rdm.sample(self.data[0:self.counter],batchsize)
        else:
            return rdm.sample(self.data,batchsize)


#this is solely used for building replay memory, not directly used in training
#might be possible to build replaymemory by selecting actions in batch too, atleast def doable for the nn part.
def select_action(epsilon,env,policynet,currentstate): #feed current state in pytorch format
    if rdm.random() <= epsilon:
        return env.action_space.sample()
    else:
        return policynet(currentstate).argmax(1).item()

def conv_obs_torch(obsimg): #change the numpy image to appropriate torch format
    #resize to 84x84 and to torch 16
    obsimg = cv2.resize(obsimg,(84,84))
    
    return torch.from_numpy(obsimg.astype(np.float16).transpose(2,0,1)).unsqueeze(0)

def fill_memory(memory,env,policynet,epsilon): #this fx only covers 1 episode 
    state_now = conv_obs_torch(env.reset())
    done = False
    while done == False:
        action_to_take = select_action(epsilon,env,policynet,state_now)
        state_next,reward,done,info = env.step(action_to_take)
        state_next = conv_obs_torch(state_next)
        clip_reward = min(max(reward,-1),1)      
        memory.store_transition(Ex(state_now,action_to_take,clip_reward,state_next,done))
        state_now = state_next

def fill_replaymem(memory,env,policynet,epsilon,cutoff): #this covers filling the entire replaymem
    while memory.full==0:
        if memory.counter >= cutoff:
            break
        else:
            fill_memory(memory,env,policynet,epsilon)
        
def get_value1 (pred_outputs,batch_actions):
    output = torch.zeros(len(pred_outputs),device=device)
    for x in range(len(pred_outputs)):
        output[x] = pred_outputs[x][batch_actions[x]]
        
    return output

def get_batch_attributes(replay_mem,batchsize):
    ex_batch = replay_mem.sample_minibatch(batchsize)
    batch_st1 = torch.empty(0,device=device)
    batch_st2 = torch.empty(0,device=device)
    batch_rewards = []
    batch_actions = []
    donelist = []
    
    for experience in ex_batch:
        batch_st1 = torch.cat((batch_st1,experience.st.float().cuda()),dim=0 )
        batch_st2 = torch.cat((batch_st2,experience.st2.float().cuda()),dim=0 )
        batch_rewards.append(experience.rt)
        batch_actions.append(experience.at)
        donelist.append(experience.done)
    return batch_st1,batch_st2,\
           torch.tensor(batch_rewards,device=device),torch.tensor(batch_actions,device=device),donelist

def sub_terminationvalue(gt_estimate,donelist,batch_rewards): # set gt estimate = r if episode terminates at st2
    for idx,x in enumerate(donelist):
        if x == True:
            gt_estimate[idx] = batch_rewards[idx]

def play_da_game(times,env,targetnet,epsilon): #to evaluate, let it play a fix number of times and compare the total rewards, i suppose.

    total_score = 0 
    
    with torch.no_grad():
        for x in tqdm(range(times),desc="Playing Game..."):
            done = False
            state = env.reset()
            while done == False:
                action = select_action(epsilon,env,targetnet,conv_obs_torch(state).float().cuda())
                state,reward,done,info = env.step(action)
                total_score += reward
    
    return total_score/times

def train_model2(epochs,batchsize,discount,C): #follow the general procedure outlined in paper
        #this one doesnt get a pretty tqdm progress bar to show because of the way it is run. no way for tqdm to predict how long an episode is.
        #they didnt put a termination condition in describing Algorithm 1 
        replay_mem.reset_counterfull()
        policynet.train()
        fill_replaymem(replay_mem,env,targetnet,1.0,cutoff = 10000)  
        #replay_mem.reset_counterfull()
        stepcount=0 #C ! the update control variable
        #still yet to think of how to properly decrease this

        for epoch in range(epochs): #epoch == episode
            print("Training Epoch %i"%epoch)
            state = env.reset()
            state = conv_obs_torch(state)
            done = False
            
            while done == False:
                
                
               # epsilon = epsilonarray[epsiloncontrol]
               #took this from pytorch becoz i don have a better idea on how to 'linearly anneal'
               #but this schedule is definitely not linear -.-
                #epsilon = 0.1 + (1-0.1)*math.exp(-1. * stepcount / 2000)
                epsilon = 1 -(0.9/400000)*stepcount
                
                with torch.no_grad():
                    action = select_action(epsilon,env,policynet,state.float().cuda())

                nextstate,reward,done,info = env.step(action)
                nextstate = conv_obs_torch(nextstate)
                reward =  min(max(reward,-1),1) 
                replay_mem.store_transition(Ex(state,action,reward,nextstate,done))
                state = nextstate
                
                #nn training here 
                optimizer.zero_grad()
                batch_st1,batch_st2,batch_rewards,batch_actions,donelist = get_batch_attributes(replay_mem,batchsize)
                pred_outputs = policynet(batch_st1)

                action_value = get_value1(pred_outputs,batch_actions)
                with torch.no_grad():
                    bootstrap_values = targetnet(batch_st2)
                    max_bsval,max_bsaction = bootstrap_values.max(1)
                    gt_estimate =  batch_rewards +  (discount*max_bsval)
                    sub_terminationvalue(gt_estimate,donelist,batch_rewards)
                

                loss = criterion(action_value,gt_estimate)
                loss.backward()
                                
                for param in policynet.parameters():
                    param.grad.data.clamp_(-1, 1)

                    
                optimizer.step()
              #  epoch_loss += loss.item()
   
                stepcount += 1
                #Every C steps set targetnet to policynet
                if stepcount % C == 0:# and epoch!=0: #seems more reasonable to do evaluation here? 
                    targetnet.load_state_dict(policynet.state_dict()) 
                    playtimes = 50
                    averagescore = play_da_game(playtimes,env2,targetnet,0)
                    printstatement = "At Epoch %i, Stepcount %i. Average Score for %i games : %r"%(epoch,stepcount,playtimes,averagescore)
                    print(printstatement)


Ex = namedtuple('Experience',('st','at','rt','st2','done')) 
env = gym.make('AssaultDeterministic-v4')
env2 = gym.make('AssaultDeterministic-v4')
policynet = Modelz(env).to(device)
targetnet = copy.deepcopy(policynet).to(device)
replay_mem = Replay_Memory(250000)
criterion = nn.MSELoss()
#optimizer = optim.RMSprop(policynet.parameters(),lr=0.00025,momentum = 0.95,eps=0.01) #550 max
#optimizer = optim.Adadelta(policynet.parameters(),lr=0.1,weight_decay=0.001) #645 max, stepcount 4950000
optimizer = optim.Adadelta(policynet.parameters(),lr=0.05)#,weight_decay=0.0)
#optimizer = optim.SGD(policynet.parameters(),0.001)

train_model2(10000,32,0.99,10000)










































































      



