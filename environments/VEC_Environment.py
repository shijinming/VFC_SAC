import copy
import random
from collections import namedtuple
import gym
import torch
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint
from scipy.optimize import fsolve

class VEC_Environment(gym.Env):
    environment_name = "Vehicular Edge Computing"

    def __init__(self, num_vehicles=50, task_num=30):
        self.num_vehicles = num_vehicles
        self.task_num_per_episode = task_num
        self.vehicle_count = 0
        self.maxR = 500 #m, max relative distance between request vehicle and other vehicles
        self.maxV = 30 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 50 # maximum vehicles in the communication range of request vehicle
        self.max_local_task = 10
        self.local_total_freq = 0
        self.local_remain = 0
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(5,11)  #GHz
        self.data_size = [0.1, 0.2] #MBytes
        self.comp_size = [0.2, 0.5] #GHz
        self.tau = [0.5, 1, 2, 4] #s
        self.max_datasize = max(self.data_size)
        self.max_compsize = max(self.comp_size)
        self.max_tau = max(self.tau)
        self.ref_price = 0.1
        self.local_price = 0.2
        self.serv_price = 0.25
        self.price_level = 10
        self.service_threshold = 0.1
        self.local_priority = 0.01
        self.distance_factor = 1
        self.penalty = -np.log(1+self.max_tau)

        self.action_space = spaces.Box(-1,1,shape=(3,),dtype='float32')
        self.observation_space = spaces.Dict({
            "snr":spaces.Box(0,self.snr_ref,shape=(self.max_v,),dtype='float32'),
            "freq_remain":spaces.Box(0,6,shape=(self.max_v+1,),dtype='float32'),
            "serv_prob":spaces.Box(0,1,shape=(self.max_v,),dtype='float32'),
            "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau),shape=(4,),dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "VEC"
        self.count = [0,0,0,0]
        self.delay = [0,0,0,0]
        self.count_file = "sac.txt"
        self.utility = 0
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading
        self.init_vehicles()
        # self.generate_offload_tasks()
        self.generate_local_tasks()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.move_vehicles()
        # self.add_vehicle()
        # self.generate_local_tasks()
        # self.generate_offload_tasks()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        for v in self.vehicles:
            v["freq"] = v["freq_init"]
            v["freq_remain"] = max(0, v["freq_init"] - sum([i[1]/i[2] for i in v["tasks"]]))
            v["position"] = v["position_init"]
        self.local_remain = self.local_total_freq
        with open(self.count_file,'a') as f:
            f.write(str(self.utility)+' '+' '.join([str(i) for i in self.count])+' '
            +' '.join([str(i) for i in self.delay])+'\n')
        self.count = [0,0,0,0]
        self.delay = [0,0,0,0]
        self.utility = 0
        task = self.tasks[0]
        self.s = {
            "snr":np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "freq_remain":np.array([self.local_remain]+[v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "serv_prob":np.array([self.compute_service_availability(task, v) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "task":np.array(task)}
        return spaces.flatten(self.observation_space, self.s)

    def step(self, action):
        self.step_count += 1
        self.reward = self.compute_reward(action)
        self.utility += self.reward
        self.move_vehicles()
        if self.step_count >= self.task_num_per_episode: 
            self.done = True
        else: 
            self.done = False
            self.s["snr"] = np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["freq_remain"] = np.array([self.local_remain]+[v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            task = self.tasks[self.step_count]
            self.s["serv_prob"]= np.array([self.compute_service_availability(task, v) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["task"] = np.array(task)
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        task = self.s["task"]
        reward, v_id, local_freq, serv_freq = self.compute_utility(action, task)
        if v_id==self.num_vehicles:
            return reward
        v = self.vehicles[v_id]
        v["freq"] -= serv_freq
        v["freq_remain"] = max(0, v["freq"] - sum([i[1]/i[2] for i in v["tasks"]]))
        self.local_remain = max(self.local_remain - local_freq, 0)
        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR*0.9,self.maxR*0.9)
            v_v = np.random.normal(0, self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[], "u_max":0})
        self.local_total_freq = random.choice(self.vehicle_F)
        self.local_remain = self.local_total_freq

    def add_vehicle(self):
        if len(self.vehicles) <= self.num_vehicles:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = np.random.normal(0,self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[], "u_max":0})

    def move_vehicles(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]/3.6*0.1
            # if abs(self.vehicles[i]["position"]) >= self.maxR:
            #     self.vehicles.pop(i)
            #     self.add_vehicle()

    def generate_local_tasks(self):
        for v in self.vehicles:
            v["tasks"] = []
            for _ in range(random.randint(1,self.max_local_task)):
                max_t = random.choice(self.tau)
                data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                v["tasks"].append([data_size, compute_size, max_t])
    
    def generate_offload_tasks(self, file, task_num, group_num):
        with open(file,'w+') as f:
            for _ in range(group_num):
                f.write("tasks:\n")
                for _ in range(task_num):
                    max_t = random.choice(self.tau)
                    data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                    compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                    task = [str(data_size), str(compute_size), str(max_t)]
                    f.write(' '.join(task)+'\n')

    def generate_change_tasks(self, file, group_num):
        with open(file,'w+') as f:
            for group in range(1, group_num+1):
                f.write("tasks:\n")
                tasks = []
                for j in range(len(self.tau)):
                    for _ in range(8):
                        max_t = self.tau[j]
                        data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                        compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                        tasks.append(str(data_size)+' '+str(compute_size)+' '+str(max_t))
                np.random.shuffle(tasks)
                f.write('\n'.join(tasks)+'\n')
        

    def produce_action(self, action_type):
        if action_type=="random":
            v_id = np.random.choice(range(self.num_vehicles))
            local_fraction = np.random.random()
            serv_fraction = np.random.random()
        if action_type=="greedy":
            v_id, local_fraction, serv_fraction = self.greedy_action()
        # selection = [0]*self.num_vehicles
        # selection[v_id] = 1
        # action = np.array([local_fraction, serv_fraction] + selection)*2-1
        action = np.array([(v_id+0.5)/self.num_vehicles, local_fraction, serv_fraction])*2-1
        return action

    def load_offloading_tasks(self, file, index):
        a = []
        self.tasks = []
        with open(file) as f:
            a = f.read().split("tasks:\n")[index].split('\n')
        for i in a[:-1]:
            tmp = i.split(' ')
            self.tasks.append([float(k) for k in tmp])

    def compute_service_availability(self, task, v):
        T = max(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 0.00001)
        p_t = max(0, 1-task[2]/T)
        R = (min(100, abs(v["position"]))**(-self.distance_factor)/self.service_threshold
        - self.local_priority*(1-v["freq_remain"]/v["freq_init"]))**(-1/self.distance_factor)
        R = min(R, 500)/1000
        epsilon = np.exp(-2*R*self.num_vehicles*0.5)
        service_availability = epsilon*p_t
        # print(service_availability,end=',')
        return service_availability

    def compute_utility(self, action, task):
        action = (action+1)/2
        if np.isnan(action[0]):
            return self.penalty, 0, 0, 0
        v_id = int(action[0]*self.num_vehicles)
        if v_id==self.num_vehicles:
            return self.penalty, 0, 0, 0
        # v_id = np.argmax(action[2:])
        v = self.vehicles[v_id]
        snr = self.s["snr"][v_id]
        time_remain = max(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 0.00001)
        local_freq = action[1]*self.local_remain
        serv_freq = action[2]*v["freq_remain"]
        local_freq = 0.00001 if local_freq==0 else local_freq
        serv_freq = 0.00001 if serv_freq==0 else serv_freq
        t_local = task[1]/local_freq
        t_serv = task[0]/(self.bandwidth*np.log2(1+snr)) + task[1]/serv_freq
        fraction = t_local/(t_local + t_serv)
        t_total = fraction*t_serv
        cost = (fraction*self.serv_price*serv_freq + (1-fraction)*self.local_price*local_freq)*task[1]
        if t_total <= min(task[2], time_remain):
            utility = np.log(1+task[2]-t_total) - cost
            self.count[int(np.log2(task[2]))+1] += 1
            self.delay[int(np.log2(task[2]))+1] += t_total
        else:
            utility = self.penalty - cost
        return utility, v_id, local_freq, serv_freq

    def greedy_action(self):
        task = self.s["task"]
        v_id = np.argmax(self.s["freq_remain"][1:])
        v = self.vehicles[v_id]
        t_trans = task[0]/(self.bandwidth*np.log2(1+self.s["snr"][v_id]))
        result=[0,0,0]
        max_u = -1e9
        step = 100
        if v["freq_remain"]<=0:
            return v_id, 0, 0
        for j in range(step+1):
            serv_freq = j/step*v["freq_remain"]
            serv_freq = 0.00001 if serv_freq==0 else serv_freq
            t_serv = t_trans + task[1]/serv_freq
            for i in range(step+1):
                local_freq = i/step*self.local_remain
                local_freq = 0.00001 if local_freq==0 else local_freq
                t_local = task[1]/local_freq
                fraction = t_local/(t_local + t_serv)
                t_total = fraction*t_serv
                cost = (fraction*self.serv_price*serv_freq + (1-fraction)*self.local_price*local_freq)*task[1]
                time_remain = max(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 0.00001)
                if t_total <= min(task[2], time_remain):
                    utility = np.log(1+task[2]-t_total) - cost
                else:
                    utility = self.penalty - cost
                if utility>max_u:
                    result = [utility, i, j]
                    max_u = utility
        return v_id, result[1]/step, result[2]/step

