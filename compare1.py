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
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(5,11)  #GHz
        self.data_size = [0.05, 0.1] #MBytes
        self.comp_size = [0.2, 0.4] #GHz
        self.tau = [0.5, 1, 2, 4] #s
        self.max_datasize = max(self.data_size)
        self.max_compsize = max(self.comp_size)
        self.max_tau = max(self.tau)
        self.priority = [0.5, 1]
        self.ref_price = 0.1
        self.price = 0.1
        self.price_level = 10
        self.service_threshold = 0.1
        self.local_priority = 0.01
        self.distance_factor = 1
        self.high_priority_factor = -np.log(1+self.max_tau)
        self.low_priority_factor = np.log(1+np.min(self.tau))

        self.action_space = spaces.Discrete(self.num_vehicles*self.price_level)
        self.observation_space = spaces.Dict({
            "snr":spaces.Box(0,self.snr_ref,shape=(self.max_v,),dtype='float32'),
            # "time_remain":spaces.Box(0,100,shape=(self.max_v,),dtype='float32'),
            "freq_remain":spaces.Box(0,6,shape=(self.max_v,),dtype='float32'),
            "serv_prob":spaces.Box(0,1,shape=(self.max_v,),dtype='float32'),
            "u_max":spaces.Box(0,self.max_local_task*self.max_tau,shape=(self.max_v,),dtype='float32'),
            "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau, max(self.priority)),shape=(4,),dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "VEC"
        self.high_count = [0,0,0,0]
        self.high_delay = [0,0,0,0]
        self.low_count = [0,0,0,0]
        self.low_delay = [0,0,0,0]
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
        # for _ in range(random.randint(1,10)):
        #     self.add_vehicle()
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
            alpha_max = v["freq_remain"]/v["freq"]
            v["u_max"] = sum([np.log(1+alpha_max*i[2]) for i in v["tasks"]])
        with open(self.count_file,'a') as f:
            f.write(str(self.utility)+' '+' '.join([str(i) for i in self.low_count])+' '
            +' '.join([str(i) for i in self.low_delay])+' '
            +' '.join([str(i) for i in self.high_count])+' '
            +' '.join([str(i) for i in self.high_delay])+' '+'\n')
        self.high_count = [0,0,0,0]
        self.high_delay = [0,0,0,0]
        self.low_count = [0,0,0,0]
        self.low_delay = [0,0,0,0]
        self.utility = 0
        task = self.tasks[0]
        self.s = {
            "snr":np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            # "time_remain":np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "freq_remain":np.array([v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "u_max":np.array([v["u_max"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
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
            self.s["freq_remain"] = np.array([v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["u_max"] = np.array([v["u_max"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            # self.s["time_remain"] = np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            task = self.tasks[self.step_count]
            self.s["serv_prob"]= np.array([self.compute_service_availability(task, v) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["task"] = np.array(task)
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        task = self.s["task"]
        reward, v_id, freq_alloc = self.compute_utility(action, task)
        if v_id==self.num_vehicles:
            return reward
        v = self.vehicles[v_id]
        v["freq"] -= freq_alloc
        v["freq_remain"] = max(0, v["freq"] - sum([i[1]/i[2] for i in v["tasks"]]))
        alpha_max = v["freq_remain"]/v["freq"]
        v["u_max"] = sum([np.log(1+alpha_max*i[2]) for i in v["tasks"]])
        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR*0.9,self.maxR*0.9)
            v_v = np.random.normal(0, self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[], "u_max":0})

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
                priority = random.choice(self.priority[1:])
                v["tasks"].append([data_size, compute_size, max_t, priority])
    
    def generate_offload_tasks(self, file, task_num, group_num):
        with open(file,'w+') as f:
            for _ in range(group_num):
                f.write("tasks:\n")
                for _ in range(task_num):
                    max_t = random.choice(self.tau)
                    data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                    compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                    priority = random.choice(self.priority)
                    task = [str(data_size), str(compute_size), str(max_t), str(priority)]
                    f.write(' '.join(task)+'\n')

    def generate_priority_tasks(self, file, group_num):
        with open(file,'w+') as f:
            for _ in range(group_num):
                f.write("tasks:\n")
                tasks = []
                for i in [0.5,1]:
                    for j in range(len(self.tau)):
                        for _ in range(4):
                            max_t = self.tau[j]
                            data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                            compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                            priority = i
                            tasks.append(str(data_size)+' '+str(compute_size)+' '+str(max_t)+' '+str(priority))
                np.random.shuffle(tasks)
                f.write('\n'.join(tasks)+'\n')
        

    def produce_action(self, action_type):
        if action_type=="random":
            v_id = np.random.choice(range(self.num_vehicles))
            fraction = np.random.choice(range(self.price_level-1))
        if action_type=="greedy":
            v_id = np.argmax(self.s["freq_remain"])
            task = self.s["task"]
            fraction = np.argmax([self.compute_utility(v_id*self.price_level+i, task, False)[0] for i in range(1,self.price_level)])
        action = v_id*self.price_level + fraction + 1
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
        R = self.priority[0]**(1/self.distance_factor)*(task[3]*min(100, abs(v["position"]))**(-self.distance_factor)/self.service_threshold
        - self.local_priority*(1-v["freq_remain"]/v["freq_init"]))**(-1/self.distance_factor)
        R = min(R, 500)/1000
        epsilon = np.exp(-2*R*self.num_vehicles*0.5)
        service_availability = epsilon*p_t
        # print(service_availability,end=',')
        return service_availability

    def compute_utility(self, action, task, is_count=True):
        v_id = action//self.price_level
        if v_id==self.num_vehicles:
            return 0, v_id, 0
        utility = -np.log(1+self.max_tau)
        v = self.vehicles[v_id]
        u_max = self.s["u_max"][v_id]
        u_alpha = u_max - (action%self.price_level+1)/self.price_level*u_max
        cost = u_max - u_alpha + self.price*task[1]
        alpha_max = v["freq_remain"]/v["freq"]
        alpha = fsolve(lambda a:sum([np.log(1+a*i[2]) for i in v["tasks"]])-u_alpha, 0.001)[0]
        alpha = min(max(0,alpha), alpha_max)
        freq_alloc = v["freq"]-(v["freq"]-v["freq_remain"])/(1-alpha)
        # fraction = (action%self.price_level)/self.price_level
        # freq_alloc = fraction*v["freq_remain"]
        if freq_alloc <= 0:
            return utility, v_id, 0
        snr = self.s["snr"][v_id]
        t_total = task[0]/(self.bandwidth*np.log2(1+snr)) + task[1]/freq_alloc
        # time_remain = max(self.s["time_remain"][v_id], 0.00001)
        time_remain = max(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 0.00001)
        # cost = self.ref_price*task[1]
        if task[3]==self.priority[0]:
            if t_total <= time_remain:
                if t_total <= task[2]:
                    utility = self.low_priority_factor -cost
                else:
                    utility = self.low_priority_factor*np.exp(-0.5*(t_total-task[2])) - cost
                if is_count:
                    self.low_count[int(np.log2(task[2]))+1] += 1
                    self.low_delay[int(np.log2(task[2]))+1] += t_total
            else:
                utility = 0 - cost
        elif task[3]==self.priority[1]:
            if t_total <=min(task[2], time_remain):
                utility = np.log(1+task[2]-t_total) - cost
                if is_count:
                    self.high_count[int(np.log2(task[2]))+1] += 1
                    self.high_delay[int(np.log2(task[2]))+1] += t_total
            else:
                utility = self.high_priority_factor - cost
        return utility, v_id, freq_alloc

actionType = "greedy"
group = 5
count_file = "../sac/"+actionType+".txt"
num_episode = 10
trials = 100
action_type = [actionType]
task_num = 32
task_file = "../sac/tasks.txt"
# config.environment = VEC_Environment(num_vehicles=50, task_num=task_num)
# config.environment.generate_change_tasks(task_file, 10)
with open(count_file,'w+') as f:
    f.write("")
for iter in range(1000):
    for num_vehicles in range(5,51,5):
        print("iter=",iter,"num_vehicles=",num_vehicles)
        environment = VEC_Environment(num_vehicles=num_vehicles, task_num=task_num)
        environment.load_offloading_tasks(task_file, iter%5+1)
        environment.count_file = count_file
        with open(count_file,'a') as f:
            f.write("num_vehicles="+str(num_vehicles)+'\n')
        for i in action_type:
            print(i)
            results = []
            rollings = []
            if i=="greedy":
                num_episode = 2
            elif i=="random":
                num_episode = 2000
            for _ in range(num_episode):
                environment.reset()
                reward = 0
                for _ in range(task_num):
                    _,r,_,_=environment.step(environment.produce_action(i))
                    reward+=r
                results.append(reward)
                rollings.append(np.mean(results[-trials:]))
            print("mean_reward=", np.mean(results),"max_reward=",max(results))