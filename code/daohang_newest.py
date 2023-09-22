# coding=utf-8
# -*- coding:utf-8 -*-
import pickle, sys
import pandas as pd

sys.path.append("../")

# from simulator.utilities import *
from alg_utility import *
from envs import *

from time import time as tttt

import networkx as nx
import numpy as np

origin_distance = np.load('90distance10_new.npy')
distance = origin_distance
origin_crime = np.load('90crime10_newest.npy')
crime = origin_crime


def cost(a):
    Dcost = []
    Ccost = []
    for i in range(len(a) - 1):
        Dcost.append(distance[a[i]][a[i + 1]])
        Ccost.append(crime[a[i]][a[i + 1]])
    return Dcost, Ccost


def danshe(a):
    return 100 / a


# def find(which,G,Source,Goal):
#     if(which=='D'):
#         a = Dijkstra(G,Source,Goal)
#         Dcost,Ccost = cost(a)
#         print "路径上的距离代价为{}, 风险系数为{}".format(Dcost,Ccost)
#     if(which=='A'):
#         a = A_star(G,Source,Goal)
#         Dcost,Ccost = cost(a)
#         print "路径上的距离代价为{}, 风险系数为{}".format(Dcost,Ccost)

def find(mapped_matrix_int, a):
    M, N = mapped_matrix_int.shape
    for i in range(M):
        for j in range(N):
            if a == mapped_matrix_int[i][j]:
                return i, j


def find_du(i, j, n):
    x1 = i / n
    y1 = i % n
    x2 = j / n
    y2 = j % n
    du1 = x1 - x2
    if du1 < 0: du1 = -du1
    du2 = y1 - y2
    if du2 < 0: du2 = -du2
    du = du1 + du2
    return du


# 加入随机生成代码
import networkx as nx
import numpy as np
import random

################## Load data ###################################
dir_prefix = ""
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print "log dir is {}".format(log_dir)

data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
order_time_dist = []  # 原来是空的
order_price_dist = []
mapped_matrix_int = np.array([[5, 8, 2, 17, 15, 26, 27, 25, 30, 28], [1, 7, 3, 23, 19, 29, 31, 33, 35, 37],
                              [6, 4, 9, 11, 22, 32, 34, 36, 38, 39],
                              [16, 10, 13, -100, 21, 40, 42, 44, 46, 48], [12, 14, 18, 20, 24, 41, 43, 45, 47, 49],
                              [50, 52, 54, 56, 58, 60, 62, 64, 66, 68],
                              [51, 53, 55, 57, 59, 61, 63, 65, 67, 69], [70, 72, 74, 76, 78, 80, 82, 84, 86, 88],
                              [71, 73, 75, 77, 79, 81, 83, 85, 87, 89], [90, 92, 94, 96, 98, 91, 93, 95, 97, 99]])
M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = 99
idle_driver_location_mat = np.zeros((144, 99))

for ii in np.arange(144):
    time_dict = {}
    for jj in np.arange(M * N):  # num of grids
        time_dict[jj] = [2]
    order_num_dist.append(time_dict)
    idle_driver_location_mat[ii, :] = [2] * num_valid_grid

idle_driver_dist_time = [[10, 1] for _ in np.arange(144)]

n_side = 6
l_max = 2
order_time = [0.2, 0.2, 0.15,
              0.15, 0.1, 0.1,
              0.05, 0.04, 0.01]
order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
               [15.02, 6.90],  # mean and std of order price when duration is 20 min
               [23.22, 11.63],
               [32.14, 16.20],
               [40.99, 20.69],
               [49.94, 25.61],
               [58.98, 31.69],
               [68.80, 37.25],
               [79.40, 44.39]]
daoda = 99
chufa = 0
n1 = 100
m1 = 10
unvalid = 33

print "请输入您的到达点"
# daoda = input()

print "您的到达点为{}".format(daoda)

print "请输入您的出发点"
# chufa = input()

print "您的出发点为{}".format(chufa)

order_real = []
onoff_driver_location_mat = []
for tt in np.arange(144):
    for i in range(1, 100):
        a = []
        for j in range(1, 100):
            flag = random.randint(1, 10000)
            if flag < 20:
                if distance[i][j] != 0:
                    [row1, col1] = find(mapped_matrix_int, i)
                    [row2, col2] = find(mapped_matrix_int, j)
                    b = row1 - row2
                    if b < 0: b = -b
                    c = col1 - col2
                    if c < 0: c = -c
                    a.append([daoda, j, tt, 10, 1000 + b * 30 + c * 30])
        order_real += a

        onoff_driver_location_mat.append([[-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 3.36596622],
                                          [0.09090909, 8.46398452],
                                          [0.09090909, 3.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 3.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 4.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 7.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 6.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [-0.625, 2.92350389],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 1.46398452],
                                          [0.09090909, 2.36596622],
                                          [0.09090909, 1.46398452],
                                          ])
# print "order_real{}".format(order_real)


################## Initialize env ###################################
n_side = 6
GAMMA = 0.9

env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
               order_time, order_price, l_max, M, N, n_side, 1, np.array(order_real),
               np.array(onoff_driver_location_mat))

state = env.reset_clean()

order_response_rates = []
T = 0
max_iter = 144
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)

temp = np.array(env.target_grids) + env.M * env.N
# print temp
target_id_states = env.target_grids + temp.tolist()
# print target_id_states
# curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
# curr_s = utility_conver_states(curr_s, target_id_states)
print "******************* Finish generating one day order **********************"

print "******************* Starting training Deep SARSA **********************"
from algorithm.IDQN import *

MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
global_step = 0
city_time_end = city_time_start + EP_LEN
EPSILON = 0.9
gamma = 0.9
learning_rate = 1e-2

prev_epsiode_reward = 0
all_rewards = []
order_response_rates = []
value_table_sum = []
episode_rewards = []
episode_conflicts_drivers = []
order_response_rate_episode = []
episode_dispatched_drivers = []
T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T

# tf.reset_default_graph()
sess = tf.Session()
tf.set_random_seed(1)
q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=log_dir)

target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")  # 初始化
sess.run(tf.global_variables_initializer())  # 不太清楚干了什么
estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
replay = ReplayMemory(memory_size=1e+6, batch_size=3000)
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)

saver = tf.train.Saver()
save_random_seed = []
N_ITER_RUNS = 25
temp_value = 10
RATIO = 1
EPSILON_start = 0.5
EPSILON_end = 0.1
epsilon_decay_steps = 15
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)

data_save = []

for n_iter in np.arange(180):
    origin_distance = np.load('90distance10_new.npy')
    distance = origin_distance
    origin_crime = np.load('90crime10_newest.npy')
    crime = origin_crime
    print n_iter, origin_crime
    RANDOM_SEED = n_iter + MAX_ITER - temp_value
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    # reset env
    is_regenerate_order = 1
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    # print "#coding=utf-8"
    # print curr_state[1]
    info = env.step_pre_order_assigin(curr_state)
    context = stateprocessor.compute_context(info)  # 剩余空闲司机的矩阵
    # print context
    curr_s = stateprocessor.utility_conver_states(
        curr_state)  # curr_s是curr_state进行flatten操作以后，再去除非法节点的数据(example中第一个节点的driver和order数据)后的一维矩阵，在example中，curr_s是18-2=16个空间大小
    # print curr_s
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # print "normalized_curr_s"
    # print normalized_curr_s
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
    # print "s_grid"
    # print s_grid
    # record rewards to update the value table
    episodes_immediate_rewards = []
    num_conflicts_drivers = []
    curr_num_actions = []
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end

    startTime1 = tttt()
    route = []
    once_action = (0, chufa, chufa)
    route.append(once_action)

    rl_distance_cost = 0
    rl_crime_cost = 0
    rl_distance_single_cost = []
    rl_crime_single_cost = []

    dj_distance_cost = 0
    dj_crime_cost = 0
    dj_crime_single_cost = []
    dj_distance_single_cost = []

    a_distance_cost = 0
    a_crime_cost = 0
    a_crime_single_cost = []
    a_distance_single_cost = []

    axing_i = 0
    dj_i = 0
    rl_i = 0
    rl_end = 0
    axing = []
    dj = []
    rl_end_time = 0

    action_step = 5000  # 总移动步长
    now_step = 5000
    for ii in np.arange(EP_LEN + 1):
        # INPUT: state,  OUTPUT: action
        qvalues, action_idx, action_idx_valid, action_neighbor_idx, \
        action_tuple, action_starting_gridids = q_estimator.action(s_grid, context, epsilon)  # Q是tf.sess
        # a0
        # print "action_tuple"
        # print action_tuple
        # print "{}iters {}  action_tuple {} ".format(n_iter,ii,action_tuple)
        reward_cost = np.zeros(100)
        your_start_node = -1
        flag = 1
        number_max = -1
        start_node = route[-1][2]
        next_node = -1
        daoda_parameter1 = 10000

        if ii == 0:
            startTime2 = tttt()
            n = 100
            m = 10
            unvalid = 33

            endTime2 = tttt()

        if rl_end == 0:
            if len(route) - rl_i == 2:
                chushi = route[rl_i][2]
                mudi = route[rl_i + 1][2]
                rl_distance_cost += distance[chushi][mudi]
                rl_crime_cost += crime[chushi][mudi]
                rl_distance_single_cost.append(distance[chushi][mudi])
                rl_crime_single_cost.append(crime[chushi][mudi])
                rl_i = rl_i + 1
                if axing_i < len(axing) - 1:
                    a_distance_cost += distance[axing[axing_i]][axing[axing_i + 1]]
                    a_crime_cost += crime[axing[axing_i]][axing[axing_i + 1]]
                    a_distance_single_cost.append(distance[axing[axing_i]][axing[axing_i + 1]])
                    a_crime_single_cost.append(crime[axing[axing_i]][axing[axing_i + 1]])
                    axing_i = axing_i + 1

                if dj_i < len(dj) - 1:
                    dj_distance_cost += distance[dj[dj_i]][dj[dj_i + 1]]
                    dj_crime_cost += crime[dj[dj_i]][dj[dj_i + 1]]
                    dj_distance_single_cost.append(distance[dj[dj_i]][dj[dj_i + 1]])
                    dj_crime_single_cost.append(crime[dj[dj_i]][dj[dj_i + 1]])
                    dj_i = dj_i + 1

        # 根据每时刻的值来统计损耗

        for node in action_tuple:
            start = node[0]
            end = node[1]
            number = node[2]
            # print "number={}".format(number)
            # crime[start][end] = crime_old[start][end]
            beizeng = 1
            for xx in route:
                if end == xx[2] and [start] == xx[1]:
                    beizeng = 0.2
            reward1 = 10 * beizeng
            reward2 = danshe(crime[start][end])
            now_step1 = now_step
            if now_step < 0:
                now_step1 = now_step * (-1)
            reward3 = now_step1 / action_step / find_du(end, daoda, 10)
            first_crime = crime[start][end]
            # print "cost1 {}  cost2{}".format(cost1, reward2)
            # print "{}iters {}  action_tuple {} ".format(n_iter,ii,action_tuple)
            # crime[start][end] = 15000.0*pow(number,k-1)*pow(2.7,-number/t)/crime[start][end]/pow(t,k)
            if first_crime <= 50:
                crime[start][end] = first_crime * 0.9
            else:
                crime[start][end] = first_crime - (first_crime - 50) * 0.05
            reward_cost[end] += reward1  + reward2 + reward3
            if rl_end == 0:
                if ii >= 1:
                    if start == start_node:  # 判断是否是当前所要出发的点
                        tianjia = 1
                        if number > number_max and tianjia == 1:
                            number_max = number
                            next_node = end
                            before_node = start
                    if start != start_node and number_max > 0 and flag != 3:  # 判断有路线而且过了出发点判定
                        once_action = (ii, before_node, next_node)
                        route.append(once_action)
                        crime[before_node][next_node] = crime[before_node][next_node] * 100
                        now_step = now_step - distance[before_node][next_node]
                        # print "加了一个"
                        flag = 3
                        if next_node == daoda:
                            reward_cost[end] += daoda_parameter1
                            rl_end = 1
                            rl_end_time = tttt()

        # ONE STEP: r0
        next_state, r, info = env.step(reward_cost, action_tuple,
                                       2)  # next_state = self.get_observation()，这里的next_state就是剩余空闲driver和剩余订单两个array

        # 新加的在step里的info里

        # print "info"
        # print info

        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)  # 每个node的reward/driver数量获得一个平均值

        # print "immediate_reward"
        # print immediate_reward

        # a0
        action_mat = stateprocessor.to_action_mat(action_neighbor_idx)

        # s0
        # print "s_grid"
        # print s_grid
        # 这里已经发生了变化，s_grid_train的N，已经变成了转移driver的数量
        if (np.sum(action_starting_gridids) == 0):
            break;

        s_grid_train = stateprocessor.to_grid_state_for_training(s_grid,
                                                                 action_starting_gridids)  # action_starting_gridids这个应该是依次储存了每个点的driver转移的数量
        # print "s_grid_train"
        # print s_grid_train.shape
        # s1
        s_grid_next = stateprocessor.to_grid_next_states(s_grid_train, next_state, action_idx_valid,
                                                         env.city_time)  # action_idx_valid应该是依次储存了各个action 的合法的the index in nodes

        # Save transition to replay memory
        if ii != 0:
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(action_idx_valid_prev, immediate_reward)
            targets_batch = r_grid + gamma * target_estimator.predict(s_grid_next_prev)  # r+衰减系数哈哈

            # s0, a0, r1
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid_next_prev)

        state_mat_prev = s_grid_train
        action_mat_prev = action_mat
        context_prev = context
        s_grid_next_prev = s_grid_next
        action_idx_valid_prev = action_idx_valid

        # c1
        context = stateprocessor.compute_context(info[1])
        # s1
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
        # Sample a minibatch f rom the replay memory and update q network training method1
        if replay.curr_lens != 0:
            for _ in np.arange(20):
                fetched_batch = replay.sample()
                mini_s, mini_a, mini_target, mini_next_s = fetched_batch
                q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
                global_step += 1

        # Perform gradient descent update
        # book keeping
        global_step += 1
        all_rewards.append(r)
        batch_reward_gmv.append(r)
        order_response_rates.append(env.order_response_rate)
        num_conflicts_drivers.append(collision_action(action_tuple))
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        curr_num_actions.append(curr_num_action)

    duration = tttt() - startTime1 - (endTime2 - startTime2)
    rl_duration = (rl_end_time - startTime1) - (endTime2 - startTime2)
    print("Times takes:", duration)
    # distance_cost = 0
    # crime_cost = 0
    # crime_single_cost = []
    # for i in range(len(route)-1):
    #     chushi = route[i][1]
    #     mudi = route[i+1][1]
    #     distance_cost += distance[chushi][mudi]
    #     crime_cost += crime[chushi][mudi]
    #     crime_single_cost.append(crime[chushi][mudi])

    final_step = len(route)
    final_tuple = []
    for xx in range(final_step):
        final_tuple.append((route[xx][1],route[xx][2]))
    my_unique_list = list(set([i for i in final_tuple]))
    final_longth = len(my_unique_list)

    print "RL  -----  RL"
    print "第{}天,您的路线长度为{}".format(n_iter, len(route))
    print "第{}天,您的总巡逻路线数量为{}".format(n_iter, final_longth)
    print "第{}天,您的终点为{}".format(n_iter, route[-1][2])
    print "第{}天,您最后所剩step为{}".format(n_iter, now_step)
    print "第{}天,您的路线为{}".format(n_iter, route)
    print "您的距离系数损耗为{},风险系数损耗为{},每步距离损耗{},每步风险损耗{}".format(rl_distance_cost, rl_crime_cost, rl_distance_single_cost,
                                                           rl_crime_single_cost)
    # print "输入任意数字以确认"
    # suiji = input()

    one_data = [rl_distance_cost, rl_crime_cost, rl_distance_single_cost, rl_crime_single_cost, route, duration,
                rl_duration]

    data_save.append(one_data)

    episode_reward = np.sum(batch_reward_gmv[1:])
    episode_rewards.append(episode_reward)
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    order_response_rate_episode.append(n_iter_order_response_rate)
    episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
    episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))

    print "iteration {} ********* reward {} order{} conflicts {} drivers {}".format(n_iter, episode_reward,
                                                                                    order_response_rate_episode[-1],
                                                                                    episode_conflicts_drivers[-1],
                                                                                    episode_dispatched_drivers[-1])

    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,
                 episode_dispatched_drivers], open(log_dir + "results.pkl", "w"))

    if n_iter == 180:
        break

    # # training method 2.
    # for _ in np.arange(4000):
    #     fetched_batch = replay.sample()
    #     mini_s, mini_a, mini_target, mini_next_s = fetched_batch
    #     q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
    #     global_step += 1

    # update target Q network
    estimator_copy.make(sess)

    saver.save(sess, log_dir + "model.ckpt")

data = pd.DataFrame(data_save,
                    columns=('rl_distance_cost', 'rl_crime_cost', 'rl_distance_single_cost', 'rl_crime_single_cost',
                             'route', 'oneday_duration', 'rl_duration'))

data.to_excel('90risk_10_data1.00_0.06_new.xls')
