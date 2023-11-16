#coding=utf-8
import pickle, sys
sys.path.append("../")

# from simulator.utilities import *
from alg_utility import *
from envs import *


################## Load data ###################################
dir_prefix = ""
current_time = time.strftime("%Y%m%d_%H-%M")
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
mkdir_p(log_dir)
print "log dir is {}".format(log_dir)

data_dir = dir_prefix + "dispatch_realdata/data_for_simulator/"
order_time_dist = [0.9,0.1]#原来是空的
order_price_dist = []
mapped_matrix_int = np.array([[1, -100, 3], [5, 4, 2], [6, 7, 8]])
M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = 8
idle_driver_location_mat = np.zeros((144, 8))

for ii in np.arange(144):
    time_dict = {}
    for jj in np.arange(M*N):  # num of grids
        time_dict[jj] = [2]
    order_num_dist.append(time_dict)
    idle_driver_location_mat[ii, :] = [2] * num_valid_grid
    

idle_driver_dist_time = [[10, 1] for _ in np.arange(144)]

n_side = 6
l_max = 2
order_time = [0.2, 0.2, 0.15,
                0.15,  0.1,  0.1,
                0.05, 0.04,  0.01]
order_price = [[10.17, 3.34],  # mean and std of order price when duration is 10 min
                [15.02, 6.90],  # mean and std of order price when duration is 20 min
                [23.22, 11.63],
                [32.14, 16.20],
                [40.99, 20.69],
                [49.94, 25.61],
                [58.98, 31.69],
                [68.80, 37.25],
                [79.40, 44.39]]

order_real = []
onoff_driver_location_mat = []
for tt in np.arange(144):
    order_real += [[1, 5, tt, 1, 13.2], [2, 6, tt, 1, 13.2], [3, 1, tt, 1, 13.2],
                    [5, 3, tt, 1, 13.2], [6, 2, tt, 1, 13.2], [7, 9, tt, 1, 13.2],
                    [9, 10, tt, 1, 13.2], [10, 6, tt, 1, 13.2], [9, 7, tt, 1, 13.2]]
    onoff_driver_location_mat.append([[-0.625,       2.92350389],
                                        [0.09090909,  1.46398452],
                                        [0.09090909,  2.36596622],
                                        [0.09090909, 2.36596622],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452],
                                        [0.09090909, 1.46398452]])


################## Initialize env ###################################
n_side = 6
GAMMA = 0.9


env = CityReal(mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price, l_max, M, N, n_side, 1, np.array(order_real), np.array(onoff_driver_location_mat))

state = env.reset_clean()

order_response_rates = []
T = 0
max_iter = 144
log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)


temp = np.array(env.target_grids) + env.M * env.N
#print temp
target_id_states = env.target_grids + temp.tolist()
#print target_id_states
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

target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")#初始化
sess.run(tf.global_variables_initializer())#不太清楚干了什么
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
epsilon_decay_steps=15
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)
for n_iter in np.arange(25):
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
    #print curr_state
    info = env.step_pre_order_assigin(curr_state)
    context = stateprocessor.compute_context(info)  #剩余空闲司机的矩阵
    #print context      
    curr_s = stateprocessor.utility_conver_states(curr_state)#curr_s是curr_state进行flatten操作以后，再去除非法节点的数据(example中第一个节点的driver和order数据)后的一维矩阵，在example中，curr_s是18-2=16个空间大小
    #print curr_s
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # record rewards to update the value table
    episodes_immediate_rewards = []
    num_conflicts_drivers = []
    curr_num_actions = []
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end
    for ii in np.arange(EP_LEN + 1):

        # INPUT: state,  OUTPUT: action
        qvalues, action_idx, action_idx_valid, action_neighbor_idx, \
        action_tuple, action_starting_gridids = q_estimator.action(s_grid, context, epsilon)
        # a0

        # ONE STEP: r0
        next_state, r, info = env.step(action_tuple, 2)#next_state = self.get_observation()，这里的next_state就是剩余空闲driver和剩余订单两个array

        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)#每个node的reward/driver数量获得一个平均值

        # a0
        action_mat = stateprocessor.to_action_mat(action_neighbor_idx)

        # s0
        #print "s_grid"
        #print s_grid
        #这里已经发生了变化，s_grid_train的N，已经变成了转移driver的数量
        s_grid_train = stateprocessor.to_grid_state_for_training(s_grid, action_starting_gridids)#action_starting_gridids这个应该是依次储存了每个点的driver转移的数量
        print "s_grid_train"
        print s_grid_train.shape
        # s1
        s_grid_next = stateprocessor.to_grid_next_states(s_grid_train, next_state, action_idx_valid, env.city_time)#action_idx_valid应该是依次储存了各个action 的合法的the index in nodes

        # Save transition to replay memory
        if ii != 0:
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(action_idx_valid_prev, immediate_reward)
            targets_batch = r_grid + gamma * target_estimator.predict(s_grid_next_prev)

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

    if n_iter == 24:
        break

    # # training method 2.
    # for _ in np.arange(4000):
    #     fetched_batch = replay.sample()
    #     mini_s, mini_a, mini_target, mini_next_s = fetched_batch
    #     q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
    #     global_step += 1

    # update target Q network
    estimator_copy.make(sess)


    saver.save(sess, log_dir+"model.ckpt")


