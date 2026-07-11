import os
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Process, Queue, set_start_method
from torch.autograd import Variable
import logging
from utils import load_trace, C_R
from PPO import Actor, Critic
from replay_memory import ReplayMemory
from test import valid
import env


# --------------------------------
A_DIM = 80
# --------------------------------
RANDOM_SEED = 28
S_INFO = 8  # bw, delay, buffer, qp, skip, r, segment_size, dynamics
S_LEN = 8  # past 8
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
UPDATE_INTERVAL = 100
L = 2

SUMMARY_DIR = 'Results/'
LOG_FILE = 'Results/log'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('train_trace/')


def train_ppo():
    os.makedirs('Results', exist_ok=True)
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    with open(LOG_FILE + '_test', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        id = 0
        net_env = env.Environment(ALL_BW[id])

        model_actor = Actor().type(dtype)
        model_critic = Critic().type(dtype)

        model_actor.train()
        model_critic.train()

        optimizer_actor = optim.Adam(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.Adam(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)

        state = np.zeros((S_INFO, S_LEN))
        state = torch.from_numpy(state)
        epoch = 0
        exploration_size = 1
        episode_steps = 1800
        # last_knob = 40
        update_num = 1
        batch_size = 256
        gamma = 0.99
        gae_param = 0.97
        c = 3
        clip = 0.2
        ent_coeff = 0.9
        memory = ReplayMemory(exploration_size * episode_steps)

        last_buff = 0

        while True:
            for explore in range(exploration_size):
                states = []
                actions = []
                rewards = []
                values = []
                returns = []
                advantages = []

                for step in range(episode_steps):
                    prob = model_actor(state.unsqueeze(0).type(dtype))

                    action = prob.multinomial(num_samples=1).detach()
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                    values.append(v)
                    knob = int(action.squeeze().cpu().numpy())
                    actions.append(torch.tensor([action]))
                    states.append(state.unsqueeze(0))

                    qp = knob // 16  # 0 to 10, because 80 // 16 = 5
                    remainder = knob % 16
                    skip = remainder // 4  # 0 to 3
                    re = remainder % 4  # 0 to 3

                    bw, latency, buffer_size, size, dynamics, f1, end_of_video = net_env.get_video_chunk(qp, skip, re)
                    reward = f1 - 0.1 * latency
                    rewards.append(reward)

                    last_buff = buffer_size
                    # last_knob = knob

                    if end_of_video:
                        state = np.zeros((S_INFO, S_LEN))
                        state = torch.from_numpy(state)
                        # last_knob = 40
                        id = id + 1
                        net_env.cooked_bw = ALL_BW[id % 4]
                        net_env.start = np.random.randint(2, len(net_env.cooked_bw))
                        net_env.video_start_shoot = net_env.start - 2
                        net_env.start_shoot_fix = net_env.video_start_shoot

                        last_buff = 0
                        break

                    state = np.roll(state, -1, axis=1)
                    state[0, -1] = bw
                    state[1, -1] = latency
                    state[2, -1] = buffer_size
                    state[3, -1] = qp
                    state[4, -1] = skip
                    state[5, -1] = re
                    state[6, -1] = size
                    state[7, -1] = dynamics
                    state = torch.from_numpy(state)

                R = torch.zeros(1, 1)
                if end_of_video == False:
                    v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                    R = v.data
                # ========================================================================
                values.append(Variable(R))
                R = Variable(R)
                A = Variable(torch.zeros(1, 1))
                for i in reversed(range(len(rewards))):
                    td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                    A = float(td) + gamma * gae_param * A
                    advantages.append(A)
                    R = A + values[i]
                    returns.append(R)
                advantages.reverse()
                returns.reverse()
                memory.push([states, actions, returns, advantages])
            model_actor_old = Actor().type(dtype)
            model_actor_old.load_state_dict(model_actor.state_dict())
            model_critic_old = Critic().type(dtype)
            model_critic_old.load_state_dict(model_critic.state_dict())

            for update_step in range(update_num):
                model_actor.zero_grad()
                model_critic.zero_grad()

                batch_states, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)

                # --------------------------------------------------------------------------------
                # Calculate policy loss
                probs_old = model_actor_old(batch_states.type(dtype).detach())
                probs_new = model_actor(batch_states.type(dtype))
                ratio = calculate_prob_ratio(probs_new, probs_old, batch_actions)

                advantages = batch_advantages.type(dtype)
                indicator_neg = (advantages < 0).float()
                indicator_pos = (advantages >= 0).float()
                scaled_advantages = c * advantages

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
                l_p = torch.min(surr1, surr2)

                loss_api = -torch.mean(indicator_neg * torch.max(scaled_advantages, l_p) + indicator_pos * l_p)

                entropy = calculate_entropy(probs_new)
                loss_ent = -ent_coeff * entropy
                total_loss_api = loss_api + loss_ent
                # -----------------------------------------------------------------------------------

                # Update critic networks
                v_pre = model_critic(batch_states.type(dtype))
                v_pre_old = model_critic_old(batch_states.type(dtype).detach())
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                total_loss_api.backward()
                loss_value.backward()
                optimizer_actor.step()
                optimizer_critic.step()
                # --------------------------------------------------------------------------------
            # test and save the model
            epoch += 1
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(loss_api.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()))

            if epoch % UPDATE_INTERVAL == 0:
                logging.info("Model saved in file")
                valid(model_actor, epoch, test_log_file)
                ent_coeff = 0.95 * ent_coeff
                if epoch >= 50000:
                    break


def calculate_entropy(probs):
    """Calculate the entropy of the policy distribution."""
    log_probs = torch.log(probs + 1e-6)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy


def calculate_prob_ratio(new_probs, old_probs, actions):
    """Calculate the ratio of new and old probabilities for selected actions."""
    new_action_probs = torch.gather(new_probs, dim=1, index=actions.unsqueeze(1).type(dlongtype))
    old_action_probs = torch.gather(old_probs, dim=1, index=actions.unsqueeze(1).type(dlongtype))
    ratio = new_action_probs / (old_action_probs + 1e-6)
    return ratio


if __name__ == '__main__':
    train_ppo()


