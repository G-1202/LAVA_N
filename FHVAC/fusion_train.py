import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from utils import load_trace, C_R
from PPO import Actor
from IL import ILAgent
from Fusion import FusionActor, FusionCritic
from replay_memory import ReplayMemory
from test import valid_fusion
import env

A_DIM = 80

# QP = [23, 28, 33, 38, 43]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
FPS = [1.0, 0.5, 0.3333, 0.1667]
RE = [1.0, 0.4444, 0.1667, 0.0370]
RANDOM_SEED = 28

S_RL_INFO = 8
S_IL_INFO = 7
S_LEN = 8

RL_LEARNING_RATE = 1e-3
IL_LEARNING_RATE = 5e-4

SUMMARY_DIR = 'Results_f/'
LOG_FILE = 'Results_f/log'

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('../CASVA/train_trace/')

RL = 'Results/RL/RL.model'
IL= 'Results/IL/IL.model'

def train_fusion_model():
    os.makedirs('Results_f', exist_ok=True)
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)
    with open(LOG_FILE + '_test', 'w') as test_log_file:
        torch.manual_seed(RANDOM_SEED)
        id = 0
        net_env = env.Environment(ALL_BW[id])

        RLmodel = Actor().type(dtype)
        RLmodel.train()
        RLmodel.load_state_dict(torch.load(RL))

        ILmodel = ILAgent().type(dtype)
        ILmodel.train()
        ILmodel.load_state_dict(torch.load(IL))

        fusion_actor = FusionActor().type(dtype)
        fusion_critic = FusionCritic().type(dtype)
        fusion_actor.train()
        fusion_critic.train()
        optimizer_actor = torch.optim.Adam(fusion_actor.parameters(), lr=1e-4)
        optimizer_critic = torch.optim.Adam(fusion_critic.parameters(), lr=1e-4)
        # optimizer_rl = optim.Adam(RLmodel.parameters(), lr=RL_LEARNING_RATE)
        optimizer_il = optim.Adam(ILmodel.parameters(), lr=IL_LEARNING_RATE)

        state_RL = np.zeros((S_RL_INFO, S_LEN))
        state_RL = torch.from_numpy(state_RL)
        state_IL = np.zeros((S_IL_INFO, S_LEN))
        state_IL = torch.from_numpy(state_IL)

        epoch = 0
        episode_steps = 1800
        batch_size = 256
        gamma = 0.98
        gae_param = 0.97
        clip = 0.2
        ent_coeff = 0.9
        memory = ReplayMemory(1 * episode_steps)

        while True:
            states_RL = []
            states_IL = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []

            for step in range(episode_steps):
                # ------------------------------------------
                frl = RLmodel.get_feature(state_RL.unsqueeze(0).type(dtype))
                fil = ILmodel.get_feature(state_IL.unsqueeze(0).type(dtype))
                prob = fusion_actor(frl, fil)
                action = prob.multinomial(num_samples=1).detach()
                v = fusion_critic(frl, fil).detach().cpu()
                values.append(v)
                knob = int(action.squeeze().cpu().numpy())
                actions.append(torch.tensor([action]))
                states_RL.append(state_RL.unsqueeze(0))
                states_IL.append(state_IL.unsqueeze(0))

                qp = knob // 16  # 0 to 10, because 80 // 16 = 5
                remainder = knob % 16
                skip = remainder // 4  # 0 to 3
                re = remainder % 4  # 0 to 3

                _, bw, latency, buffer_size, size, dynamics, f1, Q, end_of_video = net_env.get_video_chunk(qp, skip, re)
                reward = f1 - 0.1*latency

                rewards.append(reward)

                if end_of_video:
                    state_RL = np.zeros((S_RL_INFO, S_LEN))
                    state_RL = torch.from_numpy(state_RL)
                    state_IL = np.zeros((S_IL_INFO, S_LEN))
                    state_IL = torch.from_numpy(state_IL)
                    id = id + 1
                    net_env.cooked_bw = ALL_BW[id % 4]
                    net_env.start = np.random.randint(2, len(net_env.cooked_bw))
                    net_env.video_start_shoot = net_env.start - 2
                    net_env.start_shoot_fix = net_env.video_start_shoot

                    last_buff = 0
                    break

                state_RL = np.roll(state_RL, -1, axis=1)
                state_RL[0, -1] = bw
                state_RL[1, -1] = latency
                state_RL[2, -1] = buffer_size
                state_RL[3, -1] = qp
                state_RL[4, -1] = skip
                state_RL[5, -1] = re
                state_RL[6, -1] = size
                state_RL[7, -1] = dynamics
                state_RL = torch.from_numpy(state_RL)

                state_IL = np.roll(state_IL, -1, axis=1)
                state_IL[0, -1] = bw
                state_IL[1, -1] = latency
                state_IL[2, -1] = buffer_size
                state_IL[3, -1] = qp
                state_IL[4, -1] = skip
                state_IL[5, -1] = re
                state_IL[6, -1] = Q
                state_IL = torch.from_numpy(state_IL)

            R = torch.zeros(1, 1)
            if end_of_video == False:
                v = fusion_critic(frl, fil).detach().cpu()
                R = v.data
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
            memory.push([states_RL, states_IL, actions, returns, advantages])

            # update
            fusion_actor_old = FusionActor().type(dtype)
            fusion_actor_old.load_state_dict(fusion_actor.state_dict())
            fusion_critic_old = FusionCritic().type(dtype)
            fusion_critic_old.load_state_dict(fusion_critic.state_dict())

            for flag in range(2):
                batch_states_RL, batch_states_IL, batch_actions, batch_returns, batch_advantages = memory.sample(batch_size)
                frl = RLmodel.get_feature(batch_states_RL.type(dtype))
                fil = ILmodel.get_feature(batch_states_IL.type(dtype))
                probs_old = fusion_actor_old(frl.type(dtype).detach(), fil.type(dtype).detach())
                probs_new = fusion_actor(frl.type(dtype), fil.type(dtype))
                ratio = calculate_prob_ratio(probs_new, probs_old, batch_actions)
                advantages = batch_advantages.type(dtype)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
                loss_api = -torch.mean(torch.min(surr1, surr2))
                entropy = calculate_entropy(probs_new)
                loss_ent = -ent_coeff * entropy
                total_loss_api = loss_api + loss_ent

                v_pre = fusion_critic(frl.type(dtype), fil.type(dtype))
                v_pre_old = fusion_critic_old(frl.type(dtype).detach(), fil.type(dtype).detach())
                vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2
                v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-clip, clip)
                vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
                loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))

                # optimizer_rl.zero_grad()
                optimizer_il.zero_grad()
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                total_loss_api.backward(retain_graph=True)
                loss_value.backward()
                # total_loss.backward()
                optimizer_actor.step()
                optimizer_critic.step()
                # optimizer_rl.step()
                optimizer_il.step()
            epoch += 1
            memory.clear()
            logging.info('Epoch: ' + str(epoch) +
                         ' Avg_policy_loss: ' + str(loss_api.detach().cpu().numpy()) +
                         ' Avg_value_loss: ' + str(loss_value.detach().cpu().numpy()) +
                         ' Avg_entropy_loss: ' + str(A_DIM * loss_ent.detach().cpu().numpy()))
            if epoch % 100 == 0:
                logging.info("Model saved in file")
                valid_fusion(RLmodel, ILmodel, fusion_actor, epoch, test_log_file)
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
    train_fusion_model()





