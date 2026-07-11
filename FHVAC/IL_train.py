import os
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from replay_memory import ReplayMemory
from utils import load_trace
from IL import ILAgent
from Rule_Based import rule_based
import env

# QP = [23, 28, 33, 38, 43]
QP = [1, 1.2174, 1.4348, 1.6522, 1.8696]
FPS = [1.0, 0.5, 0.3333, 0.1667]
RE = [1.0, 0.4444, 0.1667, 0.0370]

RANDOM_SEED = 28
LEARNING_RATE = 1e-3
THRESHOLD = 0.3
S_INFO = 7
S_LEN = 8

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ALL_BW, ALL_NAME = load_trace('../CASVA/train_trace/')

def train_IL():
    torch.manual_seed(RANDOM_SEED)
    id = 0
    net_env = env.Environment(ALL_BW[id])

    model_IL = ILAgent().type(dtype)
    model_IL.train()
    optimizer_IL = optim.Adam(model_IL.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)

    epoch = 0
    episode_steps = 600
    batch_size = 32
    memory = ReplayMemory(1 * episode_steps)
    criterion = nn.CrossEntropyLoss()

    Q = 0
    bw_est = 0
    while True:
        states = []
        actions = []

        for step in range(episode_steps):
            states.append(state.unsqueeze(0))
            # IL action
            prob, _ = model_IL(state.unsqueeze(0).type(dtype))
            action = prob.multinomial(num_samples=1).detach()
            knob = int(action.squeeze().cpu().numpy())

            qp = knob // 16  # 0 to 10, because 80 // 16 = 5
            remainder = knob % 16
            skip = remainder // 4  # 0 to 3
            re = remainder % 4  # 0 to 3
            IL_action = (re, skip, qp)

            # expert action
            expert_action = rule_based(Q, bw_est)

            # Compare actions
            action = compare(IL_action, expert_action)
            # actions.append(torch.tensor([action[2] * 16 + action[1] * 4 + action[0]]))

            # Execute the action and collect data
            bw_est, bw, latency, buffer_size, size, dynamics, f1, Q, end_of_video = net_env.get_video_chunk(action[2], action[1], action[0])

            action = action[2] * 16 + action[1] * 4 + action[0]
            actions.append(torch.tensor([action]))
            if end_of_video:
                state = np.zeros((S_INFO, S_LEN))
                state = torch.from_numpy(state)
                id = id + 1
                net_env.cooked_bw = ALL_BW[id % 4]
                net_env.start = np.random.randint(2, len(net_env.cooked_bw))
                net_env.video_start_shoot = net_env.start - 2
                net_env.start_shoot_fix = net_env.video_start_shoot
                Q = 0
                bw_est = 0
                break

            state = np.roll(state, -1, axis=1)
            state[0, -1] = bw
            state[1, -1] = latency
            state[2, -1] = buffer_size
            state[3, -1] = qp
            state[4, -1] = skip
            state[5, -1] = re
            state[6, -1] = Q
            state = torch.from_numpy(state)
        memory.push([states, actions])
        # train
        optimizer_IL.zero_grad()
        batch_states, batch_actions = memory.sample(batch_size)

        _, predictions = model_IL(batch_states.type(dtype))
        loss = criterion(predictions, batch_actions.type(torch.long).to(predictions.device))
        loss.backward()
        optimizer_IL.step()
        epoch += 1
        memory.clear()
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
            os.makedirs("Results/IL", exist_ok=True)
            model_path = f"Results/IL/IL_{epoch}.model"
            torch.save(model_IL.state_dict(), model_path)
        if epoch > 20000:
            break

def compare(IL_action, expert_action, threshold=THRESHOLD):
    a = (RE[IL_action[0]], FPS[IL_action[1]], QP[IL_action[2]])
    b = (RE[expert_action[0]], FPS[expert_action[1]], QP[expert_action[2]])

    IL_vector = torch.tensor(a, dtype=torch.float32)
    expert_vector = torch.tensor(b, dtype=torch.float32)

    if torch.norm(IL_vector - expert_vector) <= threshold:
        return IL_action
    else:
        return expert_action


if __name__ == '__main__':
    train_IL()