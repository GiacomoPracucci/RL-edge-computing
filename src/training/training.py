import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from SAC.SAC import SAC
from SAC.replay_buffer import ReplayBuffer
from env.env import TrafficManagementEnv

def train_sac_agent(env, agent, buffer_size=1000000, batch_size=256, num_episodes=100, 
                    max_steps_per_episode=100, warm_up=512):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/SAC/' + current_time
    writer = SummaryWriter(train_log_dir)
    
    replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    total_rewards = []
    actor_losses = []
    critic1_losses = []
    critic2_losses = []
    alpha_losses = []
    steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic1_loss = 0
        episode_critic2_loss = 0
        episode_alpha_loss = 0

        for step in range(max_steps_per_episode):
            print("---------------------------------")
            print(f"Episode: {episode}, Step: {step}")
            print("---------------------------------")

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            replay_buffer.store(state, action, reward, next_state, done)

            if replay_buffer.size > warm_up:
                actor_loss, critic_1_loss, critic_2_loss, alpha_loss = agent.train(replay_buffer, batch_size)
                episode_actor_loss += actor_loss
                episode_critic1_loss += critic_1_loss
                episode_critic2_loss += critic_2_loss
                episode_alpha_loss += alpha_loss

                # scriviamo le loss nel SummaryWriter
                writer.add_scalar('Loss/Actor', actor_loss, steps)
                writer.add_scalar('Loss/Critic1',critic_1_loss, steps)
                writer.add_scalar('Loss/Critic2',critic_2_loss, steps)
                writer.add_scalar('Loss/Alpha',alpha_loss, steps)

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break
        
        writer.add_scalar('Reward', episode_reward, episode)

        total_rewards.append(episode_reward)
        actor_losses.append(episode_actor_loss / step)
        critic1_losses.append(episode_critic1_loss / step)
        critic2_losses.append(episode_critic2_loss / step)
        alpha_losses.append(episode_alpha_loss / step)

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

    writer.close() 
    #agent.save_weights("C:/Users/giaco/Desktop/local-git/Pesi reti")

    metrics = pd.DataFrame({'Reward': total_rewards, 'Actor Loss': actor_losses, 'Alpha Loss': alpha_losses,
                            'Critic 1 Loss': critic1_losses, 'Critic 2 Loss': critic2_losses})
    metrics.to_csv('metrics.csv')

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Critic 1 Loss'].rolling(10).mean(), label='Critic1 Loss')
    plt.plot(metrics['Critic 2 Loss'].rolling(10).mean(), label='Critic2 Loss')
    plt.title('Critic Networks Losses')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Actor Loss'].rolling(10).mean(), label='Actor Loss')
    plt.title('Actor Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Alpha Loss'].rolling(10).mean(), label='Alpha Loss')
    plt.title('Alpha Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Reward'].rolling(10).mean(), label='Reward')
    plt.title('Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()