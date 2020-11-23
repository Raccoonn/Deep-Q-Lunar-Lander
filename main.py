from tf_dqn import Agent
import numpy as np
import time
import gym
# from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    lr = 0.0005
    n_games = 50
    agent = Agent(gamma=0.99, epsilon=0.10, alpha=lr, input_dims=8,
                  n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.0,
                  fname='dqn_model.h5')

    agent.load_model()
    scores = []
    eps_history = []

    # env = wrappers.Monitor(env, "tmp/lunar-lander-6", video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        print('\nPlaying a game...\n')
        start_time = time.time()
        while not done:

            env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_

            # agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('Episode:', i,'Score: %.2f' % score, 'Average Score: %.2f' % avg_score)
        print('Elapsed Time:  %0.3f \n' % (time.time() - start_time))
        
        # agent.save_model()
