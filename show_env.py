#
# sites:
# https://gym.openai.com/docs/
#

import gym
import rubiks_cube_gym


def main():
    env = gym.make('ReversedAddition-v0', base=10)
    env.reset()

    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    main()
