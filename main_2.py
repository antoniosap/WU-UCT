#
# sites:
# https://gym.openai.com/docs/
#

def main():
    import gym
    env = gym.make('PongNoFrameskip-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()


if __name__ == "__main__":
    main()
