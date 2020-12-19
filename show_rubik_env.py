#
# sites:
# https://gym.openai.com/docs/
#

import gym
import rubiks_cube_gym


def main():
    env = gym.make('rubiks-cube-222-v0')
    env.reset(scramble="R U R' U' R' F R2 U' R' U' R U R' F'")

    for _ in range(4):
        env.render()
        print(env.step(1))
    env.render(render_time=0)
    env.close()


if __name__ == "__main__":
    main()
