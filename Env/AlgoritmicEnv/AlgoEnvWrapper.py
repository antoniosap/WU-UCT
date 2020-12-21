#
# 20.12.2020 - custom env wrapper for ReversedAddition test
#
import gym
from gym import spaces
from copy import deepcopy
from collections import deque


# To allow easily extending to other tasks, we built a wrapper on top of the 'real' environment.
class AlgoEnvWrapper:
    def __init__(self, env_name, max_episode_length=0):
        self.env_name = env_name
        self.env_type = None
        self.env = gym.make('ReversedAddition-v0', base=10)
        self.env = FrameStack(self.env, 4)
        self.env.reset()
        self.action_n = self.env.action_space[0].n
        self.max_episode_length = self.env._max_episode_steps if max_episode_length == 0 else max_episode_length
        self.current_step_count = 0
        self.since_last_reset = 0

    def reset(self):
        state = self.env.reset()

        self.current_step_count = 0
        self.since_last_reset = 0

        return state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)

        self.current_step_count += 1
        if self.current_step_count >= self.max_episode_length:
            done = True

        self.since_last_reset += 1

        return next_state, reward, done

    def checkpoint(self):
        pass
        # return deepcopy(self.env.clone_full_state()), self.current_step_count

    def restore(self, checkpoint):
        if self.since_last_reset > 20000:
            self.reset()
            self.since_last_reset = 0

        self.env.restore_full_state(checkpoint[0])

        self.current_step_count = checkpoint[1]

        return self.env.get_state()

    def render(self):
        self.env.render()

    def capture_frame(self):
        self.recorder.capture_frame()

    def store_video_files(self):
        self.recorder.write_metadata()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    def get_action_n(self):
        return self.action_n

    def get_max_episode_length(self):
        return self.max_episode_length


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        # shp = env.observation_space.shape
        # self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob / 255.0)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob / 255.0)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        pass
        # assert len(self.frames) == self.k
        # return LazyFrames(list(self.frames))

    def clone_full_state(self):
        state_data = self.unwrapped.clone_full_state()
        frame_data = self.frames.copy()

        full_state_data = (state_data, frame_data)

        return full_state_data

    def restore_full_state(self, full_state_data):
        state_data, frame_data = full_state_data

        self.unwrapped.restore_full_state(state_data)
        self.frames = frame_data.copy()

    def get_state(self):
        pass
        # return self._get_ob()

