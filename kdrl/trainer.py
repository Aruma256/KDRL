class GymTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    def train(self, episode, render=False):
        env, agent = self.env, self.agent
        result_reward = [0] * episode
        result_steps = [0] * episode
        for i in range(episode):
            state = env.reset()
            action = agent.start_episode(state)
            while True:
                if render:
                    env.render()
                state, reward, done, info = env.step(action)
                result_reward[i] += reward
                result_steps[i] += 1
                if not done:
                    action = agent.step(state, reward)
                    continue
                else:
                    agent.end_episode(state, reward)
                    break
        return {'reward' : result_reward, 'steps' : result_steps}
    def test(self, episode, render=False):
        env, agent = self.env, self.agent
        result_reward = [0] * episode
        result_steps = [0] * episode
        for i in range(episode):
            state = env.reset()
            while True:
                if render:
                    env.render()
                action = agent.select_best_action(state)
                state, reward, done, info = env.step(action)
                result_reward[i] += reward
                result_steps[i] += 1
                if done:
                    break
        return {'reward' : result_reward, 'steps' : result_steps}

