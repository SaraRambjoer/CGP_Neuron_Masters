import gym
from HelperClasses import randchoice
# TODO use template for an abstract problem class
class PoleBalancingProblem():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.actions = [0, 1] # 0 <- left, 1 -> right
        self.current_observation = self.env.reset()
        # Action space:

        # cart position
        # cart velocity
        # pole angle
        # pole angular velocity

        self.input_arity = 4
        self.output_arity = 1

    def get_problem_instance(self):
        return self.current_observation

    # Target is to minimize this
    def error(self, problem, solution, logger):
        action = solution[0]
        if action not in [0, 1]:
            self.current_observation, reward, done, _ = self.env.step(randchoice([0, 1]))
            if done:
                self.current_observation = self.env.reset()
            return 1, False
        self.current_observation, reward, done, _ = self.env.step(int(action))
        if done:
            self.current_observation = self.env.reset()

        return 1-reward, True
        #solution = solution[0]
        #return error, valid output bool

    def get_reward(self, error):
        return 1 - error



