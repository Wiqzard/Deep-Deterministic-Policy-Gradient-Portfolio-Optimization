from typing import Optional, List
import math

from tqdm import tqdm
from utils.tools import logger
from portfolio_manager.algorithms import *
from environment.environment import Environment
from agent.agent import Agent
from utils.constants import *
class Exp_Main:
    
    def __init__(self, args) -> None:
        self.args = args

        self.agent = self._set_agent()
        self.train_env = self._set_environment(flag="train")
        self.test_env = self._set_environment(flag="test")

        self.train_benchmark = self.get_benchmark(args.benchmark_name, flag="train")
        self.test_benchmark = self.get_benchmark(args.benchmark_name, flag="test")

        self.initial_value = START_VALUE 


    def _set_agent(self) -> None:
        return Agent(self.args, flag="train")


    def _set_environment(self, flag) -> None:
        if flag == "train":
            return Environment(self.args, flag="train")
        else:
            return Environment(self.args, flag="test") 


    def get_benchmark(self, model_name, flag: str="train"):
        args = self.args
        model_map = {
            "CRP": CRP, 
            "UBAH": UBAH, 
            "BCRP": BCRP, 
            "BestMarkowitz": BestMarkowitz,
            "UP": UP, 
            "Anticor": Anticor, 
            "OLMAR": OLMAR, 
            "RMR": RMR
        }
        if model_name not in model_map:
            logger.warn(f"No model named {model_name}")
            return
        model = model_map[model_name](args, flag=flag)
        weights = model.run(model.X)
        return model.calculate_returns(weights)


    def log_episode_result(self, episode:int, train_scores:List, test_scores:Optional[List]):
        """Logs the training result after each episode"""
        train_value = self.initial_value * math.exp(sum(train_scores))
        test_value = self.initial_value * math.exp(sum(test_scores)) if test_scores else 0
         
        logger.info(f"Episode: {episode} --- Train Value: {train_value:.2f} --- Test Value: {test_value:.2f}")
    

    def log_benchmark(self, in_dollar: bool=True) -> None:
        """Logs the benchmark of the train and test datasat. Specific algorithm is specified under args.bechmark_name"""
        total_return_train = self.train_benchmark.prod()
        total_return_test =  self.test_benchmark.prod()
        portfolio_value_train = self.initial_value * total_return_train if in_dollar else total_return_train 
        portfolio_value_test = self.initial_value * total_return_test if in_dollar else total_return_test
        logger.info(f"Start Training: \n Benchmark: {self.args.benchmark_name} --- Train Value: {portfolio_value_train:.2f} - Trading Periods: {self.train_env.num_steps} --- Test Value: {portfolio_value_test:.2f} - Trading Periods: {self.test_env.num_steps}")


    def train(self, with_test:bool=False, resume:bool=False) -> None:
        if resume:
            self.agent.load_models()

        test_steps = self.test_env.num_steps - (2 * self.args.seq_len + 1) if with_test else 0
        self.log_benchmark(in_dollar=True) 
        #if self.args.noise == "OU":
        #   self.agent.noise.reset()
        for episode in range(self.args.episodes):
            done = False
            train_scores = []
            obs, train_steps  = self.train_env.reset()
            total_steps = train_steps - (2 * self.args.seq_len + 1) + test_steps
            with tqdm(total=total_steps, leave=False, position=1) as pbar:
                while not done:
                    act = self.agent.choose_action(obs)
                    new_state, reward, done = self.train_env.step(act)
                    #train_score += reward
                    train_scores.append(reward)
                    self.agent.remember(obs, act, reward, new_state, int(done))
                    self.agent.learn()
                    obs = new_state
                    pbar.update(1)

                test_scores = self.backtest(bar=pbar) if with_test else None
            self.log_episode_result(episode=episode, train_scores=train_scores, test_scores=test_scores) 
            if episode % 5 == 0 and episode != 0:
                self.agent.save_models()


    def backtest(self, bar=None) -> None:
        score_history = []
        done = False
        obs, _= self.test_env.reset()
        while not done:
            act = self.agent.choose_action(obs, flag="test")
            new_state, reward, done = self.test_env.step(act)
            #score += reward
            score_history.append(reward)
            obs = new_state
            bar.update(1)
        return score_history 