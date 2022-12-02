from typing import Optional, List
import math


from utils.tools import logger
from portfolio_manager.algorithms import *
from environment.environment import Environment
class Exp_Main:
    

    def __init__(self, args) -> None:
        self.args = args

        self.agent = self._set_agent()
        self.train_env = self._set_environment(flag="train")
        self.test_env = self._set_environment(flag="test")

        self.train_benchmark = self._get_benchmark(args.benchmark_name, flag="train")
        self.test_benchmark = self._get_benchmark(args.benchmark_name, flag="test")

       
         

    def _set_agent(self) -> None:
        pass

    def _set_environment(self, flag: str) -> None:
        if self.flag == "train":
            self.train_env = Environment(self.args, flag="train")
        else:
            self.test_env = Environment(self.args, flag="test") 

    def _one_episode(self):
        pass 

    def _get_benchmark(self, model_name, flag: str="train"):
        args = self.args
        if model_name == "CRP":
            model = CRP(flag=flag) 
        elif model_name == "UBAH":
            model = UBAH(flag=flag) 
        elif model_name == "BCRP":
            model = BCRP(flag=flag) 
        elif model_name == "BestMarkowitz":
            model = BestMarkowitz(flag=flag) 
        elif model_name == "UP":
            model = UP(flag=flag) 
        elif model_name == "Anticor":
            model = Anticor(flag=flag) 
        elif model_name == "OLMAR":
            model = OLMAR(flag=flag) 
        elif model_name == "RMR":
            model = RMR(flag=flag)
        else:
            logger.warn(f"No model named {model_name}")
        weights = model.run(model.X)
        return model.calculate_returns(weights)


    def log_episode_result(self, episode:int, train_scores:List, test_scores:Optional[List]):
        initial_value = 10000
        train_value = initial_value * math.exp(sum(train_scores))
        test_value = initial_value * math.exp(sum(test_scores))
         
        logger.info(f"Episode: {episode} --- Train Value: {train_value:.2f} --- Test Value: {test_value:.2f}")

    def train(self, with_test:bool=False, resume:bool=False) -> None:
        # sourcery skip: hoist-statement-from-loop
        if resume:
            self.agent.load_models()
        score_history = []
        logger.info(f"Start Training: \n Benchmark: {self.args.benchmark_name} --- Train Value: {self.train_benchmark:.2f} --- Test Value: {self.test_benchmark:.2f}")
        for episode in range(self.args.episodes):
            train_scores = 0
            done = False
            obs = self.train_env.reset()
            while not done:
                act = self.agent.choose_action(obs)
                new_state, reward, done = self.train_env.step(act)
                #train_score += reward
                score_history.append(reward)
                self.agent.remember(obs, act, reward, new_state, int(done))
                self.agent.learn()
                obs = new_state

            test_scores = self.backtest() if with_test else None
            self.log_episode_result(episode=episode, train_scores=train_scores, test_scores=test_scores) 
            if episode % 5 == 0:
                self.agent.save_models()

    def backtest(self) -> None:
        score_history = []
        done = False
        obs = self.test_env.reset()
        while not done:
            act = self.agent.choose_action(obs)
            new_state, reward, done = self.test_env.step(act)
            #score += reward
            score_history.append(reward)
            obs = new_state
        return score_history 