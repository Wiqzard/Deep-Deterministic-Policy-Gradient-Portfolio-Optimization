class Exp_Fed:
    def __init__(self, args) -> None:
        self.args = args

    def get_embedding(self): 

    def get_datalaoder(self):
         
    def get_optimizer(self):
        #if self.args.optim == "adam":
        optimizer = optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
        return  optimizer

    def learn(self):
        dataloader = self.get_dataloader("train")
        optimizer = self.get_optimizer()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GrandScaler()
            
        self.actor.train()
        for eposide in range(self.args.episodes):
            for idxs, scales, states, prev_actions, _ in tqdm(dataloader, total=len(dataloader), leave=True)
                states, _, state_time_marks, _ = states

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        actions = self.actor(states, state_time_marks, prev_actions) 
                else:
                    actions = self.actor(states, state_time_marks, prev_actions) 
                
                rewards = calculate_rewards_torch(scales, states, prev_actions, actions, self.args)
                reward = calculate_cummulative_reward(rewards) 

                if self.args.use_amp:
                    scaler.scale(reward).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    reward.backward()
                    optimizer.step()
                optimizer.zero_grad() 
            
            
            
            
            