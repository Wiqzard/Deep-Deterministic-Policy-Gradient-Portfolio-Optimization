import matplotlib.pyplot as plt


from portfolio_manager.algorithms import *

def plot_model(args, model_name=str,  flag="train") -> None:
    if model_name == "CRP":
        model = CRP(args=args, flag=flag) 
    elif model_name == "UBAH":
        model = UBAH(args=args, flag=flag)
    elif model_name == "BCRP":
        model = BCRP(args=args, flag=flag)
    elif model_name == "BestMarkowitz":
        model = BestMarkowitz(args=args, flag=flag)
    elif model_name == "UP":
        model = UP(args=args, flag=flag)
    elif model_name == "Anticor":
        model = Anticor(args=args, flag=flag)
    elif model_name == "OLMAR":
        model = OLMAR(args=args, flag=flag)
    elif model_name == "RMR":
        model = RMR()
    weights = model.run(model.X)
    #weights
    returns = model.calculate_returns()
    #sharpe = model.sharpe()
    #print(np.prod(returns.values))
    ##print(sharpe)
    model.plot_portfolio_value(r=returns)
    model.plot_portfolio_weights()


def plot_portfolio_algos(args, flag="train", comission=None):
  ubah = UBAH(args=args, flag=flag)
  crp = CRP(args=args, flag=flag)
  olmar = OLMAR(args=args, flag=flag)
  bestmarkowitz = BestMarkowitz(args=args, flag=flag)
  up = UP(args=args, flag=flag)
  anticor = Anticor(args=args, flag=flag)
  rmr = RMR(args=args, flag=flag)

  plt.figure(figsize=(20, 5), dpi=80)
  plt.title("Portfolio Values")

  algos = [ubah, crp, bestmarkowitz, up, anticor, olmar, rmr]
  for algo in algos:
    weights = algo.run(algo.X)
    returns = algo.calculate_returns(weights)
    sharpe = algo.sharpe(returns) 
    algo.plot_portfolio_value(r=returns)
    #plt.set_label(modelk)
    #plt.label 
    print(f" {algo.name} Total Return: {round(np.prod(returns), 4)} Sharpe Ratio(d): {round(sharpe, 2)} ")
  plt.grid(b=None, which='major', axis='y', linestyle='--')
  plt.xlabel(f"Periods [{int(ubah.state_space.granularity / 60)} min]")
  plt.legend()
  plt.show()