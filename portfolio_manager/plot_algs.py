import matplotlib.pyplot as plt


from portfolio_manager.algorithms import *

def plot_model(model_name=str) -> None:
    if model_name == "CRP":
        model = CRP() 
    elif model_name == "UBAH":
        model = UBAH() 
    elif model_name == "BCRP":
        model = BCRP() 
    elif model_name == "BestMarkowitz":
        model = BestMarkowitz() 
    elif model_name == "UP":
        model = UP() 
    elif model_name == "Anticor":
        model = Anticor() 
    elif model_name == "OLMAR":
        model = OLMAR() 
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


def benchmark(comission=None):
  ubah = UBAH()
  crp = CRP()
  bestmarkowitz = BestMarkowitz()
  up = UP()
  anticor = Anticor()
  olmar = OLMAR()
  rmr = RMR()

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