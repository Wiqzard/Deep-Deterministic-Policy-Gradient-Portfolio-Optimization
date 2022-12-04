import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.constants import *
from portfolio_manager.algorithms import *
from agent.noise import OUActionNoisePlain


def plot_model(args, model_name=str, flag="train") -> None:
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
    # weights
    returns = model.calculate_returns()
    # sharpe = model.sharpe()
    # print(np.prod(returns.values))
    ##print(sharpe)
    model.plot_portfolio_value(r=returns)
    model.plot_portfolio_weights()


def plot_portfolio_algos(args, flag="train", commission=None):
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
        # plt.set_label(modelk)
        # plt.label
        print(
            f" {algo.name} Total Return: {round(np.prod(returns), 4)} Sharpe Ratio(d): {round(sharpe, 2)} "
        )
    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.xlabel(f"Periods [{int(ubah.state_space.granularity / 60)} min]")
    plt.legend()
    plt.show()


save_path = None
# def plot_asset_values(env, scale: bool=True, difference: bool=False, save_path=None) -> None:
#  plt.figure(figsize=(20, 5), dpi=80)
#  plt.title("Relative Asset Values")
#  closes = env.state_space.filled_feature_matrices[0].iloc[:, 1:]
#  if scale:
#    scaler = MinMaxScaler() #MinMaxScaler()#StandardScaler()
#    scaler.fit(closes)
#    closes = scaler.transform(closes)
#  for i in range(8):
#    coin = env.state_space.coins[i].split("-")[0]
#    data = closes[:, i]
#    if difference:
#      data = np.diff(data, n=1, axis=0)
#    #plt.plot(states[0][0][0,:,i]/states[0][0][0,0,i], label=coin) #Close price
#    plt.plot(data, label=coin)
#  #plt.plot(actions)
#  plt.grid(b=None, which='major', axis='y', linestyle='--')
#  plt.xlabel(f"Periods [{int(env.state_space.granularity / 60)} min]")
#  plt.legend()
#  if not save_path==None:
#    plt.savefig(save_path)
#  plt.show()


from matplotlib.widgets import Button, CheckButtons

# Define the function to plot the asset values


def plot_asset_values(
    price_matrix,
    granularity,
    scale: bool = True,
    difference: bool = False,
    save_path=None,
) -> None:
    plt.figure(figsize=(20, 5), dpi=80)
    plt.title("Asset Values")
    closes = price_matrix.iloc[:, 1:].values
    if scale:
        scaler = MinMaxScaler()  # MinMaxScaler()#StandardScaler()
        scaler.fit(closes)
        closes = scaler.transform(closes)
    lines = []
    for i in range(8):
        coin = COINS[i]
        data = closes[:, i]
        if difference:
            data = np.diff(data, n=1, axis=0)
        # Create the line
        (line,) = plt.plot(data, label=coin)
        lines.append(line)

    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.xlabel(f"Periods [{int(granularity / 60)} min]")
    plt.legend()
    plt.show()


def plot_weights_last_backtest(action_history, k=1):
    k = 10
    plt.figure(figsize=(20, 5), dpi=80)
    plt.title("Portfolio Wheigts")

    for i in range(NUM_ASSETS):
        coin = COINS[i]
        plt.plot(range(0, len(action_history), k), action_history[::k, i], label=coin)

    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.axhline(y=0.125, color="black")
    plt.legend()


def plot_value_last_backtest(reward_history, k=1) -> None:
    k = 10
    plt.figure(figsize=(20, 5), dpi=80)
    plt.title("Portfolio Value")
    portfolio_value = [
        START_VALUE * np.prod(reward_history[:i]) for i in range(len(reward_history))
    ]
    plt.plot(range(0, len(reward_history), k), portfolio_value[::k])

    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.legend()


def plot_results_episodes(end_scores) -> None:
    plt.figure(figsize=(20, 5), dpi=80)
    plt.title("Portfolio Value")

    plt.plot(end_scores)
    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.legend()


def plot_weight_changes_episodes(action_histories, k=1) -> None:
    averages = []

    for action_history in action_histories:
        average = np.average(action_history, axis=1)
        averages.append(average)
    averages = np.array(averages)
    plt.figure(figsize=(20, 5), dpi=80)
    plt.title("Average Portfolio Wheigts Per Episode")

    for i in range(NUM_ASSETS):
        coin = COINS[i]
        plt.plot(
            range(0, len(averages), k),
            averages[::k, i],
            label=coin,
        )

    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.axhline(y=0.125, color="black")
    plt.legend()


# def plot_ou_action_noise(mu, sigma, theta, x0, dt, steps):
#    ou = OUActionNoisePlain(mu=mu, theta=theta, sigma=sigma, dt=dt, x0=x0)
#    outputs = [ou() for _ in range(steps)]
#    plt.figure(figsize=(20, 5), dpi=80)
#    plt.title("mu: {mu}, sigma: {sigma}, theta: {theta}, x0: {x0}, dt: {dt}")
#    plt.plot(outputs)
#    plt.grid(b=None, which="major", axis="y", linestyle="--")
#    plt.axhline(y=0.125, color="black")


from ipywidgets import interact


def plot_ou_action_noise(dim_mu, sigma, theta, x0, dt, steps):
    mu = np.zeros(dim_mu)
    ou = OUActionNoisePlain(mu=mu, theta=theta, sigma=sigma, dt=dt, x0=x0)
    outputs = [ou() for _ in range(steps)]
    plt.figure(figsize=(20, 5), dpi=80)
    plt.title("mu: {mu}, sigma: {sigma}, theta: {theta}, x0: {x0}, dt: {dt}")
    plt.plot(outputs)
    plt.grid(b=None, which="major", axis="y", linestyle="--")
    plt.axhline(y=0.125, color="black")


# Use the interact() function to automatically update the plot
# Note that the mu parameter has been removed from the function call
# interact(plot_ou_action_noise,
#         sigma=(0.0, 1.0, 0.01),
#         theta=(0.0, 1.0, 0.01),
#         x0=(0.0, 1.0, 0.01),
#         dt=(0.0, 1.0, 0.01),
#         steps=(1, 100, 1))
