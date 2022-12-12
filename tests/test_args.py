from utils.tools import dotdict
from utils.tools import logger, add_periods_to_datetime
import torch

args = dotdict()
args.is_training = True
args.colab = True
args.linear2 = 64
args.conv_dim = 64
args.noise = "OU"
args.sigma = 0.25
args.theta = 0.25
args.dt = 0.002
args.x0 = None
args.batch_size = 3
args.gamma = 0.99
args.tau = 1e-2
args.max_size = 100000

args.conv1_out = 32  # 32
args.conv2_out = 32  # 64
args.conv3_out = 16  # 32
args.fc1_out = 64  # 128


args.critic_learning_rate = 1e-2
args.actor_learning_rate = 1e-3
args.chkpt_dir = "contents/outputs/ddpg"

args.episodes = 500
args.ratio = 0.8
args.benchmark_name = "UBAH"
args.compute_before = False
args.seq_len = 50

args.database_path = "outputs/coin_history.db"
args.granularity = 900
args.start_date = "2022-09-01-00-00"
args.end_date = "2022-10-20-00-00"

args.commission_rate_selling = 0.0025
args.commission_rate_purchasing = 0.0025

args.fill_database = True
args.with_test = True
args.resume = False

args.use_gpu = True
args.use_amp = False
args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)


args.d_model = 2048  # 512
args.hidden_size = 256
args.num_layers = 1
args.fc1_out = 1024  # 32
args.fc2_out = 32
args.fc3_out = 16
args.colab = False
args.bb = False
args.ba = False
args.ab = False

args.episodes = 500
args.ratio = 0.8
args.benchmark_name = "UBAH"
args.compute_before = False
args.seq_len = 50

args.database_path = "outputs/coin_history.db"
args.granularity = 900
args.start_date = "2022-10-01-00-00"
args.end_date = "2022-10-20-00-00"
args.commission_rate_selling = 0.0025
args.commission_rate_purchasing = 0.0025

# args.chkpt_dir = "contents/outputs/dpg"
args.chkpt_dir = "outputs/dpg"
args.d_model = 512
args.embed_type = "timef"
args.hidden_size = 256
args.num_layers = 1
args.fc1_out = 64
args.fc2_out = 32
args.dropout = 0.1

args.optim = "adam"
args.actor_learning_rate = 1e-3

args.batch_size = 32
args.shuffle = False
args.drop_last = False

args.use_gpu = False
args.use_amp = False
