#!/usr/bin/env python3

# Imports
from logbook import INFO, WARNING, DEBUG

import warnings
warnings.filterwarnings("ignore") # suppress h5py deprecation warning

import numpy as np
import os
import backtrader as bt

from btgym.research.casual_conv.strategy import CasualConvStrategyMulti
from btgym.research.casual_conv.networks import conv_1d_casual_attention_encoder

from btgym.algorithms.launcher.base import Launcher
from btgym.algorithms.aac import A3C
from btgym.algorithms.policy import StackedLstmPolicy

from btgym import MultiDiscreteEnv
from btgym.datafeed.casual import BTgymCasualDataDomain
from btgym.datafeed.multi import BTgymMultiData

# The engine we will pass to the environment.

engine = bt.Cerebro()

num_features = 16

engine.addstrategy(
    CasualConvStrategyMulti,
    cash_name='EUR',  # just a naming for cash asset
    start_cash=2000,
    commission=0.0001, 
    leverage=10.0,
    asset_names={'USD', 'CHF'},  # that means we use JPY and GBP as information data lines only
    drawdown_call=10,
    target_call=10,
    skip_frame=10,
    gamma=0.99,
    state_ext_scale = {  # strategy specific CWT preprocessing params:
        'USD': np.linspace(1, 2, num=num_features),
        'GBP': np.linspace(1, 2, num=num_features),
        'CHF': np.linspace(1, 2, num=num_features),
        'JPY': np.linspace(5e-3, 1e-2, num=num_features),
    },
    cwt_signal_scale=4e3,
    cwt_lower_bound=4.0, 
    cwt_upper_bound=90.0,
    reward_scale=7, 
    order_size={
        'CHF': 1000,
        #'GBP': 1000,
        #'JPY': 1000,
        'USD': 1000,
    },
)

data_config = {
    'USD': {'filename': './data/DAT_ASCII_EURUSD_M1_2017.csv'},
    'GBP': {'filename': './data/DAT_ASCII_EURGBP_M1_2017.csv'},
    'JPY': {'filename': './data/DAT_ASCII_EURJPY_M1_2017.csv'},
    'CHF': {'filename': './data/DAT_ASCII_EURCHF_M1_2017.csv'},
}

dataset = BTgymMultiData(
    data_class_ref=BTgymCasualDataDomain,
    data_config=data_config,
    trial_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 30, 'hours': 0, 'minutes': 0},
        start_00=False,
        time_gap={'days': 15, 'hours': 0},
        test_period={'days': 7, 'hours': 0, 'minutes': 0},
        expanding=True,
    ),
    episode_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 2, 'hours': 23, 'minutes': 55},
        start_00=False,
        time_gap={'days': 2, 'hours': 15},
    ),
    frozen_time_split={'year': 2017, 'month': 3, 'day': 1},
)
#########################

env_config = dict(
    class_ref=MultiDiscreteEnv, 
    kwargs=dict(
        dataset=dataset,
        engine=engine,
        render_modes=['episode'],
        render_state_as_image=True,
        render_size_episode=(12,16),
        render_size_human=(9, 4),
        render_size_state=(11, 3),
        render_dpi=75,
        port=5000,
        data_port=4999,
        connect_timeout=90,
        verbose=0,
    )
)

cluster_config = dict(
    host='127.0.0.1',
    port=12230,
    num_workers=6,  # Set according CPU's available or so
    num_ps=1,
    num_envs=1,
    log_dir=os.path.expanduser('~/tmp/multi_discrete'),
)
policy_config = dict(
    class_ref=StackedLstmPolicy,
    kwargs={
        'lstm_layers': (256, 256),
        'dropout_keep_prob': 1.0,  # 0.25 - 0.75 opt
        'encode_internal_state': False,
        'conv_1d_num_filters': 64,
        'state_encoder_class_ref': conv_1d_casual_attention_encoder,
    }
)

trainer_config = dict(
    class_ref=A3C,
    kwargs=dict(
        opt_learn_rate=1e-4,
        opt_end_learn_rate=1e-5,
        opt_decay_steps=50*10**6,
        model_gamma=0.99,
        model_gae_lambda=1.0,
        model_beta=0.001, # entropy reg: 0.001 for non_shared encoder_params; 0.05 if 'share_encoder_params'=True
        rollout_length=20,
        time_flat=True, 
        model_summary_freq=10,
        episode_summary_freq=1,
        env_render_freq=5,
    )
)

launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=False,
    max_env_steps=100*10**6,
    root_random_seed=0,
    purge_previous=1,  # ask to override previously saved model and logs
    verbose=0
)

# Train it:
launcher.run()