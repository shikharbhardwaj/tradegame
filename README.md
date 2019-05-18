## tradegame

An Algorithmic Trader using Reinforcement learning, [Financial Trading
as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1807.02787).

## Dependencies

See `requirements.txt`.

## Training and evaluation
To train a new model, configure the training using the JSON config file, a sample is
provided in `sample.json`

    {
        "data_location": "/data/tradegame_data/sampled_data_15T",
        "pairs": ["AUDJPY", "AUDNZD", "AUDUSD", "CADJPY", "CHFJPY", "EURCHF",
            "EURGBP", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD",
            "USDCHF", "USDJPY"],
        "trade_pair": "EURUSD",
        "begin_year": 2012,
        "end_year": 2017,
        "start_cash": 200000,
        "trade_size": 100000
    }

Execute the training script

    $ python train.py config.json

To evaluate a trained model update the JSON config file for testing, a sample is provided
in `test.json`

    {
        "data_location": "D:\\tradegame_data\\sampled_data_15T",
        "model_location": "E:\\dev\\tradegame\\src\\models\\flat_state_exp\\eval_model2.h5",
        "pairs": ["AUDJPY", "AUDNZD", "AUDUSD", "CADJPY", "CHFJPY", "EURCHF",
            "EURGBP", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD",
            "USDCHF", "USDJPY"],
        "trade_pair": "EURUSD",
        "begin_year": 2017,
        "end_year": 2018,
        "start_cash": 200000,
        "trade_size": 100000
    }

Execute the evaluation script

    $ python train.py test.json
