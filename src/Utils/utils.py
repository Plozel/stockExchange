import json


def load_config():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config


def box_print(msg):
    print("=" * max(len(msg), 100))
    print(msg)
    print("=" * max(len(msg), 100))