import mlflow
import yaml
import time


def read_config(name, root):
    try:
        filename = root.replace('dbfs:', '/dbfs') + '/' + name
        with open(filename) as conf_file:
            conf = yaml.load(conf_file, Loader=yaml.FullLoader)
            return conf
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}. Please include a config file!")

def setupMlflowConf(conf):
    mlflow.set_experiment(conf['experiment-path'])
    time.sleep(5)
    experimentID = mlflow.get_experiment_by_name(conf['experiment-path']).experiment_id
    return experimentID