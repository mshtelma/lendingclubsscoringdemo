import sys

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession

from lendingclub_scoring.pipelines.LendingClubConsumerPipeline import LendingClubConsumerPipeline
from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf

spark = SparkSession.builder.appName('Test').getOrCreate()
conf = read_config('consumer_config.yaml', sys.argv[1])
setupMlflowConf(conf)

p = LendingClubConsumerPipeline(spark, conf['data-path'],conf['output-path'],conf['model-name'])
p.run()

