import sys

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import LendingClubTrainingPipeline

spark = SparkSession.builder.appName('ForecastingTest').getOrCreate()
conf = read_config('train_config.yaml', sys.argv[1])
setupMlflowConf(conf)
p = LendingClubTrainingPipeline(spark, conf['data-path'], conf['model-name'])
p.run()

