import sys

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import LendingClubModelEvaluationPipeline

spark = SparkSession.builder.appName('ForecastingTest').getOrCreate()
conf = read_config('train_config.yaml', sys.argv[1])
experimentID = setupMlflowConf(conf)
p = LendingClubModelEvaluationPipeline(spark, experimentID, conf['model-name'], conf['data-path'] )
p.run()

