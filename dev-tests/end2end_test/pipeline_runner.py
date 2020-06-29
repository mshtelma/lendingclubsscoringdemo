import sys

import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession

from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf
from lendingclub_scoring.pipelines.LendingClubTrainingPipeline import LendingClubTrainingPipeline
from lendingclub_scoring.pipelines.LendingClubModelEvaluationPipeline import LendingClubModelEvaluationPipeline
from lendingclub_scoring.pipelines.LendingClubConsumerPipeline import LendingClubConsumerPipeline

spark = SparkSession.builder.appName('ForecastingTest').getOrCreate()
conf = read_config('e2e_int_config.yaml', sys.argv[1])
experimentID = setupMlflowConf(conf)

limit = 100

# train
p = LendingClubTrainingPipeline(spark, conf['data-path'], conf['model-name'], limit=limit)
p.run()

spark_df = spark.read.format("mlflow-experiment").load(experimentID)
assert spark_df.where("tags.candidate='true'").count() > 0

# deploy
p = LendingClubModelEvaluationPipeline(spark, experimentID, conf['model-name'], conf['data-path'], limit=limit)
p.run()
assert spark_df.where("tags.candidate='true'").count() == 0
assert len(MlflowClient().get_latest_versions(conf['model-name'], stages=['Production'])) > 0

# consume
p = LendingClubConsumerPipeline(spark, conf['data-path'], conf['test-output-path'], conf['model-name'],
                                stage=['Production'], limit=limit)
p.run()

res_df = spark.read.load(conf['test-output-path'])
assert res_df.count() > 0
assert 'prediction' in res_df.columns
