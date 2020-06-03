import sys

from pyspark.sql import SparkSession

from lendingclub_scoring.pipelines.LendingClubConsumerPipeline import LendingClubConsumerPipeline
from lendingclub_scoring.config.ConfigProvider import read_config, setupMlflowConf

spark = SparkSession.builder.appName('Test').getOrCreate()
conf = read_config('consumer_config.yaml', sys.argv[1])
setupMlflowConf(conf)

p = LendingClubConsumerPipeline(spark, conf['data-path'],conf['output-path'],conf['model-name'])
p.run()

spark.read.load(conf['output-path']).show(1000, False)

