from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import *
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

sc = SparkContext(appName="myApp")
sc.setLogLevel('ERROR')
sqlcontext = SQLContext(sc)
# read crime data and external 311 request data
df = sqlcontext.read.csv("hdfs://wolf:9000/user/szy/data/crime/Crimes_-_2001_to_present.csv", header = True)
df_311 = sqlcontext.read.csv("hdfs://wolf:9000/user/szy/data/crime/extra/311_Service_Requests_-_Vacant_and_Abandoned_Buildings_Reported_-_Historical.csv", header = True)
df_311 = df_311.select("DATE SERVICE REQUEST WAS RECEIVED","SERVICE REQUEST NUMBER").na.drop()
# extract year and week from date and count by week
df_311 = df_311.withColumn("Date", from_unixtime(unix_timestamp("DATE SERVICE REQUEST WAS RECEIVED",'MM/dd/yyyy'),'yyyy-MM-dd').cast('date'))\
               .withColumn("Year", year("Date"))\
               .withColumn("Week_of_year", weekofyear("Date"))\
               .groupby("Year","Week_of_year")\
               .count()\
               .withColumnRenamed('count', 'complaint_cnt')
# extract year and week from date and find violent cases
df = df.withColumn("Date", from_unixtime(unix_timestamp("Date",'MM/dd/yyyy hh:mm:ss a'),'yyyy-MM-dd').cast('date'))
df = df.withColumn("Week_of_year", weekofyear("Date"))
df = df.withColumn("Violent", when(col('Primary Type')\
       .isin(['HOMICIDE','ROBBERY','CRIMINAL SEXUAL ASSAULT','ASSAULT']), 1).otherwise(0))

# merge two aggregated datasets
df_merged = df.groupby('Year','Week_of_year','Beat','Violent')\
       .count()\
       .join(df_311, (df.Year == df_311.Year) & (df.Week_of_year == df_311.Week_of_year), "left")\
       .drop(df_311.Year)\
       .drop(df_311.Week_of_year)\
       .sort('Violent','Beat', 'Year','Week_of_year')
# create lag and lead counts
df_model = df_merged\
        .withColumn('lag1', lag('count').over(Window.partitionBy("Violent","Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag2', lag('count', 2).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag3', lag('count', 3).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag4', lag('count', 4).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag5', lag('count', 5).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag6', lag('count', 6).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag7', lag('count', 7).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('lag8', lag('count', 8).over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .withColumn('next_count', lead('count').over(Window.partitionBy("Violent", "Beat").orderBy('Year','Week_of_year')))\
        .orderBy('Violent','Beat','Year','Week_of_year')\
        .na.drop()
# indexer for region and time
beatIdx = StringIndexer(inputCol='Beat', outputCol='BeatIdx')
yearIdx = StringIndexer(inputCol='Year', outputCol='YearIdx')
weekIdx = StringIndexer(inputCol='Week_of_year', outputCol='WeekIdx')

# encoder for indexes and assembler for all variables
encoder = OneHotEncoder(inputCols = ['BeatIdx','YearIdx','WeekIdx'], outputCols = ['BeatVec','YearVec','WeekVec'], handleInvalid = 'keep')
assembler = VectorAssembler(inputCols=['BeatVec','YearVec','WeekVec', 'complaint_cnt', 'count', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8'], outputCol='features')

# model and pipelineinitialization
rf = RandomForestRegressor(labelCol="next_count", featuresCol="features", seed=10)
pipeline = Pipeline(stages=[beatIdx, yearIdx, weekIdx, encoder, assembler, rf])

# split data by violent cases
non_violent = df_model.filter(col("Violent")==0)
violent = df_model.filter(col("Violent")==1)

# training and evaluation
train, test = df_model.randomSplit([0.75, 0.25], seed=10)
model = pipeline.fit(train)
predictions = model.transform(test)

predictions2 = predictions.select(col("next_count").cast("Float"), col("prediction"))
evaluator_mse = RegressionEvaluator(labelCol="next_count", predictionCol="prediction", metricName="mse")
mse = evaluator_mse.evaluate(predictions2)

# write evaluation results
text_file = open("zhu_3.txt", "w")
text_file.write("The MSE for all crime cases is " + str(mse) + '\n')

# repeat for violent cases
train, test = violent.randomSplit([0.75, 0.25], seed=10)

model = pipeline.fit(train)
predictions = model.transform(test)

predictions2 = predictions.select(col("next_count").cast("Float"), col("prediction"))
evaluator_mse = RegressionEvaluator(labelCol="next_count", predictionCol="prediction", metricName="mse")
mse = evaluator_mse.evaluate(predictions2)

text_file.write("The MSE for violent crime cases is " + str(mse) + '\n')
text_file.close()

