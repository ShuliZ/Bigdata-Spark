from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *

import matplotlib.pyplot as plt


sc = SparkContext(appName="myApp")
sc.setLogLevel('ERROR')
sqlcontext = SQLContext(sc)

df = sqlcontext.read.csv("hdfs://wolf:9000/user/szy/data/crime/Crimes_-_2001_to_present.csv", header = True)
df = df.withColumn("Date", to_timestamp("Date","MM/dd/yyyy hh:mm:ss a"))\
       .withColumn("Month", month("Date"))\
       .withColumn("Week_of_year", weekofyear("Date"))\
       .withColumn("Day_of_week", dayofweek("Date"))\
       .withColumn("Hour_of_day",hour("Date"))\
       .withColumn("Date_only", from_unixtime(unix_timestamp("Date",'MM/dd/yyyy hh:mm:ss a'),'yyyy-MM-dd').cast('date'))

df_arrest = df.filter(col("Arrest") == True)

# hour of the day
df_arrest\
    .groupBy("Year", "Hour_of_day").count() \
    .groupBy("Hour_of_day").mean().sort("Hour_of_day") \
    .toPandas() \
    .plot \
    .bar(x="Hour_of_day", y="avg(count)")
plt.savefig("zhu_4a.png")

# day of the week
df_arrest\
    .groupBy("Year", "Day_of_week").count() \
    .groupBy("Day_of_week").mean().sort("Day_of_week") \
    .toPandas() \
    .plot \
    .bar(x="Day_of_week", y="avg(count)")
plt.savefig("zhu_4b.png")

# month of the year
df_arrest\
    .groupBy("Year", "Month").count() \
    .groupBy("Month").mean().sort("Month") \
    .toPandas() \
    .plot \
    .bar(x="Month", y="avg(count)")
plt.savefig("zhu_4c.png")