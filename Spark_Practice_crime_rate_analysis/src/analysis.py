from pyspark import SparkContext
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt

sc = SparkContext(appName="myApp")
sc.setLogLevel('ERROR')
sqlcontext = SQLContext(sc)

df = sqlcontext.read.csv("hdfs://wolf:9000/user/szy/data/crime/Crimes_-_2001_to_present.csv", header = True)
df = df.withColumn('Month', df['Date'].substr(0, 2))
aggregated = df \
    .groupBy("Year", "Month").count() \
    .groupBy("Month").mean().sort("Month")

aggregated.toPandas().plot.bar(x="Month", y="avg(count)")
plt.savefig("zhu_1.png")