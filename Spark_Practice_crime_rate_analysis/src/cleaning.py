import re
import math

from pyspark import SparkContext
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import Vectors

import matplotlib.pyplot as plt


sc = SparkContext(appName="myApp")
sc.setLogLevel('ERROR')

# load data
myRDD = sc.textFile("hdfs://wolf:9000/user/szy/data/crime/Crimes_-_2001_to_present.csv")
header = myRDD.first()
content = myRDD.filter(lambda x: x != header)
data = content.map(lambda line: re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', line))

# Q1
top = data.filter(lambda x: x[17] in ["2020","2019","2018"])\
          .map(lambda x: (x[3][0:5], 1))\
          .reduceByKey(lambda x, y: x+y)\
          .sortBy(lambda x: x[1], False)\
          .take(10)


# Q2
last_five_year = ["2016","2017","2018", "2019","2020"]
beat_set = data.map(lambda x: x[10])\
               .distinct()\
               .sortBy(lambda x: int(x))\
               .collect()
full_df = sc.parallelize(beat_set).cartesian(sc.parallelize(last_five_year))

beat_count = full_df.map(lambda x: ((x[0],x[1]),0))\
                    .leftOuterJoin(
                        data.filter(lambda x:  x[17] in last_five_year)\
                            .map(lambda x: ((x[10], x[17]), 1))\
                            .reduceByKey(lambda a, b: a+b)
                            )\
                    .map(lambda x: (x[0], x[1][1] if x[1][1] else 0))\
                    .sortBy(lambda x: (int(x[0][0]), int(x[0][1])))\
                    .map(lambda x: (x[0][1], [x[1]]))\
                    .reduceByKey(lambda a, b: a+b)\
                    .sortBy(lambda x: int(x[0]))\
                    .map(lambda row: row[0:][1])

corr_mat = Statistics.corr(beat_count, method = "pearson")

corr_dict = {}
for i in range(len(beat_set)):
    for j in range(len(beat_set)):
        if i < j:
            if not math.isnan(corr_mat[i][j]):
                corr_dict[beat_set[i] + "-" + beat_set[j]] = corr_mat[i][j]
sorted_result = sorted(corr_dict.items(), key=lambda x : x[1], reverse = True)



# Q3
daly = data.filter(lambda x: (int(x[17]) >= 2004) and (int(x[17]) <= 2010))\
                    .map(lambda x: (x[17], 1))\
                    .reduceByKey(lambda x, y: x+y)\
                    .sortBy(lambda x: int(x[0]))\
                    .map(lambda x: x[1])\
                    .collect()  

emanuel = data.filter(lambda x: (int(x[17]) >= 2011) and (int(x[17]) <= 2017))\
                    .map(lambda x: (x[17], 1))\
                    .reduceByKey(lambda x, y: x+y)\
                    .sortBy(lambda x: int(x[0]))\
                    .map(lambda x: x[1])\
                    .collect() 

test = Statistics.chiSqTest((Vectors.dense(daly),Vectors.dense(emanuel)))
pval = test.pValue


# write to separate txt files
text_file = open("zhu_2a.txt", "w")
text_file.write("The top 10 blocks in crime events in the last 3 years are" + '\n')
for i in range(len(top)):
    text_file.write(str(top[i]) + '\n')
text_file.close()


text_file = open("zhu_2b.txt", "w")
text_file.write("The correlation n the number of crime events between two beats are as follows:" + '\n')
for i in range(len(sorted_result)):
    text_file.write(str(sorted_result[i]) + '\n')
text_file.close()


text_file = open("zhu_2c.txt", "w")
text_file.write("The numbers of crime events from 2004 to 2010 when Daly was in charge are" + '\n')
for i in range(len(daly)):
    text_file.write(str(daly[i]) + '\n')
text_file.write("The numbers of crime events from 2011 to 2017 when Emanuel was in charge are" + '\n')
for i in range(len(emanuel)):
    text_file.write(str(emanuel[i]) + '\n')
text_file.write("The p value obtained from chi-square test is" + '\n')
text_file.write(str(pval) + '\n')
text_file.close()