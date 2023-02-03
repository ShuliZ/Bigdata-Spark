from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import TimestampType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor,GBTRegressor
import pandas as pd



def get_mape(model_df, y_start, y_end, m1, m2):
    train_year = []
    train_month = []
    test_year = []
    test_month = []
    mape_lst = []
    for y in range(y_start, y_end):
        for m in [m1, m2]:
            if m == m1:
                train = model_df.filter(col("year")==y).filter(col("month")<=6)
                test = model_df.filter(col("year")==y).filter(col("month")==7)
                train_year.append(y)
                train_month.append("1-6")
                test_year.append(y)
                test_month.append("7")
                mape_lst.append(model(train, test))
            elif m == m2:
                train = model_df.filter(col("year")==y).filter(col("month")>=7).filter(col("month")<=12)
                test = model_df.filter(col("year")==y+1).filter(col("month")==1)
                train_year.append(y)
                train_month.append("7-12")
                test_year.append(y+1)
                test_month.append("1")
                mape_lst.append(model(train, test))
            elif y == 2015 and m == 12:
                break
    return train_year, train_month, test_year, test_month, mape_lst

def model(train, test):
    # select features
    target = 'profit'
    features = ['var12', 'var13', 'var14', 'var15', 'var16', 'var17', 'var18', 'var23', 'var24', 'var25',
                'var26', 'var27', 'var28', 'var34', 'var35', 'var36', 'var37', 'var38', 'var45', 'var46', 
                'var47', 'var48', 'var56', 'var57', 'var58', 'var67', 'var68', 'var78',
                'p_last', 'p_mean', 'p_min','p_max']
    # build model pipline
    assembler = VectorAssembler(inputCols=features, outputCol='features').setHandleInvalid('skip')
    rf = RandomForestRegressor(featuresCol="features", labelCol="profit")
    pipeline = Pipeline(stages=[assembler, rf])

    # fit the model
    model = pipeline.fit(train)
    pred = model.transform(test)
    pred = pred.select('profit', 'prediction')\
               .withColumn('mape', abs((pred['profit']-pred['prediction'])/pred['profit']))
    mape = pred.select(mean(pred['mape']).alias('mape')).collect()[0]['mape']
    return mape 


if __name__ == "__main__":
    sc = SparkContext()
    sqlcontext = SQLContext(sc)
    path = 's3://msia-431-hw-data/full_data.csv'
    df = sqlcontext.read.csv(path, header=True)
    # drop missing valuues
    df = df.na.drop()

    # Get Year and Month
    df = df.withColumn("month", col("time_stamp").substr(6, 2).cast("Integer"))\
        .withColumn("year", col("time_stamp").substr(0, 4).cast("Integer"))

    df = df.orderBy("trade_id", "bar_num")
    df = df.withColumn("bar_group", ((col("bar_num")-1)/10).cast("Integer"))

    df_right = df.withColumn('p_last', last('profit', True).over(Window.partitionBy('trade_id', 'bar_group').orderBy('bar_group')))\
             .withColumn('p_mean', mean('profit').over(Window.partitionBy('trade_id', 'bar_group').orderBy('bar_group')))\
             .withColumn('p_min', min('profit').over(Window.partitionBy('trade_id', 'bar_group').orderBy('bar_group')))\
             .withColumn('p_max', max('profit').over(Window.partitionBy('trade_id', 'bar_group').orderBy('bar_group')))\
             .filter(col('bar_num') % 10 == 0)\
             .orderBy(['trade_id', 'bar_group'], ascending =[True, True])
    df_right = df_right.withColumn('next_bar_group', col('bar_group') + 1)
    df_right = df_right.select('trade_id', 'p_last','p_mean', 'p_min','p_max','next_bar_group')

    for c in ['trade_id', 'p_last','p_mean', 'p_min','p_max','next_bar_group']:
        df_right = df_right.withColumn(c, df_right[c].cast('int'))
    
    for c in ['var12', 'var13', 'var14', 'var15', 'var16', 'var17', 'var18', 'var23', 'var24', 'var25',
                'var26', 'var27', 'var28', 'var34', 'var35', 'var36', 'var37', 'var38', 'var45', 'var46', 
                'var47', 'var48', 'var56', 'var57', 'var58', 'var67', 'var68', 'var78',
                'bar_num','year','month','profit','bar_group','trade_id']:
        df = df.withColumn(c, df[c].cast('int'))
    
    merged_df = df.join(df_right, on=(df.bar_group == df_right.next_bar_group) & (df.trade_id == df_right.trade_id), how='left')
    model_df = merged_df.filter(col('bar_num') > 10)

    model_df = model_df.select('var12', 'var13', 'var14', 'var15', 'var16', 'var17', 'var18', 'var23', 'var24', 'var25',
                'var26', 'var27', 'var28', 'var34', 'var35', 'var36', 'var37', 'var38', 'var45', 'var46', 
                'var47', 'var48', 'var56', 'var57', 'var58', 'var67', 'var68', 'var78',
                'p_last', 'p_mean', 'p_min','p_max', 'profit','year','month')
    
    train_year, train_month, test_year, test_month, mape_lst = get_mape(model_df, 2008, 2016, 6, 12)

    output1 = pd.DataFrame({"train_year": train_year, 
                        "train_month":train_month,
                        "test_year": test_year,
                        "test_month":test_month,
                        "mape": mape_lst})

    output2 = pd.DataFrame({'avg_MAPE': [output1['mape'].mean()],
                            'min_MAPE': [output1['mape'].min()],
                            'max_MAPE': [output1['mape'].max()]})

    # save output file to S3
    output1.to_csv('s3://2022-msia431-shulizhu/Exercise1.txt', header=True, index=False, sep=',')
    output2.to_csv('s3://2022-msia431-shulizhu/Exercise1.txt', header=True, index=False, sep=',', mode='a')

    sc.stop()



    