#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import sys
import os


spark = SparkSession     .builder     .appName(
    "Python Spark SQL Test")     .getOrCreate()


# In[5]:


df = spark.read.option("header", "true")     .option("delimiter", ",")     .option(
    "inferSchema", "true")     .csv("./backend-dev-data-dataset.txt")


# In[9]:


df.printSchema()


# In[22]:


df.show(truncate=False)


stddev_num = os.getenv('STD_DEV_NUM')
cat_7_filter_value = os.getenv('FILTER_VALUE')

stddev_num = int(stddev_num or 3)
cat_7_filter_value = cat_7_filter_value or "frequent"


moments_df_3 = df.crossJoin(
    df.select(
        F.stddev_pop("cont_3").alias("cont_3_std_pop"),
        F.avg("cont_3").alias("cont_3_avg")
    )
)\
    .withColumn("cont_3_norm", (F.col("cont_3") - F.col("cont_3_avg")) / F.col("cont_3_std_pop"))

outliers_df_3 = moments_df_3.filter(~(F.abs("cont_3_norm") <= stddev_num))

outliersremoved_df_3 = moments_df_3.filter(F.abs("cont_3_norm") <= stddev_num).filter(F.col("cat_7") == cat_7_filter_value).withColumn("date", F.to_date("date_2")).groupBy("key_1", "date").agg(
    F.avg(F.col("cont_3")).alias("avg_cont_3"),
    F.avg(F.col("cont_4")).alias("avg_cont_4"),
    F.first(F.col("disc_5")).alias("avg_disc_5"),
    F.first(F.col("disc_6")).alias("first_disc_6"),
    F.first(F.col("cat_7")).alias("first_cat_7"),
    F.first(F.col("cat_8")).alias("first_cat_8"),
    F.avg(F.col("cont_9")).alias("avg_cont_9"),
    F.avg(F.col("cont_10").cast("double")).alias("avg_cont_10"),
)\
    .orderBy("key_1", "date")\
    .fillna(0, subset=["avg_cont_3", "avg_cont_4", "avg_disc_5", "avg_cont_9", "avg_cont_10"])\
    .withColumn("transformation", F.pow("avg_cont_9", 3) + F.exp("avg_cont_10"))


# In[38]:
print("outlies removed:")
outliers_df_3.show(truncate=False)
print("schema of dataset")
outliersremoved_df_3.printSchema()


# In[45]:

print("The transformation column")
outliersremoved_df_3.select("key_1", "date", "transformation",
                            "avg_cont_9", "avg_cont_10").show(truncate=False)


# In[48]:

print("distict values of cat_8")
outliersremoved_df_3.select("first_cat_8").distinct().show()
