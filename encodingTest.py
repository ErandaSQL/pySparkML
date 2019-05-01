# Databricks notebook source
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

languageList = [(1,'python','DS'),(2,'R','ST'),(3,'sas','ST'),(4,'scala','BD')]
languageDF = spark.createDataFrame(languageList,['id','language','use'])
languageDF.show()

#string indexer
stringIndexer = StringIndexer(inputCol = 'use', outputCol = 'useIndex')
indexed_df = stringIndexer.fit(languageDF).transform(languageDF)
indexed_df.show()

#one hot encorder
encoder = OneHotEncoder(inputCol='useIndex', outputCol='useVec')
encoded_df = encoder.transform(indexed_df)
encoded_df.show()

# COMMAND ----------

# first note that there are only 2 values
# thats because one value is deduced ( for an example, without haveing two columns for male (value 1 for male) and female(value 1 for female)
# you can have one column Male (1 for male, 0 for female)
# How to interpr
# (2,[0],[1.0]) means a vector of length 2 with 1.0 at position 0 and 0 elsewhere.

