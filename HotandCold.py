'''
Business problem - at a large shopping mall the management wants to know how shoppers respond to different temperature settings. The 
                   management has got hold of data from a different shopping mall that shows how shoppers reacted to different 
                   temperature settings. Based on the data gathered, the management predict what shoppers will feel for different 
                   temperature settings.

Analytic approach -shoppers reaction is recorded as a label (Hot\ Cold), therefore output is a categorical value. Hence classification\ 
                   logistic regression is used.

Data requirments - temparature and shoppers reaction

Data collection - HotandCold.csv

Data undertanding - see below EDA section

Data preparation - see below DW (data wrangle) section

Modeling - 

Evaluation -

Deployment -

Feedback -

'''

# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('HotandCold').getOrCreate()

df_f=spark.sql('select * from HotandCold')
df_f.show()

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

stages=[]
label_stringIdx = StringIndexer(inputCol = 'Feel', outputCol = 'label')
stages = [label_stringIdx]

numericCol=['Temparature']
assemblerInputs =  numericCol
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# COMMAND ----------

from pyspark.ml import Pipeline  
df_dw=df_f
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(df_dw)

preppedDataDF = pipelineModel.transform(df_dw)
preppedDataDF.show()

# COMMAND ----------

cols=df_dw.columns
selectedCols = ['label', 'features'] + cols
preppedDataDF = preppedDataDF.select(selectedCols)
preppedDataDF.show()
print(preppedDataDF.count())

# COMMAND ----------

train, test = preppedDataDF.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)

predictions = lrModel.transform(test)

#from pyspark.ml.evaluation import BinaryClassificationEvaluator
#evaluator = BinaryClassificationEvaluator()
#print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

trainingSummary = lrModel.summary

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

# COMMAND ----------

lrModel.write().overwrite().save('lrModel_v1')
loadedModel=lrModel.load('lrModel_v1')

# COMMAND ----------

from pyspark.mllib.linalg import Vectors

productionDataSchema = StructType([StructField("temperature", DoubleType(), True)])
productionTemparatureData = [40.0]
productionData=sqlContext.createDataFrame(list(zip(productionTemparatureData)),schema=productionDataSchema)

vectorAssembler = VectorAssembler(inputCols=["temperature"],outputCol="features")
df3 = vectorAssembler.transform(productionData).select('features')

prediction1 = loadedModel.transform(df3)
selected1 = prediction1.select("features", "prediction")
for row in selected1.collect():
    print(row)

