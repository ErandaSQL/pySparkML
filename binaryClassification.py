# Databricks notebook source
'''
Spark session isolation
Every notebook attached to a cluster running Apache Spark 2.0.0 and above has a pre-defined variable called spark that represents a SparkSession. SparkSession is the entry point for using Spark APIs as well as setting runtime configurations.

Spark session isolation is enabled by default. You can also use global temporary views to share temporary views across notebooks. See Create View. To disable Spark session isolation, set spark.databricks.session.share to true in the Spark configuration.
'''


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('sparkClassfication1').getOrCreate()

# COMMAND ----------



# COMMAND ----------

df_f=spark.sql('select * from bank_new')
df_f.show()
#or
display(df_f)
#to list all the columns
#df_f.columns

# COMMAND ----------

#dealing with missing data
#unlike python in pySpark you have to go through each column to check for missing values.
#df_f.describe('age','job').show() but this doesnt give us all the information we need
#therefore, you data frame to pandas data frame by using toPandas o
#mind you, pandas may not be able to handle large volumnes of data, hence, we can take a sample and then convert data frame to pandas

def checkNull(df,cols,threshold):
  listDrop = []
  for col in cols:
    nullCount=df.where(df[col].isNull()).count()
    if (nullCount/df.count())*100 > threshold:
      print(str(col) + ' null count ' + str(nullCount) + ' null percentage ' + str(round((nullCount/df.count())*100,0)))
      #adding columns to a list to examine later
      listDrop.append(col)
  return listDrop

cols=df_f.columns    
listDrop=checkNull(df_f,cols,0) #threshold can be .6(60%) but i have used 0% means, columns with at least one null value will be dropped
print(listDrop)

# COMMAND ----------

#you decide to drop all columns with 60% or more Nulls
#first create a new data frame for data wrangling
df_dw=df_f
#drop all the columns in the list, when you use * , the drop comman will iterate through all list items
df_dw = df_dw.drop(*listDrop)  
#in addition day and month columns are not useful therfore dropped
df_dw=df_dw.drop('day','month')
df_dw.show()

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumns = ['marital', 'default', 'housing', 'loan']
stages=[]
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])    
    stages += [stringIndexer, encoder]


label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]

#convert string columns that contain numerical values to numeric
df_dw=df_dw.withColumn('age',df_dw.age.cast('double'))
df_dw=df_dw.withColumn('balance',df_dw.balance.cast('double'))
df_dw=df_dw.withColumn('campaign',df_dw.campaign.cast('double'))
df_dw=df_dw.withColumn('duration',df_dw.duration.cast('double'))
df_dw=df_dw.withColumn('pdays',df_dw.pdays.cast('double'))
df_dw=df_dw.withColumn('previous',df_dw.previous.cast('double'))


numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

from pyspark.ml import Pipeline  
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(df_dw)

preppedDataDF = pipelineModel.transform(df_dw)
preppedDataDF.show()

# COMMAND ----------

cols=df_dw.columns
selectedCols = ['label', 'features'] + cols
preppedDataDF = preppedDataDF.select(selectedCols)
preppedDataDF.printSchema()

# COMMAND ----------

train, test = preppedDataDF.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)


# COMMAND ----------

predictions = lrModel.transform(test)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

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

