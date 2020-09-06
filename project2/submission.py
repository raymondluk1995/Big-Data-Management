"""
UNSW 20T2 COMP9313 Big Data Management Project 2
Student Name: Minrui Lu
Student Number: z5277884
"""

from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline, Transformer
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame


class Selector(Transformer):
    def __init__(self, outputCols=['features', 'label']):
        self.outputCols=outputCols
        
    def _transform(self, df: DataFrame) -> DataFrame:
        return df.select(*self.outputCols)

# Execute binary and operation for the prediction of two base models
def binary_and(x,y):
  temp = str(int(x))+str(int(y))
  temp = 2*int(temp[0])+int(temp[1])
  return(float(temp))

binary_udf = udf(binary_and,DoubleType())


# Generate base features for model training
def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
  word_tokenizer = Tokenizer(inputCol=input_descript_col,outputCol='words')
  count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)
  label_maker = StringIndexer(inputCol = input_category_col, outputCol = output_label_col)
  selector = Selector(outputCols = ['id','features','label'])
  pipeline = Pipeline(stages=[word_tokenizer,count_vectors,label_maker,selector])
  return (pipeline)    

# Meta features are based on kth-fold model training
def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
  model_pipeline = Pipeline(stages=[nb_0, nb_1, nb_2, svm_0, svm_1, svm_2])
  result_df = None
  # Get the number of group
  group_num = training_df.select('group').distinct().count() 
  for i in range(group_num):
    condition = training_df['group']==i
    train_df = training_df.filter(~condition).cache()
    test_df = training_df.filter(condition).cache() 

    if (result_df==None):
      result_df = model_pipeline.fit(train_df).transform(test_df)
    else:
      temp_df = model_pipeline.fit(train_df).transform(test_df)
      result_df = result_df.union(temp_df)
  
  result_df = result_df.withColumn('joint_pred_0',binary_udf(result_df.nb_pred_0,result_df.svm_pred_0))
  result_df = result_df.withColumn('joint_pred_1',binary_udf(result_df.nb_pred_1,result_df.svm_pred_1))
  result_df = result_df.withColumn('joint_pred_2',binary_udf(result_df.nb_pred_2,result_df.svm_pred_2))
  return(result_df.select('id','group','features','label','label_0','label_1','label_2','nb_pred_0','nb_pred_1','nb_pred_2',\
  'svm_pred_0','svm_pred_1','svm_pred_2','joint_pred_0','joint_pred_1','joint_pred_2'))




def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
  test_set = base_features_pipeline_model.transform(test_df) # get <id, features, labels>
  test_df = gen_base_pred_pipeline_model.transform(test_set) # get base model features
  # get joint predictions for the whole test data
  test_df = test_df.withColumn('joint_pred_0',binary_udf(test_df.nb_pred_0,test_df.svm_pred_0))
  test_df = test_df.withColumn('joint_pred_1',binary_udf(test_df.nb_pred_1,test_df.svm_pred_1))
  test_df = test_df.withColumn('joint_pred_2',binary_udf(test_df.nb_pred_2,test_df.svm_pred_2))

  test_features = gen_meta_feature_pipeline_model.transform(test_df) # get meta features based on base model features
  test_predict = meta_classifier.transform(test_features) # make final prediction based on meta features
  result = test_predict.select('id','label','final_prediction')
  return(result)
