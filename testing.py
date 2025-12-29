from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

text = "A former French surgeon, Dominique Pelicot, was sentenced to 20 years in prison for raping and abusing hundreds of patients. "
print(pipeline.predict(text))