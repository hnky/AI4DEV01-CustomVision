from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

prediction_key = '<INSERT PREDICTION KEY>'
cv_endpoint = '<INSERT CUSTOM VISION ENDPOINT>'

predictor = CustomVisionPredictionClient(prediction_key, endpoint=cv_endpoint)

image_url = "https://missedprints.com/wp-content/uploads/2014/03/marge-simpson-lego-minifig.jpg"
predictor.classify_image_url(project.id,publish_iteration_name,url=image_url)

for prediction in results.predictions:
  print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
