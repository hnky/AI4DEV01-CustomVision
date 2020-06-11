from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

prediction_key = '<INSERT PREDICTION KEY>'
cv_endpoint = '<INSERT CUSTOM VISION ENDPOINT>'
publish_iteration_name = '<INSERT ITERATION NAME>'
project_id = "<INSERT PROJECT ID>" # use get_project_id.py to find it

image_url = "https://raw.githubusercontent.com/hnky/dataset-lego-figures/master/_test/marge.jpg"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(cv_endpoint, prediction_credentials)
predictor = CustomVisionPredictionClient(prediction_key, endpoint=cv_endpoint)

results = predictor.classify_image_url(project_id,publish_iteration_name,url=image_url)

for prediction in results.predictions:
  print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
