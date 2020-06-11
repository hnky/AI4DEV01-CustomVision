from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient

cv_endpoint = "<INSERT CUSTOM VISION ENDPOINT>"
training_key = "<INSERT TRAINING KEY>"

trainer = CustomVisionTrainingClient(training_key, endpoint= cv_endpoint)
projects = trainer.get_projects()

for project in projects:
    print("\t",project.id,"\t",project.name)