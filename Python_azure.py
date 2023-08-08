from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.storage.blob import BlobServiceClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sklearn.metrics import confusion_matrix
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

subscription_key = "21c102c6e2e44810b868e6679d9794a5"
endpoint = "https://imagerecognitionpython.cognitiveservices.azure.com/"

text_analytics_key = "22a5674ad5244beb88e2ffed20e8d19f"
text_analytics_endpoint = "https://textreconitionser.cognitiveservices.azure.com/"
text_analytics_client = TextAnalyticsClient(endpoint=text_analytics_endpoint, credential=AzureKeyCredential(text_analytics_key),text_analytics_account="opinionmining")

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

connection_string = "DefaultEndpointsProtocol=https;AccountName=storageaccountimages77;AccountKey=8sSxsmE8+3OqWxksOiZZQuUwselC/y1FiyedrVAOAZ8OWD2mH0ksDgOZJPCC7FoWChJzvPvj0iPN+AStf7uP3w==;EndpointSuffix=core.windows.net"
container_name = "imagescv"  # Replace with your container name
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

folder = 'D:/azure_cv/cv_demo_images/'
out_folder = 'D:/azure_cv/cv_demo_images_output/'

files = os.listdir(folder)
font = ImageFont.truetype('arial.ttf', 16)

confidence_values = []  # Store confidence values
object_names = []  # Store object names

ground_truth_labels = [ 'dog']  # Ground truth labels
predicted_labels = []

for file in files:
    print(file)
    file_path = os.path.join(folder, file)

    image = Image.open(file_path)

    image_draw = ImageDraw.Draw(image)
    with open(file_path, mode='rb') as image_stream:
        result = computervision_client.detect_objects_in_stream(image_stream, visual_features=[VisualFeatureTypes.objects])

        for obj in result.objects:

            left = obj.rectangle.x
            top = obj.rectangle.y
            width = obj.rectangle.w
            height = obj.rectangle.h

            shape = [(left, top), (left + width, top + height)]
            image_draw.rectangle(shape, outline='red', width=5)

            text = f'{obj.object_property} ({obj.confidence * 100:.2f}%)'

            image_draw.text((left + 5, top + height - 30), text, (255, 0, 0), font=font)
            image_draw.text((left + 5 + 1, top + height - 30 + 1), text, (0, 0, 0), font=font)

            # Append confidence value and object name to the lists
            confidence_values.append(obj.confidence)
            object_names.append(obj.object_property)
            predicted_labels.append(obj.object_property)

            # Perform sentiment analysis on the object name
            response = text_analytics_client.analyze_sentiment(documents=[{"id": "1", "language": "en", "text": obj.object_property}],show_opinion_mining=True)

            sentiment = response[0].sentences[0].sentiment
            positive_score = response[0].sentences[0].confidence_scores.positive
            negative_score = response[0].sentences[0].confidence_scores.negative
            neutral_score = response[0].sentences[0].confidence_scores.neutral

            blob_client = blob_service_client.get_blob_client(container=container_name, blob=file)
            with open(os.path.join(out_folder, file), "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            # Print detailed sentiment scores
            print(f"Sentiment for {obj.object_property}: {sentiment}")
            # print(f"Translated name: {translated_name} (French)")
            print(f"Blob URL: {blob_client.url}")
            print(f"Positive score: {positive_score}")
            print(f"Negative score: {negative_score}")
            print(f"Neutral score: {neutral_score}")



        # image.show()
        image.save(os.path.join(out_folder, file))



missing_labels = list(set(predicted_labels) - set(ground_truth_labels))
ground_truth_labels.extend(missing_labels)
predicted_labels.extend(missing_labels)

predicted_labels = list(set(predicted_labels))


if len(ground_truth_labels) != len(predicted_labels):
    print("Number of ground truth labels and predicted labels do not match.")
    print(f"Ground truth labels: {len(ground_truth_labels)}")
    print(f"Predicted labels: {len(predicted_labels)}")
# Calculate confusion matrix
cm = confusion_matrix(ground_truth_labels, predicted_labels,labels=ground_truth_labels)


tp = np.diag(cm)
fn = np.sum(cm, axis=1) - tp
fp = np.sum(cm, axis=0) - tp
tn = np.sum(cm) - (tp + fp + fn)

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
sensitivity = recall  # Sensitivity is the same as recall

# Create a bar chart to display the metrics and confidence rate
metrics = ['Accuracy', 'Precision', 'Recall', 'Sensitivity']
values = [accuracy, precision, recall, sensitivity]
values1 = [np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(sensitivity)]
print(values)
print(values1)


# Plotting the confidence rate graph with labeled bars
plt.figure()
plt.bar(object_names, confidence_values)
plt.xlabel('Object Name')
plt.ylabel('Confidence Rate')
plt.title('Object Detection Confidence Rate')
plt.xticks(rotation=45, ha='right')
plt.show()

plt.bar(metrics, values1, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics and Confidence Rate')
plt.ylim([0, 1])
plt.tight_layout()
plt.show()



