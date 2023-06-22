from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image,ImageDraw,ImageFont
import sys
import time


subscription_key = "426804134b8c424eb992d26b62fb8b25"
endpoint = "https://shristi29.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

folder = 'D:/azure_cv/cv_demo_images/'
out_folder = 'D:/azure_cv/cv_demo_images_output/'

files = os.listdir(folder)
# print(files)

font = ImageFont.truetype('arial.ttf',16)

for file in files:
    print(file)
    file_path = os.path.join(folder,file)

    image = Image.open(file_path)

    image_draw = ImageDraw.Draw(image)
    with open(file_path,mode='rb') as image_stream:
        result = computervision_client.detect_objects_in_stream(image_stream)

        for object in result.objects:

            left = object.rectangle.x
            top = object.rectangle.y
            width = object.rectangle.w
            height = object.rectangle.h

            shape = [(left,top),(left+width,top+height)]
            image_draw.rectangle(shape,outline='red',width=5)

            text = f'{ object.object_property } ({ object.confidence * 100 }%)'

            image_draw.text((left+5,top+height-30), text, (255,0,0), font = font)
            image_draw.text((left+5+1,top+height-30+1), text, (0,0,0), font = font)
        
        # image.show()
        image.save(os.path.join(out_folder,file))