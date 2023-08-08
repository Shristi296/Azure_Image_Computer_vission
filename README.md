
# Object Detection and Analysis with Azure Cognitive Services

This Python application utilizes Azure Cognitive Services to perform object detection, sentiment analysis, and Azure Blob Storage integration on a set of images. The goal is to showcase how multiple Azure services can be combined to enhance image processing and analysis tasks.

## Prerequisites

Before running the application, ensure you have the following:

1. Azure Cognitive Services subscription keys and endpoints for Computer Vision and Text Analytics.
2. An Azure Blob Storage account with the connection string and container name.
3. Python 3.x installed.
4. Required Python packages: azure-cognitiveservices-vision-computervision, msrest, azure-storage-blob, azure-ai-textanalytics, PIL, matplotlib, numpy.


## Setup

1. Clone this repository to your local machine.
2. Install the required Python packages:

```bash
pip install azure-cognitiveservices-vision-computervision msrest azure-storage-blob azure-ai-textanalytics pillow matplotlib numpy
```  

3. Replace the placeholders in the code with your actual subscription keys, endpoints, connection string, and container name.

## Usage

1. Place your images in the cv_demo_images directory.
2. Run the Python script:

```bash
python image_analysis.py

```
3. The script will perform the following tasks on each image:

- Detect objects using Computer Vision API.

-  Perform sentiment analysis using Text Analytics API.

-  Upload the analyzed image to Azure Blob Storage.

4. The analysis results, including sentiment scores and Blob Storage URLs, will be printed to the console.

## Output

The script will save the processed images with object detection bounding boxes in the cv_demo_images_output directory. Additionally, it will display bar charts showing the confidence rate of object detection and performance metrics.