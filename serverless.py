import boto3
import os
import io
import json
from PIL import Image
import torch

# Initialize S3 client
s3_client = boto3.client('s3')

def load_model_from_s3(bucket_name, model_key):
    """
    Load a model from an S3 bucket.
    """
    with io.BytesIO() as f:
        s3_client.download_fileobj(bucket_name, model_key, f)
        f.seek(0)
        model = torch.load_model(f)
    return model

def lambda_handler(event, context):
    """
    AWS Lambda handler.
    """
    # Specify your bucket name and model key
    bucket_name = 'your-bucket-name'
    model_key = 'your-model-path/model.h5'

    # Load the model
    model = load_model_from_s3(bucket_name, model_key)

    # Load and preprocess the image (example)
    # Assuming the image path is passed in the event
    image_key = event['image_key']
    image = load_image_from_s3(bucket_name, image_key)

    # Perform inference
    prediction = model.predict(image)

    # Process the prediction and return the result
    # (This will depend on your specific use case)
    processed_result = process_prediction(prediction)

    return {
        'statusCode': 200,
        'body': json.dumps(processed_result)
    }

def load_image_from_s3(bucket_name, image_key):
    """
    Load an image from an S3 bucket.
    """
    with io.BytesIO() as f:
        s3_client.download_fileobj(bucket_name, image_key, f)
        f.seek(0)
        image = Image.open(f)
        # Additional image preprocessing steps go here
    return image

def process_prediction(prediction):
    """
    Process the prediction output from the model.
    """
    # Implement processing logic based on your use case
    # For example, converting output tensors to a certain format
    return prediction.tolist()  # Example conversion to list

# For local testing (optional)
if __name__ == "__main__":
    event = {'image_key': 'path/to/your/image.jpg'}
    result = lambda_handler(event, None)
    print(result)