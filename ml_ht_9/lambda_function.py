import json
import numpy as np
import tensorflow as tf
from PIL import Image
import urllib.request
from io import BytesIO

# Load the model (already present in the container)
model_path = 'model_2024_hairstyle_v2.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Function to download and prepare the image
def download_image(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.astype(np.float32)  # Cast to FLOAT32
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def lambda_handler(event, context):
    # Get image URL from event (e.g., an API call)
    image_url = event['image_url']
    
    # Download and prepare the image
    img = download_image(image_url)
    img_array = prepare_image(img)

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the output (prediction)
    output = interpreter.get_tensor(output_details[0]['index'])

    # Convert output to a native Python float to make it serializable
    prediction = float(output[0][0])

    # Return the result in a JSON-compatible format
    result = {
        'prediction': prediction
    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }

