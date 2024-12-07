FROM agrigorev/model-2024-hairstyle:v3

# Install required libraries
RUN pip install tensorflow==2.14.0
RUN pip install numpy==1.23.1
RUN pip install pillow  # For image handling

# Add your Lambda function code here
COPY lambda_function.py .

# Set the entrypoint to Lambda-compatible function handler
CMD ["lambda_function.lambda_handler"]
