import boto3
import json
import logging
from botocore.exceptions import BotoCoreError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the SageMaker runtime client
sagemaker_client = boto3.client('sagemaker-runtime')

def classify_document(endpoint_name, features):
    """
    Invokes the SageMaker classification model endpoint with the given features.

    Args:
        endpoint_name (str): The name of the SageMaker endpoint for classification.
        features (dict): The input features for the classification model.

    Returns:
        dict: The classification result.
    """
    try:
        logger.info("Invoking SageMaker endpoint: %s", endpoint_name)
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(features),
            ContentType='application/json'
        )

        result = json.loads(response['Body'].read().decode('utf-8'))
        logger.info("Classification result: %s", result)
        return result

    except (BotoCoreError, ClientError) as e:
        logger.error("Failed to invoke SageMaker endpoint: %s", e)
        raise

def handler(event, context):
    """
    AWS Lambda handler function for document classification.

    Args:
        event (dict): The input event containing document data and features.
        context (object): AWS Lambda context object.

    Returns:
        dict: A response containing the classification results or an error message.
    """
    try:
        # Extract input data from the event
        endpoint_name = event['endpoint_name']
        features = event['features']

        # Log the incoming request
        logger.info("Received event: %s", json.dumps(event))

        # Perform classification
        classification_result = classify_document(endpoint_name, features)

        # Construct and return the response
        response = {
            "statusCode": 200,
            "body": {
                "classification": classification_result
            }
        }
        logger.info("Handler response: %s", response)
        return response

    except KeyError as e:
        logger.error("Missing key in the event: %s", e)
        return {
            "statusCode": 400,
            "body": {
                "error": f"Missing key in the input: {str(e)}"
            }
        }

    except Exception as e:
        logger.error("Error during classification: %s", e)
        return {
            "statusCode": 500,
            "body": {
                "error": "An internal server error occurred."
            }
        }

if __name__ == "__main__":
    # Example usage
    example_event = {
        "endpoint_name": "classification-endpoint",
        "features": {
            "summary": "This is a test summary.",
            "keywords": ["test", "example"]
        }
    }

    # Simulating a Lambda context
    class Context:
        def __init__(self):
            self.aws_request_id = "12345"
            self.log_stream_name = "example-log-stream"

    context = Context()
    response = handler(example_event, context)
    print(response)
