import unittest
from unittest.mock import patch, MagicMock
import json
from classification_handler import classify_document, handler

class TestClassificationHandler(unittest.TestCase):

    @patch("classification_handler.sagemaker_client")
    def test_classify_document_success(self, mock_sagemaker_client):
        # Mock SageMaker response
        mock_response = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps({"category": "Category A"}).encode('utf-8')))
        }
        mock_sagemaker_client.invoke_endpoint.return_value = mock_response

        # Input features
        endpoint_name = "test-endpoint"
        features = {"summary": "Test summary", "keywords": ["test", "data"]}

        # Call classify_document
        result = classify_document(endpoint_name, features)

        # Assertions
        self.assertEqual(result, {"category": "Category A"})
        mock_sagemaker_client.invoke_endpoint.assert_called_once_with(
            EndpointName=endpoint_name,
            Body=json.dumps(features),
            ContentType='application/json'
        )

    @patch("classification_handler.sagemaker_client")
    def test_classify_document_failure(self, mock_sagemaker_client):
        # Mock SageMaker exception
        mock_sagemaker_client.invoke_endpoint.side_effect = Exception("SageMaker error")

        # Input features
        endpoint_name = "test-endpoint"
        features = {"summary": "Test summary", "keywords": ["test", "data"]}

        # Call classify_document and expect an exception
        with self.assertRaises(Exception) as context:
            classify_document(endpoint_name, features)

        self.assertEqual(str(context.exception), "SageMaker error")

    def test_handler_success(self):
        # Mock event
        event = {
            "endpoint_name": "test-endpoint",
            "features": {"summary": "Test summary", "keywords": ["test", "data"]}
        }

        # Mock context
        context = MagicMock()

        # Patch classify_document
        with patch("classification_handler.classify_document", return_value={"category": "Category A"}) as mock_classify:
            result = handler(event, context)

            # Assertions
            self.assertEqual(result["statusCode"], 200)
            self.assertEqual(result["body"], {"classification": {"category": "Category A"}})
            mock_classify.assert_called_once_with("test-endpoint", {"summary": "Test summary", "keywords": ["test", "data"]})

    def test_handler_missing_key(self):
        # Mock event with missing key
        event = {
            "features": {"summary": "Test summary", "keywords": ["test", "data"]}
        }

        # Mock context
        context = MagicMock()

        # Call handler
        result = handler(event, context)

        # Assertions
        self.assertEqual(result["statusCode"], 400)
        self.assertIn("Missing key", result["body"]["error"])

    def test_handler_internal_error(self):
        # Mock event
        event = {
            "endpoint_name": "test-endpoint",
            "features": {"summary": "Test summary", "keywords": ["test", "data"]}
        }

        # Mock context
        context = MagicMock()

        # Patch classify_document to raise an exception
        with patch("classification_handler.classify_document", side_effect=Exception("Unexpected error")):
            result = handler(event, context)

            # Assertions
            self.assertEqual(result["statusCode"], 500)
            self.assertIn("internal server error", result["body"]["error"].lower())

if __name__ == "__main__":
    unittest.main()
