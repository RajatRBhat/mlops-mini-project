import unittest
from fastapi.testclient import TestClient
from fastapi_api.app import app

class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Hello', response.content)

    def test_predict_page(self):
        response = self.client.post('/predict', json={"user_query": "I love this!"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data.get("sentiment", ""), ["happiness", "sadness"], "Response should contain either 'happiness' or 'sadness'")

if __name__ == '__main__':
    unittest.main()