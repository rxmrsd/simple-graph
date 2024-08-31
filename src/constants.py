"""constants"""

import os

from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

OPENAI_API_TYPE = "azure"
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
