import os
from dotenv import load_dotenv
from pathlib import Path

import opik
import google.generativeai as genai


def configure_env(project_name: str):
    dotenv_path = Path('../.env')
    load_dotenv(dotenv_path=dotenv_path)

    os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")

    os.environ["OPIK_PROJECT_NAME"] = project_name
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    print("Environment configured successfully.")

    opik.configure()