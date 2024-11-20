import os
import ast
import csv
import opik
from opik import Opik
from opik import track
from opik.integrations.openai import track_openai
from openai import OpenAI
import litellm
import comet_ml
from utils import configure_env

configure_env(project_name="food_chatbot")

# Create or get the dataset
client = Opik()
dataset = client.get_or_create_dataset(name="foodchatbot_eval")


comet_ml.login(api_key=os.environ["OPIK_API_KEY"])
experiment = comet_ml.start(project_name="foodchatbot_eval")

logged_artifact = experiment.get_artifact(artifact_name="foodchatbot_eval",
                                          workspace="examples")
local_artifact = logged_artifact.download("../data/")
experiment.end()

# Read the CSV file and insert items into the dataset
with open('../data/foodchatbot_clean_eval_dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) # skip the header
    for row in reader:
        index, question, response = row
        dataset.insert([
            {"question": question, "response": response}
        ])