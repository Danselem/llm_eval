from opik import track
import opik
from opik.integrations.openai import track_openai
from openai import OpenAI
import litellm
from utils import configure_env

configure_env(project_name="Multi-step-Chain-Demo")

# Define first step of the chain
@track
def generate_meal(ingredient):
    prompt = f"Generate one example of a meal that can be made with {ingredient}."
    response = litellm.completion(
    model="gemini/gemini-pro",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Define second step of the chain
@track
def generate_recipe(meal):
    prompt = f"Generate a step-by-step recipe for {meal}"
    response = litellm.completion(
    model="gemini/gemini-pro",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Chain the steps together

@track
def generate_recipe_from_ingredient(ingredient):
    meal = generate_meal(ingredient)
    story = generate_recipe(meal)
    return story

# generate_recipe_from_ingredient("garlic")
for ingredient in ["ogbono soup", "banga soup", "goat meat", "beef", "fish"]:
    generate_recipe_from_ingredient(ingredient)