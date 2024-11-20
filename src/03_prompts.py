from opik import track
import opik
from opik.integrations.openai import track_openai
from openai import OpenAI
import litellm
from utils import configure_env

configure_env(project_name="food_chatbot")

# Define Menu

menu_items = """
Menu: Kids Menu
Food Item: Mini Cheeseburger
Price: $6.99
Vegan: N
Popularity: 4/5
Included: Mini beef patty, cheese, lettuce, tomato, and fries.

Menu: Appetizers
Food Item: Loaded Potato Skins
Price: $8.99
Vegan: N
Popularity: 3/5
Included: Crispy potato skins filled with cheese, bacon bits, and served with sour cream.

Menu: Appetizers
Food Item: Bruschetta
Price: $7.99
Vegan: Y
Popularity: 4/5
Included: Toasted baguette slices topped with fresh tomatoes, basil, garlic, and balsamic glaze.

Menu: Main Menu
Food Item: Grilled Chicken Caesar Salad
Price: $12.99
Vegan: N
Popularity: 4/5
Included: Grilled chicken breast, romaine lettuce, Parmesan cheese, croutons, and Caesar dressing.

Menu: Main Menu
Food Item: Classic Cheese Pizza
Price: $10.99
Vegan: N
Popularity: 5/5
Included: Thin-crust pizza topped with tomato sauce, mozzarella cheese, and fresh basil.

Menu: Main Menu
Food Item: Spaghetti Bolognese
Price: $14.99
Vegan: N
Popularity: 4/5
Included: Pasta tossed in a savory meat sauce made with ground beef, tomatoes, onions, and herbs.

Menu: Vegan Options
Food Item: Veggie Wrap
Price: $9.99
Vegan: Y
Popularity: 3/5
Included: Grilled vegetables, hummus, mixed greens, and a wrap served with a side of sweet potato fries.

Menu: Vegan Options
Food Item: Vegan Beyond Burger
Price: $11.99
Vegan: Y
Popularity: 4/5
Included: Plant-based patty, vegan cheese, lettuce, tomato, onion, and a choice of regular or sweet potato fries.

Menu: Desserts
Food Item: Chocolate Lava Cake
Price: $6.99
Vegan: N
Popularity: 5/5
Included: Warm chocolate cake with a gooey molten center, served with vanilla ice cream.

Menu: Desserts
Food Item: Fresh Berry Parfait
Price: $5.99
Vegan: Y
Popularity: 4/5
Included: Layers of mixed berries, granola, and vegan coconut yogurt.
"""

# Simple little client class for using different LLM APIs (OpenAI or LiteLLM)
class LLMClient:
  def __init__(self, client_type: str ="litellm", model: str ="gemini/gemini-pro"):
    self.client_type = client_type
    self.model = model

    if self.client_type == "openai":
      self.client = track_openai(OpenAI())

    else:
      self.client = None

  # LiteLLM query function
  def _get_litellm_response(self, query: str, system: str = "You are a helpful assistant."):
    messages = [
        {"role": "system", "content": system },
        { "role": "user", "content": query }
    ]

    response = litellm.completion(
        model=self.model,
        messages=messages
    )

    return response.choices[0].message.content

  # OpenAI query function - use **kwargs to pass arguments like temperature
  def _get_openai_response(self, query: str, system: str = "You are a helpful assistant.", **kwargs):
    messages = [
        {"role": "system", "content": system },
        { "role": "user", "content": query }
    ]

    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        **kwargs
    )

    return response.choices[0].message.content


  def query(self, query: str, system: str = "You are a helpful assistant.", **kwargs):
    if self.client_type == 'openai':
      return self._get_openai_response(query, system, **kwargs)

    else:
      return self._get_litellm_response(query, system)



# Initialize LLMClient

llm_client = LLMClient()

@track
def reasoning_step(user_query, menu_items):
    prompt_template = """
    Your task is to answer questions factually about a food menu, provided below and delimited by +++++. The user request is provided here: {request}

    Step 1: The first step is to check if the user is asking a question related to any type of food (even if that food item is not on the menu). If the question is about any type of food, we move on to Step 2 and ignore the rest of Step 1. If the question is not about food, then we send a response: "Sorry! I cannot help with that. Please let me know if you have a question about our food menu."

    Step 2: In this step, we check that the user question is relevant to any of the items on the food menu. You should check that the food item exists in our menu first. If it doesn't exist then send a kind response to the user that the item doesn't exist in our menu and then include a list of available but similar food items without any other details (e.g., price). The food items available are provided below and delimited by +++++:

    +++++
    {menu_items}
    +++++

    Step 3: If the item exists in our food menu and the user is requesting specific information, provide that relevant information to the user using the food menu. Make sure to use a friendly tone and keep the response concise.

    Perform the following reasoning steps to send a response to the user:
    Step 1: <Step 1 reasoning>
    Step 2: <Step 2 reasoning>
    Response to the user (only output the final response): <response to user>
    """

    prompt = prompt_template.format(request=user_query, menu_items=menu_items)
    response = llm_client.query(prompt)
    return response


@track
def extraction_step(reasoning):
    prompt_template = """
    Extract the final response from delimited by ###.

    ###
    {reasoning}.
    ###

    Only output what comes after "Response to the user:".
    """

    prompt = prompt_template.format(reasoning=reasoning)
    response = llm_client.query(prompt)
    return response

@track
def refinement_step(final_response):
    prompt_template = """
    Perform the following refinement steps on the final output delimited by ###.

    1). Shorten the text to one sentence
    2). Use a friendly tone

    ###
    {final_response}
    ###
    """

    prompt = prompt_template.format(final_response=final_response)
    response = llm_client.query(prompt)
    return response

@track
def verification_step(user_question, refined_response, menu_items):
    prompt_template = """
    Your task is to check that the refined response (delimited by ###) is providing a factual response based on the user question (delimited by @@@) and the menu below (delimited by +++++). If yes, just output the refined response in its original form (without the delimiters). If no, then make the correction to the response and return the new response only.

    User question: @@@ {user_question} @@@

    Refined response: ### {refined_response} ###

    +++++
    {menu_items}
    +++++
    """

    prompt = prompt_template.format(user_question=user_question, refined_response=refined_response, menu_items=menu_items)
    response = llm_client.query(prompt)
    return response

@track
def generate_food_chatbot(user_query, menu_items):
    reasoning = reasoning_step(user_query, menu_items)
    extraction = extraction_step(reasoning)
    refinement = refinement_step(extraction)
    verification = verification_step(user_query, refinement, menu_items)
    return verification

generate_food_chatbot("What is the price of the cheeseburger?", menu_items)