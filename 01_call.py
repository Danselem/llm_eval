from litellm.integrations.opik.opik import OpikLogger
from opik.opik_context import get_current_span_data
from opik import track
import litellm
from utils import configure_env

configure_env(project_name="logging-traces-litellm")



opik_logger = OpikLogger()
# In order to log LiteLLM traces to Opik, you will need to set the Opik callback
litellm.callbacks = [opik_logger]

messages = [{ "content": "There's a goat in my garden 😱 What should I do?","role": "user"}]

response = litellm.completion(
    model="gemini/gemini-pro",
    messages=messages
)

print(response.choices[0].message.content)