# Import the metric
import os
from ragas.metrics import AnswerRelevancy

# Import some additional dependencies
# from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import nest_asyncio
import asyncio
from ragas.integrations.opik import OpikTracer
from ragas.dataset_schema import SingleTurnSample

from opik import track, opik_context
from datasets import load_dataset
from ragas.metrics import context_precision, answer_relevancy, faithfulness
from ragas import evaluate
from utils import configure_env

configure_env(project_name="ragas-integration")

# Initialize the Ragas metric
# llm = LangchainLLMWrapper(ChatOpenAI())
llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        # temperature=0,
        # max_tokens=None,
        # timeout=None,
        # max_retries=2,
    )
)

# emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"))

answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=emb)

# Run this cell first if you are running this in a Jupyter notebook


nest_asyncio.apply()


# Define the scoring function
def compute_metric(metric, row):
    row = SingleTurnSample(**row)

    opik_tracer = OpikTracer(tags=["ragas"])

    async def get_score(opik_tracer, metric, row):
        score = await metric.single_turn_ascore(row, callbacks=[opik_tracer])
        return score

    # Run the async function using the current event loop
    loop = asyncio.get_event_loop()

    result = loop.run_until_complete(get_score(opik_tracer, metric, row))
    return result


# Score a simple example
row = {
    "user_input": "What is the capital of France?",
    "response": "Paris",
    "retrieved_contexts": ["Paris is the capital of France.", "Paris is in France."],
}

score = compute_metric(answer_relevancy_metric, row)
print("Answer Relevancy score:", score)


@track
def retrieve_contexts(question):
    # Define the retrieval function, in this case we will hard code the contexts
    return ["Paris is the capital of France.", "Paris is in France."]


@track
def answer_question(question, contexts):
    # Define the answer function, in this case we will hard code the answer
    return "Paris"


@track(name="Compute Ragas metric score", capture_input=False)
def compute_rag_score(answer_relevancy_metric, question, answer, contexts):
    # Define the score function
    row = {"user_input": question, "response": answer, "retrieved_contexts": contexts}
    score = compute_metric(answer_relevancy_metric, row)
    return score


@track
def rag_pipeline(question):
    # Define the pipeline
    contexts = retrieve_contexts(question)
    answer = answer_question(question, contexts)

    score = compute_rag_score(answer_relevancy_metric, question, answer, contexts)
    opik_context.update_current_trace(
        feedback_scores=[{"name": "answer_relevancy", "value": round(score, 4)}]
    )

    return answer


rag_pipeline("What is the capital of France?")

# Evaluating dataset


fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")

# Reformat the dataset to match the schema expected by the Ragas evaluate function
dataset = fiqa_eval["baseline"].select(range(3))

dataset = dataset.map(
    lambda x: {
        "user_input": x["question"],
        "reference": x["ground_truths"][0],
        "retrieved_contexts": x["contexts"],
    }
)

opik_tracer_eval = OpikTracer(tags=["ragas_eval"], metadata={"evaluation_run": True})

result = evaluate(
    dataset,
    metrics=[context_precision, faithfulness, answer_relevancy],
    callbacks=[opik_tracer_eval],
)

print(result)