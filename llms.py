import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain_openrouter import ChatOpenRouter
from openai import OpenAI
from sentence_transformers import CrossEncoder

load_dotenv()


SYSTEM_PROMPT = (
    "You answer factual questions using only the supplied context. "
    "Extract the explicit answer when it appears in the context."
)


local_llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    device=-1,  # CPU
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False,
        "return_full_text": False,
    },
)

local_model = ChatHuggingFace(llm=local_llm)

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",
                                            model_kwargs={"device": "cpu"},
                                            encode_kwargs={"normalize_embeddings": True})
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

if not api_key:
    raise RuntimeError(
        "La variable OPENROUTER_API_KEY est absente."
    )

if not model_name:
    raise RuntimeError(
        "La variable OPENROUTER_MODEL est absente."
    )

openrouter_llm = ChatOpenRouter(
    model=model_name,
    temperature=0,
)


def answer_question(
    combined_input: str,
    which_model: str = "local",
) -> AIMessage:
    if which_model == "local":
        return answer_question_local(combined_input)

    if which_model == "openrouter":
        return answer_question_openrouter(combined_input)

    raise ValueError(
        f"Modèle inconnu : {which_model!r}. "
        "Valeurs acceptées : 'local' ou 'openrouter'."
    )


def answer_question_local(combined_input: str) -> AIMessage:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=combined_input),
    ]

    response = local_model.invoke(messages)

    if not isinstance(response, AIMessage):
        raise TypeError(
            "Le modèle local n'a pas retourné un AIMessage."
        )

    return response

def get_client_and_model_for_openrouter() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise RuntimeError(
            "La variable OPENROUTER_API_KEY est absente."
        )

    if not model_name:
        raise RuntimeError(
            "La variable OPENROUTER_MODEL est absente."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    return client, model_name


def answer_question_openrouter(combined_input: str) -> AIMessage:

    client, model_name = get_client_and_model_for_openrouter()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": combined_input,
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Le modèle OpenRouter n'a retourné aucune réponse."
        )

    return AIMessage(content=answer)

# We can actually use openrouter via langchain's ChatOpenRouter class.
def get_llm(which_model: str = "local"):
    if which_model == "local":
        return local_model

    if which_model == "openrouter":
        return openrouter_llm

    raise ValueError(
        f"Modèle inconnu : {which_model!r}. "
        "Valeurs acceptées : 'local' ou 'openrouter'."
    )
    

def get_embedding_model():
    return embedding_model

def get_reranker():
    return reranker