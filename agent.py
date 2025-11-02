import os

from dotenv import load_dotenv
from openai import OpenAI
# from pinecone import Pinecone
import litellm
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .custom_agent import SelfRagAgent
from .prompts import (
    GENERATE_INSTRUCTION,
    GRADE_DOCUMENT_INSTRUCTION,
    QUERY_REWRITER_INSTRUCTION,
    RETRIEVER_INSTRUCTION,
    HALLUCINATION_CHECK_INSTRUCTION,
    RELEVANCE_CHECK_INSTRUCTION,
)
# from .tools.tools import PineconeIndexRetrieval
# old
# from .tools.tools import PineconeIndexRetrieval
from .tools.tools import FaissIndexRetrieval

load_dotenv()

openai = OpenAI()




def get_embedding(text: str) -> list[float]:
    embedding = openai.embeddings.create(model="text-embedding-3-large", input=text)
    return embedding.data[0].embedding

from self_rag.tools.tools import FaissIndexRetrieval

# # Create retriever instance (shared)
# faiss_retriever = FaissIndexRetrieval(
#     index_path=os.environ.get("FAISS_INDEX_PATH"),
#     embedder=get_embedding,
#     top_k=5,
# )


# pinecone_tool = PineconeIndexRetrieval(
#     name="pinecone_retrieval_tool",
#     description="This tool retrieves data from the pinecone vector database.",
#     index_name=os.environ.get("PINECONE_INDEX_NAME"),
#     namespace=os.environ.get("PINECONE_NAMESPACE"),
#     pinecone=Pinecone(),
#     embedder=get_embedding,
#     top_k=5,
# )
faiss_retriever = FaissIndexRetrieval(
    name="faiss_retrieval_tool",
    description="This tool retrieves data from a local FAISS index.",
    index_path=os.environ.get("FAISS_INDEX_PATH"),  # e.g. ./data/faiss_index.index
    embedder=get_embedding,
    top_k=5,
)
# Make it callable for the LlmAgent
def faiss_retrieval_tool(query: str):
    """Tool function for retrieving semantically similar documents from FAISS index."""
    return faiss_retriever.retrieve(query)


# retriever = LlmAgent(
#     name="Retriever",
#     model="gemini-2.0-flash",
#     description="This tool retrieves data from the pinecone vector database.",
#     instruction=RETRIEVER_INSTRUCTION,
#     tools=[pinecone_tool],
#     output_key="retriever_result"
# )
model_name = "deepseek/deepseek-chat-v3.1"
LITELLM_PROXY_API_KEY = os.getenv(
    "LITELLM_PROXY_API_KEY")
litellm.use_litellm_proxy = True
model = LiteLlm(
    model=model_name,
    api_key=LITELLM_PROXY_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)
retriever = LlmAgent(
    name="Retriever",
    model=model,
    description="This tool retrieves data from a FAISS vector index.",
    instruction=RETRIEVER_INSTRUCTION,
    tools=[faiss_retriever],
    output_key="retriever_result"
)

grade_document = LlmAgent(
    name="GradeDocument",
    model=model,
    description="This tool grades documents based on their relevance to a given query and user input.",
    instruction=GRADE_DOCUMENT_INSTRUCTION,
    output_key="grade_document_result"
)

query_rewriter = LlmAgent(
    name="QueryRewriter",
    model=model,
    description="This tool rewrites queries to improve their relevance.",
    instruction=QUERY_REWRITER_INSTRUCTION,
    output_key="query"
)

generate = LlmAgent(
    name="Generate",
    model=model_name,
    description="This tool generates answers based on the retrieved documents and user input.",
    instruction=GENERATE_INSTRUCTION,
    output_key="generate_result"
)

hallucination_checker = LlmAgent(
    name="HallucinationChecker",
    model=model_name,
    description="This tool checks for hallucinations in generated answers.",
    instruction=HALLUCINATION_CHECK_INSTRUCTION,
    output_key="hallucination_check_result"
)

relevance_check = LlmAgent(
    name="RelevanceCheck",
    model=model_name,
    description="This tool checks the relevance of generated answers.",
    instruction=RELEVANCE_CHECK_INSTRUCTION,
    output_key="relevance_check_result"
)

root_agent = SelfRagAgent(
    name="SelfRAGAgent",
    description="This agent performs self-retrieval-augmented generation.",
    retriever=retriever,
    document_grader=grade_document,
    query_rewriter=query_rewriter,
    generator=generate,
    hallucination_checker=hallucination_checker,
    relevence_checker=relevance_check,
    output_key="self_rag_result",
)
