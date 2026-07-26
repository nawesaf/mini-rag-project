from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from ingestion_pipeline import get_absolute_path
from llms import answer_question, get_llm

load_dotenv()

persistent_directory = get_absolute_path("db/chroma_db")
print(f"Persistent directory: {persistent_directory}")

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",
                                            model_kwargs={"device": "cpu"},
                                            encode_kwargs={"normalize_embeddings": True})
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_name="rag_documents",
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "In what year did Tesla begin production of the Roadster?"

# retriever = db.as_retriever(search_kwargs={"k": 2}) #The retriever is configured to return the 
                                                    #top 2 most relevant documents based on cosine similarity.

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    
    
# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, say "I don't have enough information to 
answer that question based on the provided documents."
"""

# Define the messages for the model
messages = [
    SystemMessage(content="You answer factual questions using only the supplied context. "
                            "Extract the explicit answer when it appears in the context."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
local_result = answer_question(combined_input, "local")
openrouter_result = answer_question(combined_input, "openrouter")

test_get_llms = get_llm("openrouter")  # Use OpenRouter for multi-modal capabilities
get_llm_result = test_get_llms.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(local_result.content)
print("\n--- OpenRouter Response ---")
print("Content only:")
print(openrouter_result.content)
print("\n--- GetLLM Response ---")
print("Content only:")
print(get_llm_result.content)

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"