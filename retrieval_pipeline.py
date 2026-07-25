from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion_pipeline import get_absolute_path

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

retriever = db.as_retriever(search_kwargs={"k": 5}) #The retriever is configured to return the 
                                                    #top 5 most relevant documents based on cosine similarity.

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"