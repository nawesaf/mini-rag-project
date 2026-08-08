
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma

# LangChain components
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from unstructured.chunking.title import chunk_by_title

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf

from llms import get_embedding_model, get_llm

logger = logging.getLogger(__name__)
load_dotenv()

class DocumentNotFoundError(Exception):
    pass


class AnswerGenerationError(Exception):
    pass

PERSIST_DIRECTORY = "db/appdb"

def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding_model(),
        collection_name="app_documents",
    )

def partition_document(file_path: str, strategy: str='hi_res'):
    """Extract elements from PDF using unstructured"""
    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy=strategy, # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )
    return elements

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )
    return chunks

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    # Initialize LLM
    llm = get_llm("openrouter")
    
    # Build the text prompt
    prompt_text = f"""You are creating a searchable description for document content retrieval.

    CONTENT TO ANALYZE:
    TEXT CONTENT:
    {text}

    """
    
    # Add tables if present
    if tables:
        prompt_text += "TABLES:\n"
        for i, table in enumerate(tables):
            prompt_text += f"Table {i+1}:\n{table}\n\n"
    
    prompt_text += """
    YOUR TASK:
    Generate a comprehensive, searchable description that covers:

    1. Key facts, numbers, and data points from text and tables
    2. Main topics and concepts discussed  
    3. Questions this content could answer
    4. Visual content analysis (charts, diagrams, patterns in images)
    5. Alternative search terms users might use

    Make it detailed and searchable - prioritize findability over brevity."""

    # Build message content starting with text
    message_content: list[str | dict[str,Any]]= [{"type": "text", "text": prompt_text}]
    
    # Add images to the message
    for image_base64 in images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })
    
    # Send to AI and get response
    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    content = response.content
    if not isinstance(content, str):
        raise TypeError("The language model did not return text content")
    return content

def summarise_chunk(chunk, document_id):
    content_data = separate_content_types(chunk)
    # Create AI-enhanced summary if chunk has tables/images
    if content_data['tables'] or content_data['images']:
        try:
            enhanced_content = create_ai_enhanced_summary(
                content_data['text'],
                content_data['tables'], 
                content_data['images']
            )
        except Exception:
            logger.exception(
                "Error enhancing chunk for document %s",
                document_id,
            )            
        enhanced_content = content_data['text']
    else:
        enhanced_content = content_data['text']
    return Document(
        page_content=enhanced_content,
        metadata={
            "original_content": json.dumps({
                "raw_text": content_data["text"],
                "tables_html": content_data["tables"],
                "images_base64": content_data["images"],
            }),
            "document_id": document_id,
        },
    )
    
def summarise_chunks(chunks, document_id, max_workers: int = 3):
    """Process all chunks with AI Summaries"""    
        
        # Create LangChain Document with rich metadata
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        langchain_documents = list(
            executor.map(summarise_chunk, chunks, repeat(document_id))
        )
    return langchain_documents


def run_complete_ingestion_pipeline(pdf_path: str, document_id: str):
    """Run the complete RAG ingestion pipeline"""
    
    # Step 1: Partition
    elements = partition_document(pdf_path)
    
    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)
    
    # Step 3: AI Summarisation
    summarised_chunks = summarise_chunks(chunks, document_id)
    
    # Step 4: Vector Store
    vector_store = get_vector_store()
    vector_store.add_documents(summarised_chunks)

def get_answer(query, document_id):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3, 
                       "filter": {"document_id": document_id}}
    )
    chunks = retriever.invoke(query)
    if not chunks:
        raise DocumentNotFoundError(
            f"No chunks found for document {document_id}"
        )
    response = generate_final_answer(chunks, query)
    return response
    
def generate_final_answer(chunks, query) -> str:
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = get_llm("openrouter")  # Use OpenRouter for multi-modal capabilities
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content: list[str | dict[str,Any]] = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        content = response.content
        if not isinstance(content, str):
            raise TypeError("The language model did not return text content")
        return content
        
    except Exception as exc:
        logger.exception("Error generating final answer")
        raise AnswerGenerationError(
            "Unable to generate an answer"
        ) from exc


