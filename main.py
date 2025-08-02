import os
import json
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field, ValidationError

# FastAPI UI and Static Files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# LangChain Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models for API ---
class Clause(BaseModel):
    name: str = Field(..., description="The name or identifier of the clause.")
    content: str = Field(..., description="The full text of the clause from the document.")
    source_snippet: str = Field(default="", description="A larger snippet from the original document containing this clause.")
    highlight_phrases: List[str] = Field(default=[], description="Key phrases that should be highlighted in the snippet.")
    page_number: Optional[int] = Field(default=None, description="Page number where this clause was found.")
    document_name: Optional[str] = Field(default=None, description="Name of the source document.")

class PolicyDecisionResponse(BaseModel):
    Decision: str = Field(..., description="The final decision on the claim (e.g., 'Approved', 'Rejected', 'More Information Required').")
    Amount: float = Field(..., description="The amount approved for the claim, if applicable. Returns 0 if rejected.")
    Justification: str = Field(..., description="A detailed explanation for the decision, based on the policy document.")
    Clauses: List[Clause] = Field(..., description="A list of relevant clauses from the policy document that support the justification.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query about a medical insurance claim.")
    include_history: bool = Field(default=True, description="Whether to include conversation history for context.")

class UploadResponse(BaseModel):
    message: str = Field(..., description="Status message about the upload.")
    filename: str = Field(..., description="Name of the uploaded file.")
    chunks: int = Field(..., description="Number of text chunks created from the document.")


# --- Global Variables and Setup ---
vector_store = None
uploaded_documents = []
conversation_history = []  # Store conversation history for multi-turn conversations

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    global vector_store
    
    # Initialize with empty vector store - documents will be uploaded later
    vector_store = None
    print("Application initialized. Ready to accept document uploads.")
    
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="LLM Document Processing System",
    description="An API to process natural language queries against policy documents using LLMs.",
    lifespan=lifespan
)

# Mount the static folder to serve static files (like your index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

def create_vector_store_from_documents(documents):
    """
    Creates a Chroma vector store from a list of documents.
    """
    if not documents:
        raise ValueError("No documents provided to create vector store.")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    # Create embeddings using Hugging Face (local and free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and return the Chroma vector store
    db = Chroma.from_documents(docs, embeddings)
    print("Vector store created from uploaded documents.")
    return db

def process_query_with_rag(query: str, vector_store, conversation_history=None):
    """
    Retrieves relevant context and uses an LLM to answer the query in a structured format.
    Now supports multi-turn conversations by including conversation history.
    """
    print(f"Processing query with RAG: {query}")
    
    # 1. Retrieve relevant documents from the vector store
    retriever = vector_store.as_retriever()
    context_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    print(f"Found {len(context_docs)} relevant context chunks.")

    # 2. Define the LLM (Gemini)
    # The API key is automatically picked up from the GOOGLE_API_KEY environment variable.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 3. Build conversation context if history exists
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "\n\nPrevious Conversation History:\n"
        for i, item in enumerate(conversation_history[-3:], 1):  # Include last 3 exchanges
            conversation_context += f"Q{i}: {item['query']}\n"
            conversation_context += f"A{i}: Decision: {item['response']['Decision']}, Amount: {item['response']['Amount']}, Justification: {item['response']['Justification'][:100]}...\n\n"
        
        conversation_context += "Note: The current query might be a follow-up question. Please consider the conversation history when understanding context, references to 'that', 'it', 'the previous case', age/gender mentions, etc.\n"

    # 4. Create the enhanced prompt template
    template = """
    You are an expert at analyzing medical insurance policies with conversational memory.
    Use the following policy document excerpts to determine the outcome of a medical insurance claim.
    
    Policy Document Excerpts:
    {context}
    {conversation_context}
    
    Current Query:
    {query}
    
    Instructions:
    - If this seems like a follow-up question, use the conversation history to understand the full context
    - Look for references like "what about", "for a", age/gender changes, procedure modifications, etc.
    - Maintain consistency with previous responses when appropriate
    - If the query references previous information, incorporate those details
    - For each clause, provide the EXACT text from the document and identify key phrases that support your decision
    
    Your task is to provide a structured JSON response with a final decision, a justified amount, and a list of specific clauses from the policy that support your reasoning.
    The response must follow this exact JSON format:
    {{
      "Decision": "string",
      "Amount": float,
      "Justification": "string",
      "Clauses": [
        {{
          "name": "string",
          "content": "string",
          "source_snippet": "string (larger excerpt from document containing this clause)",
          "highlight_phrases": ["phrase1", "phrase2", "phrase3"],
          "page_number": number or null,
          "document_name": "string or null"
        }}
      ]
    }}
    
    Important for Clauses:
    - "content": The specific clause text that supports your decision
    - "source_snippet": A larger excerpt (2-3 sentences) from the original document that provides context around the clause
    - "highlight_phrases": Array of 2-5 key phrases from the source_snippet that directly support your decision
    - Include page numbers and document names when available from the metadata
    
    Ensure the JSON is perfectly formed, with no trailing commas or extra text.
    The "name" in the Clauses array should be a brief, descriptive name for the clause (e.g., "3-Month Waiting Period", "Surgical Procedures Covered").
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "query", "conversation_context"])

    # 5. Create the chain using the new RunnableSequence syntax
    chain = prompt | llm
    print("Chain created using new RunnableSequence syntax with conversation context.")

    # 6. Invoke the chain and get the response
    response_str = chain.invoke({
        "query": query, 
        "context": context, 
        "conversation_context": conversation_context
    })
    # Extract content from the response object
    if hasattr(response_str, 'content'):
        response_str = response_str.content
    print("Raw LLM response received.")

    # 7. Parse the LLM's response
    try:
        # The LLM often wraps the JSON in a code block. We need to clean this up.
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:-3].strip()
        
        response_json = json.loads(response_str)
        print("Successfully parsed LLM response as JSON.")
        
        # 8. Validate the JSON against the Pydantic model
        decision_response = PolicyDecisionResponse.model_validate(response_json)
        print("Successfully validated JSON against Pydantic model.")
        return decision_response

    except (json.JSONDecodeError, ValidationError) as e:
        print("Error: Failed to parse or validate LLM response.")
        print(f"Exception: {e}")
        print(f"Raw Response:\n---\n{response_str}\n---")
        # For debugging, re-raise the exception to see the full traceback in the terminal
        raise


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the static index.html file for the frontend UI.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document and processes it for querying.
    """
    global vector_store, uploaded_documents
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Check if file already exists to prevent duplicates
    existing_filenames = [doc.metadata.get('source', '').split('/')[-1] for doc in uploaded_documents if hasattr(doc, 'metadata')]
    if file.filename in existing_filenames:
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' has already been uploaded.")
    
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Read file content
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Load the document
        print(f"Processing uploaded file: {file.filename}")
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        
        # Add filename to metadata for tracking
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['filename'] = file.filename
            # Add page number if available
            if 'page' in doc.metadata:
                doc.metadata['page_number'] = doc.metadata['page'] + 1  # Convert to 1-based indexing
            elif hasattr(doc, 'page'):
                doc.metadata['page_number'] = doc.page + 1
            else:
                doc.metadata['page_number'] = i + 1  # Fallback to document chunk index
        
        # Add to uploaded documents list
        uploaded_documents.extend(documents)
        
        # Create/Update vector store with all uploaded documents
        vector_store = create_vector_store_from_documents(uploaded_documents)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Calculate total chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        total_chunks = len(text_splitter.split_documents(uploaded_documents))
        
        print(f"Successfully processed {file.filename}. Total documents: {len(set(doc.metadata.get('filename', '') for doc in uploaded_documents))}")
        
        return UploadResponse(
            message="Document uploaded and processed successfully.",
            filename=file.filename,
            chunks=total_chunks
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        print(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Ensure file is properly closed
        if hasattr(file, 'file') and hasattr(file.file, 'close'):
            file.file.close()

@app.post("/clear-documents")
async def clear_documents():
    """
    Clears all uploaded documents and resets the vector store and conversation history.
    """
    global vector_store, uploaded_documents, conversation_history
    
    uploaded_documents = []
    vector_store = None
    conversation_history = []  # Clear conversation history too
    
    print("All documents and conversation history cleared.")
    return {"message": "All documents and conversation history have been cleared."}

@app.get("/conversation-history")
async def get_conversation_history():
    """
    Returns the conversation history for the current session.
    """
    global conversation_history
    
    return {
        "conversation_history": conversation_history,
        "total_queries": len(conversation_history)
    }

@app.post("/clear-conversation")
async def clear_conversation():
    """
    Clears only the conversation history while keeping documents.
    """
    global conversation_history
    
    conversation_history = []
    
    print("Conversation history cleared.")
    return {"message": "Conversation history has been cleared."}

@app.get("/document-status")
async def get_document_status():
    """
    Returns the current status of uploaded documents and conversation history.
    """
    global uploaded_documents, vector_store, conversation_history
    
    # Get unique filenames
    unique_files = set()
    for doc in uploaded_documents:
        if hasattr(doc, 'metadata') and 'filename' in doc.metadata:
            unique_files.add(doc.metadata['filename'])
    
    return {
        "documents_count": len(unique_files),
        "vector_store_ready": vector_store is not None,
        "status": "Ready for queries" if vector_store is not None else "No documents uploaded",
        "filenames": list(unique_files),
        "conversation_history_count": len(conversation_history)
    }

@app.post("/process-query", response_model=PolicyDecisionResponse)
async def process_document_query(request: QueryRequest):
    """
    Processes a natural language query and retrieves relevant information from uploaded documents.
    Now supports multi-turn conversations with history tracking.
    """
    global vector_store, conversation_history
    
    if vector_store is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents have been uploaded yet. Please upload a document first."
        )
    
    print(f"Received query: {request.query}")
    print(f"Include history: {request.include_history}")
    print(f"Current conversation history length: {len(conversation_history)}")
    
    try:
        # Pass conversation history if requested
        history_to_use = conversation_history if request.include_history else None
        decision = process_query_with_rag(request.query, vector_store, history_to_use)
        
        # Store this query and response in conversation history
        conversation_entry = {
            "query": request.query,
            "response": {
                "Decision": decision.Decision,
                "Amount": decision.Amount,
                "Justification": decision.Justification,
                "Clauses": [{
                    "name": clause.name, 
                    "content": clause.content,
                    "source_snippet": clause.source_snippet,
                    "highlight_phrases": clause.highlight_phrases,
                    "page_number": clause.page_number,
                    "document_name": clause.document_name
                } for clause in decision.Clauses]
            },
            "timestamp": json.dumps(datetime.now(), default=str)  # For potential future use
        }
        conversation_history.append(conversation_entry)
        
        # Keep only last 10 conversation entries to prevent memory bloat
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        print("Query processing completed and added to conversation history.")
        return decision
        
    except Exception as e:
        print(f"An unexpected error occurred during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    
    print(f"Starting FastAPI server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)