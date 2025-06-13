import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse, HTMLResponse



# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.2 # Lowered threshold for better recall
MAX_RESULTS = 15  # Increased to get more context
load_dotenv()
MAX_CONTEXT_CHUNKS = 10  # Increased number of chunks per source
API_KEY = os.getenv("API_KEY")  # Get API key from environment variable

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]


# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")



# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database exists or create it
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )
    ''')
    
    # Create markdown_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()


# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays with float type for safety
        vec1 = np.array(vec1, dtype=float)
        vec2 = np.array(vec2, dtype=float)
        
        # Handle zero vectors explicitly
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            logger.debug("One of the vectors is zero; returning similarity 0.0")
            return 0.0

        # Calculate dot product and norms
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Check norms again for safety
        if norm_vec1 == 0 or norm_vec2 == 0:
            logger.debug("Norm is zero after conversion; returning similarity 0.0")
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return zero similarity on any failure

# Function to get embedding from AIPipe proxy with retry mechanism
async def get_embedding(text: str, max_retries: int = 3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }

    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Attempt {retries + 1}: Requesting embedding for text length {len(text)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached. Retrying after backoff... ({retries + 1})")
                        await asyncio.sleep(5 * (retries + 1))  # exponential backoff
                        retries += 1
                        continue

                    else:
                        error_text = await response.text()
                        logger.error(f"Embedding API Error {response.status}: {error_text}")
                        raise HTTPException(status_code=response.status, detail=error_text)

        except Exception as e:
            logger.error(f"Exception during embedding request (attempt {retries + 1}/{max_retries}): {str(e)}")
            logger.debug(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail="Failed to get embedding after multiple attempts")
            await asyncio.sleep(3 * retries)  # linear backoff

    raise HTTPException(status_code=500, detail="Unexpected error in embedding logic")
# Function to find similar content in the database with improved logic
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        processed_count = 0
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted
                    url = chunk["url"]
                    if not url.startswith("http"):
                        # Fix missing protocol
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(discourse_chunks)} discourse chunks")
                    
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
        
        # Search markdown chunks
        logger.info("Querying markdown chunks")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks")
        processed_count = 0
        
        for chunk in markdown_chunks:
            try:
                embedding = np.frombuffer(chunk["embedding"], dtype=np.float32).tolist()
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Ensure URL is properly formatted
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        # Use a default URL if missing
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(markdown_chunks)} markdown chunks")
                    
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        # Group by source document and keep most relevant chunks
        grouped_results = {}
        
        for result in results:
            # Create a unique key for the document/post
            if result["source"] == "discourse":
                key = f"discourse_{result['post_id']}"
            else:
                key = f"markdown_{result['title']}"
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        # For each source, keep only the most relevant chunks
        final_results = []
        for key, chunks in grouped_results.items():
            # Sort chunks by similarity
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            # Keep top chunks
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        # Sort again by similarity
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top results, limited by MAX_RESULTS
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping")
        return final_results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

# Function to enrich content with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []

        for result in results:
            enriched_result = result.copy()
            additional_content = []

            # Common chunk indexing and fetching
            current_chunk_index = result["chunk_index"]

            if result["source"] == "discourse":
                post_id = result["post_id"]

                # Fetch previous chunk
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content FROM discourse_chunks 
                        WHERE post_id = ? AND chunk_index = ?
                    """, (post_id, current_chunk_index - 1))
                    prev = cursor.fetchone()
                    if prev:
                        additional_content.append(prev["content"])

                # Fetch next chunk
                cursor.execute("""
                    SELECT content FROM discourse_chunks 
                    WHERE post_id = ? AND chunk_index = ?
                """, (post_id, current_chunk_index + 1))
                next_ = cursor.fetchone()
                if next_:
                    additional_content.append(next_["content"])

            elif result["source"] == "markdown":
                title = result["title"]

                # Fetch previous chunk
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content FROM markdown_chunks 
                        WHERE doc_title = ? AND chunk_index = ?
                    """, (title, current_chunk_index - 1))
                    prev = cursor.fetchone()
                    if prev:
                        additional_content.append(prev["content"])

                # Fetch next chunk
                cursor.execute("""
                    SELECT content FROM markdown_chunks 
                    WHERE doc_title = ? AND chunk_index = ?
                """, (title, current_chunk_index + 1))
                next_ = cursor.fetchone()
                if next_:
                    additional_content.append(next_["content"])

            # Combine original and additional content
            if additional_content:
                enriched_result["content"] = f"{result['content']} {' '.join(additional_content)}"

            enriched_results.append(enriched_result)

        logger.info(f"Successfully enriched {len(enriched_results)} results")
        return enriched_results

    except Exception as e:
        logger.error(f"Error in enrich_with_adjacent_chunks: {e}")
        logger.error(traceback.format_exc())
        raise

# Function to generate an answer using LLM with improved prompt
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            
            # Construct context string
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                snippet = result["content"][:1500].strip().replace("\n", " ")
                context += f"\n\n{source_type} (URL: {result['url']}):\n{snippet}"

            # Prompt engineering
            prompt = f"""
Answer the following question based ONLY on the provided context.
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Return your response in this exact format:
1. A comprehensive yet concise answer
2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer

Sources must be in this exact format:
Sources:
1. URL: [exact_url_1], Text: [brief quote or description]
2. URL: [exact_url_2], Text: [brief quote or description]

Make sure the URLs are copied exactly from the context without any changes.
""".strip()

            # Prepare API call
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,  # Lower temperature for more focused answers
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        logger.warning(f"Rate limit hit, retry {retries+1}")
                        await asyncio.sleep(3 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API error {response.status}: {error_text}")
                        raise HTTPException(status_code=response.status, detail=error_text)

        except Exception as e:
            logger.error(f"Exception during answer generation: {e}")
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=f"Failed after retries: {str(e)}")
            await asyncio.sleep(2)


# Function to process multimodal content (text + image)
async def process_multimodal_query(question, image_base64):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")

        if not image_base64:
            logger.info("No image provided, processing as text-only query")
            return await get_embedding(question)

        logger.info("Processing multimodal query with image")

        # Use correct media type (adjust based on actual input format: jpeg, png, webp, etc.)
        image_content = f"data:image/webp;base64,{image_base64}"  # change to image/jpeg if needed

        # GPT-4o Vision input payload
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Look at this image and tell me what you see relevant to the question: {question}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_content
                            }
                        }
                    ]
                }
            ]
        }

        logger.info("Sending multimodal request to GPT-4o-mini")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image response: {image_description[:100]}")

                    # Combine question and image analysis
                    combined_query = f"{question}\nImage context: {image_description}"
                    return await get_embedding(combined_query)

                else:
                    error_text = await response.text()
                    logger.error(f"Image processing failed (status {response.status}): {error_text}")
                    logger.info("Falling back to text-only query")
                    return await get_embedding(question)

    except Exception as e:
        logger.error(f"Exception during multimodal query: {e}")
        logger.error(traceback.format_exc())
        logger.info("Falling back to text-only query due to exception")
        return await get_embedding(question)

def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")

        # Normalize the response line endings
        response = response.replace("\r\n", "\n").replace("\r", "\n")

        # Attempt to split response by "Sources:" or variants
        split_markers = ["Sources:", "Source:", "References:", "Reference:"]
        parts = [response, ""]  # default fallback
        
        for marker in split_markers:
            if marker in response:
                parts = response.split(marker, 1)
                break

        answer = parts[0].strip()
        links = []

        if len(parts) > 1 and parts[1].strip():
            sources_text = parts[1].strip()
            source_lines = [line.strip() for line in sources_text.split("\n") if line.strip()]

            for line in source_lines:
                # Remove bullet or numeric list markers
                line = re.sub(r'^\s*[\-\*\d]+\.\s*', '', line)

                # Extract URL using various patterns
                url_match = re.search(
                    r'URL:\s*\[?(https?://[^\]\s]+)\]?', line, re.IGNORECASE
                ) or re.search(
                    r'\[(https?://[^\]]+)\]', line
                ) or re.search(
                    r'(https?://[^\s]+)', line
                )

                # Extract Text using various patterns
                text_match = re.search(
                    r'Text:\s*\[?(.*?)\]?', line, re.IGNORECASE
                ) or re.search(
                    r'["“](.*?)["”]', line
                )

                url = url_match.group(1).strip() if url_match else None
                text = text_match.group(1).strip() if text_match else "Source reference"

                if url:
                    links.append({"url": url, "text": text})

        logger.info(f"Parsed answer (length: {len(answer)}) and {len(links)} sources")
        return {"answer": answer, "links": links}

    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }



# Define API routes
@app.post("/api/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    try:
        logger.info(f"Received query request: question='{request.question[:50]}...', image_provided={request.image is not None}")
        
        if not API_KEY:
            error_msg = "API_KEY environment variable not set"
            logger.error(error_msg)
            return JSONResponse(
                status_code=500,
                content={"answer": "API key not configured.", "links": []}
            )

        conn = get_db_connection()

        try:
            logger.info("Processing query and generating embedding")
            query_embedding = await process_multimodal_query(
                request.question,
                request.image
            )

            logger.info("Finding similar content")
            relevant_results = await find_similar_content(query_embedding, conn)

            if not relevant_results:
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }

            logger.info("Enriching results with adjacent chunks")
            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)

            logger.info("Generating answer")
            llm_response = await generate_answer(request.question, enriched_results)

            logger.info("Parsing LLM response")
            result = parse_llm_response(llm_response)

            # Ensure result is valid
            if not result or "answer" not in result:
                result = {
                    "answer": "I couldn't generate a meaningful answer.",
                    "links": []
                }

            if not result.get("links"):
                logger.info("No links extracted, creating from relevant results")
                links = []
                seen = set()
                for res in relevant_results[:5]:
                    url = res["url"]
                    if url not in seen:
                        seen.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                result["links"] = links

            logger.info(f"Returning result: answer_length={len(result['answer'])}, num_links={len(result['links'])}")
            return result

        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"answer": "Something went wrong while processing your request.", "links": []}
            )

        finally:
            conn.close()

    except Exception as e:
        error_msg = f"Unhandled exception: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"answer": "Unexpected server error.", "links": []}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Try to connect to the database as part of health check
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        # Check if any embeddings exist
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>TDS Virtual Teaching Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f9f9f9; padding: 40px; }
                h1 { color: #2c3e50; }
                p { font-size: 16px; }
                code { background-color: #eee; padding: 2px 4px; border-radius: 4px; }
                .container { max-width: 800px; margin: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>👩‍🏫 TDS Virtual Teaching Assistant</h1>
                <p>Welcome! This is an API-based teaching assistant for the <strong>Tools in Data Science</strong> course (Jan 2025).</p>
                <p>Use the <code>/api/</code> endpoint to ask a question via POST request.</p>

                <h3>Sample usage (cURL):</h3>
                <pre><code>curl -X POST http://127.0.0.1:8000/api/ \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What is GPT-4?", "image": null}'</code></pre>

                <p>You can also explore the API through the <a href="/docs">interactive Swagger UI</a>.</p>
            </div>
        </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True) 