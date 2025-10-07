from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from groq import Groq
from tavily import TavilyClient
import arxiv
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import fitz  # PyMuPDF - CHANGED THIS LINE
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SAMPLE_PDFS'] = 'sample_pdfs'


# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('sample_pdfs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize APIs
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize embedding model for PDF RAG
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index for PDFs
pdf_index = None
pdf_chunks = []
pdf_metadata = []

print("✅ All agents initialized!")

# =========================
# DECISION LOGGER
# =========================
def log_decision(query, decision, rationale, agents_called, results):
    """Log controller decisions"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'decision': decision,
        'rationale': rationale,
        'agents_called': agents_called,
        'results': results
    }
    
    log_file = 'logs/decisions.json'
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Decision logged")

# =========================
# PDF RAG AGENT
# =========================
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    doc = fitz.open(pdf_path)  # Use fitz, not PyMuPDF
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_pdf(filepath, filename):
    """Ingest PDF into FAISS"""
    global pdf_index, pdf_chunks, pdf_metadata
    
    print(f"📄 Ingesting PDF: {filename}")
    
    # Extract and chunk
    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text)
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Initialize or update FAISS index
    if pdf_index is None:
        dimension = embeddings.shape[1]
        pdf_index = faiss.IndexFlatL2(dimension)
    
    pdf_index.add(np.array(embeddings).astype('float32'))
    
    # Store chunks and metadata
    for chunk in chunks:
        pdf_chunks.append(chunk)
        pdf_metadata.append({'filename': filename})
    
    print(f"✅ Added {len(chunks)} chunks from {filename}")
    return len(chunks)

def query_pdf_rag(query, top_k=3):
    """Query PDF RAG system"""
    global pdf_index, pdf_chunks, pdf_metadata
    
    if pdf_index is None or len(pdf_chunks) == 0:
        return {
            'success': False,
            'message': 'No PDFs uploaded yet',
            'documents': []
        }
    
    # Embed query
    query_embedding = embedding_model.encode([query])
    
    # Search FAISS
    distances, indices = pdf_index.search(np.array(query_embedding).astype('float32'), top_k)
    
    # Gather results
    results = []
    context = []
    
    for i, idx in enumerate(indices[0]):
        if idx < len(pdf_chunks):
            chunk = pdf_chunks[idx]
            metadata = pdf_metadata[idx]
            
            results.append({
                'title': f"📄 {metadata['filename']}",
                'snippet': chunk[:250] + '...',
                'similarity': float(1 / (1 + distances[0][i]))  # Convert distance to similarity
            })
            
            context.append(f"[Doc {i+1}] {chunk}")
    
    return {
        'success': True,
        'context': '\n\n'.join(context),
        'documents': results
    }

# =========================
# WEB SEARCH AGENT
# =========================
def web_search_agent(query):
    """Web search using Tavily"""
    print("🌐 Web Search Agent...")
    
    try:
        results = tavily_client.search(query=query, max_results=5)
        
        documents = []
        context = []
        
        for i, result in enumerate(results.get('results', [])):
            documents.append({
                'title': f"🌐 {result.get('title', 'Web Result')}",
                'snippet': result.get('content', '')[:250] + '...',
                'url': result.get('url', '')
            })
            context.append(f"[Web {i+1}] {result.get('title')}\n{result.get('content')}")
        
        return {
            'success': True,
            'context': '\n\n'.join(context),
            'documents': documents
        }
    
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return {
            'success': False,
            'message': str(e),
            'documents': []
        }

# =========================
# ARXIV AGENT
# =========================
def arxiv_agent(query):
    """ArXiv paper search"""
    print("📚 ArXiv Agent...")
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        documents = []
        context = []
        
        for i, paper in enumerate(search.results()):
            authors = ', '.join([a.name for a in paper.authors[:3]])
            if len(paper.authors) > 3:
                authors += ' et al.'
            
            documents.append({
                'title': f"📚 {paper.title}",
                'snippet': f"Authors: {authors}\n\n{paper.summary[:250]}...",
                'url': paper.entry_id,
                'authors': authors,
                'published': str(paper.published.date())
            })
            
            context.append(f"[Paper {i+1}] {paper.title}\nAuthors: {authors}\nAbstract: {paper.summary}")
        
        return {
            'success': True,
            'context': '\n\n'.join(context),
            'documents': documents
        }
    
    except Exception as e:
        print(f"❌ ArXiv error: {e}")
        return {
            'success': False,
            'message': str(e),
            'documents': []
        }

# =========================
# CONTROLLER AGENT
# =========================
def controller_agent(query, use_pdf=False):
    """Main controller - decides which agents to call"""
    
    print(f"\n{'='*60}")
    print(f"🎯 CONTROLLER ANALYZING: {query}")
    print(f"{'='*60}")
    
    query_lower = query.lower()
    agents_to_call = []
    rationale = []
    
    # Rule 1: PDF RAG if user forces or mentions document
    if use_pdf or any(word in query_lower for word in ['pdf', 'document', 'uploaded', 'summarize this']):
        agents_to_call.append('pdf_rag')
        rationale.append("PDF/document keywords detected or explicitly requested")
    
    # Rule 2: ArXiv for research papers
    if any(word in query_lower for word in ['paper', 'papers', 'research', 'arxiv', 'academic', 'study', 'publication']):
        agents_to_call.append('arxiv')
        rationale.append("Academic research papers requested")
    
    # Rule 3: Web search for current info
    if any(word in query_lower for word in ['latest', 'recent', 'current', 'news', 'who is', 'what is', 'today', '2025', '2024', 'now', 'trending']):
        agents_to_call.append('web_search')
        rationale.append("Current/real-time information required")
    
    # Rule 4: If no match, use LLM to decide
    if not agents_to_call:
        print("⚡ Using LLM for routing decision...")
        
        prompt = f"""You are a routing agent. Decide which agents to call for this query.

Available agents:
- pdf_rag: Query uploaded PDF documents
- web_search: Search current web information
- arxiv: Search academic research papers

Query: "{query}"

Respond ONLY with agent names comma-separated, then | then brief reasoning.
Format: agent1,agent2|reasoning

Example: web_search,arxiv|Query needs current info and academic papers"""

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            
            if '|' in result:
                agents_str, reasoning = result.split('|', 1)
                agents_to_call = [a.strip() for a in agents_str.split(',')]
                rationale.append(f"LLM routing: {reasoning.strip()}")
            else:
                agents_to_call = ['web_search']
                rationale.append("LLM routing failed, defaulting to web search")
        
        except Exception as e:
            print(f"❌ LLM routing error: {e}")
            agents_to_call = ['web_search']
            rationale.append("Fallback to web search due to LLM error")
    
    final_rationale = " | ".join(rationale)
    
    print(f"✅ Decision: {agents_to_call}")
    print(f"📝 Rationale: {final_rationale}\n")
    
    return agents_to_call, final_rationale

def synthesize_answer(query, agent_results):
    """Synthesize final answer using Groq"""
    
    # Combine all contexts
    all_context = []
    for result in agent_results:
        if result.get('success') and result.get('context'):
            all_context.append(result['context'])
    
    if not all_context:
        return "Sorry, I couldn't find relevant information to answer your query."
    
    combined_context = "\n\n=== Next Source ===\n\n".join(all_context)
    
    prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the information provided below.

User Question: {query}

Retrieved Information:
{combined_context}

Instructions:
- Provide a clear, comprehensive answer
- Cite sources (e.g., "According to Web Source 1..." or "Based on Paper 2...")
- Be factual and concise
- If information is incomplete, acknowledge it

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return f"Error generating answer: {str(e)}"

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload and ingest PDF"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        chunks_added = ingest_pdf(filepath, filename)
        
        return jsonify({
            'message': 'PDF uploaded and indexed successfully',
            'filename': filename,
            'chunks_added': chunks_added
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_samples', methods=['POST'])
def load_samples():
    """Load sample PDFs"""
    sample_dir = app.config['SAMPLE_PDFS']
    
    if not os.path.exists(sample_dir):
        return jsonify({'error': 'Sample directory not found'}), 404
    
    loaded = 0
    for filename in os.listdir(sample_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(sample_dir, filename)
            try:
                ingest_pdf(filepath, filename)
                loaded += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return jsonify({'loaded': loaded, 'message': f'Loaded {loaded} sample PDFs'})

@app.route('/ask', methods=['POST'])
def ask():
    """Main query endpoint"""
    data = request.json
    query = data.get('query', '').strip()
    use_pdf = data.get('use_uploaded_pdf', False)
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    # Step 1: Controller decides routing
    agents_to_call, rationale = controller_agent(query, use_pdf)
    
    # Step 2: Call selected agents
    agent_results = []
    all_documents = []
    
    for agent in agents_to_call:
        if agent == 'pdf_rag':
            result = query_pdf_rag(query)
            agent_results.append(result)
            if result.get('documents'):
                all_documents.extend(result['documents'])
        
        elif agent == 'web_search':
            result = web_search_agent(query)
            agent_results.append(result)
            if result.get('documents'):
                all_documents.extend(result['documents'])
        
        elif agent == 'arxiv':
            result = arxiv_agent(query)
            agent_results.append(result)
            if result.get('documents'):
                all_documents.extend(result['documents'])
    
    # Step 3: Synthesize answer
    print("🔄 Synthesizing final answer...")
    final_answer = synthesize_answer(query, agent_results)
    
    # Step 4: Log decision
    log_decision(
        query=query,
        decision=f"Called agents: {', '.join(agents_to_call)}",
        rationale=rationale,
        agents_called=agents_to_call,
        results={'num_documents': len(all_documents)}
    )
    
    print(f"✅ Query complete!\n{'='*60}\n")
    
    return jsonify({
        'query': query,
        'agents_used': agents_to_call,
        'rationale': rationale,
        'answer': final_answer,
        'documents': all_documents
    })

@app.route('/logs')
def get_logs():
    """Get decision logs"""
    log_file = 'logs/decisions.json'
    
    if not os.path.exists(log_file):
        return jsonify({'logs': []})
    
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    limit = request.args.get('limit', 10, type=int)
    return jsonify({'logs': logs[-limit:]})

if __name__ == '__main__':
    print("🚀 Flask Multi-Agent System starting...")
    print("📍 Open: http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
