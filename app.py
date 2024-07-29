from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import os
from werkzeug.security import check_password_hash
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.schema import Document 

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Set your secret key here

# Initialize your existing code
directory = 'pdfstore'

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

def chunk_data(docs, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

doc = read_doc(directory)
documents = chunk_data(docs=doc)

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [d.page_content for d in documents]
embeddings = model.encode(texts, convert_to_tensor=True)

api_key = os.getenv('PINECONE_API_KEY')
if api_key is None:
    raise ValueError("Pinecone API key not found. Set the 'PINECONE_API_KEY' environment variable.")

pc = Pinecone(api_key=api_key)

if 'langchainbot' not in pc.list_indexes().names():
    pc.create_index(
        name='langchainbot',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index_name = 'langchainbot'
index = pc.Index(index_name)

vectors_to_upsert = []
for i, (doc_, embedding) in enumerate(zip(documents, embeddings.tolist())):
    vector_id = str(i)
    metadata = {"text": doc_.page_content}
    vectors_to_upsert.append((vector_id, embedding, metadata))

index.upsert(vectors=vectors_to_upsert)

def embed_query(query):
    return model.encode(query).tolist()

def retrieve_query(query, k=2):
    query_vector = embed_query(query)
    matching_results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True
    )
    return matching_results

groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
chain = load_qa_chain(llm, chain_type="stuff")

def retrieve_answers(query):
    doc_search = retrieve_query(query)
    docs = [
        Document(page_content=result['metadata'].get('text', ''))
        for result in doc_search.get('matches', [])
        if isinstance(result, dict) and 'metadata' in result
    ]
    response = chain.run(input_documents=docs, question=query)
    return response

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    stored_username = os.getenv('ADMIN_USERNAME')
    stored_password_hash = os.getenv('ADMIN_PASSWORD_HASH')

    if username == stored_username and check_password_hash(stored_password_hash, password):
        session['username'] = username
        session['chat_history'] = []  # Initialize chat history in session
        return redirect(url_for('chat'))
    
    # Redirect back to login page with error message
    return render_template('login.html', error='Invalid credentials')


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('chat_history', None)
    return redirect(url_for('home'))


@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    try:
        answer = retrieve_answers(user_query)
        # Save the question and answer to session chat history
        if 'chat_history' in session:
            session['chat_history'].append({'user': user_query, 'bot': answer})
        else:
            session['chat_history'] = [{'user': user_query, 'bot': answer}]
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    if 'chat_history' in session:
        return jsonify(session['chat_history'])
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
