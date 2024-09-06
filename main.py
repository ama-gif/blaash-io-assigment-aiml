main.py 
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import uuid
import os
from dotenv import load_dotenv
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import json
import random
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
random.seed(5)

# Set index name and dimension to match SentenceTransformer model
index_name = 'my-index'
embedding_dimension = 768  # Set to match SentenceTransformer output dimension
print(f"Using index: {index_name} with dimension: {embedding_dimension}")

# Check or create index
try:
    existing_indexes = pinecone_client.list_indexes()
    if index_name not in existing_indexes:
        pinecone_client.create_index(
            name=index_name,
            dimension=embedding_dimension,  # Ensure correct dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    print(f"Index '{index_name}' is ready.")
except Exception as e:
    print(f"Error while checking or creating index: {e}")

# Initialize Firebase
cred = credentials.Certificate(json.loads(os.getenv('FIREBASE_CONFIG')))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Google Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize SentenceTransformer
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_document():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request")
            flash("No file part")
            return redirect(url_for('index'))

        file = request.files['file']
        chat_name = request.form.get('chat_name')

        if file.filename == '':
            print("No file selected")
            flash("No selected file")
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            try:
                print(f"File {file.filename} received for chat: {chat_name}")
                text = extract_text_from_pdf(file)
                print(f"Extracted text (first 100 chars): {text[:100]}...")

                index_id = index_document(text, chat_name)
                print(f"Document indexed with ID: {index_id}")

                store_metadata(chat_name, index_id)
                session['chat_name'] = chat_name  # Store chat_name in session
                flash("Document uploaded and indexed successfully")
                return redirect(url_for('chat_page'))
            except Exception as e:
                print(f"Error while indexing document: {e}")
                flash(f"An error occurred: {str(e)}")
                return redirect(url_for('index'))
        else:
            print(f"File {file.filename} is not allowed")
            flash("File type not allowed")
            return redirect(url_for('index'))
    else:
        return render_template('upload.html')


@app.route('/chat', methods=['GET'])
def chat_page():
    chat_name = session.get('chat_name')
    if not chat_name:
        flash("No active chat session")
        return redirect(url_for('index'))
    return render_template('chat.html', chat_name=chat_name)


@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    chat_name = data.get('chat_name')
    question = data.get('question')

    if not chat_name or not question:
        print("Missing chat_name or question in the request")
        return jsonify({"error": "Missing chat_name or question"}), 400

    if not validate_question(question):
        print("Invalid question format")
        return jsonify({
            "error":
            "Invalid question. Please ask a clear and relevant question."
        }), 400

    index_id = get_index_id(chat_name)
    if not index_id:
        print(f"Chat not found for name: {chat_name}")
        return jsonify({"error": "Chat not found"}), 404

    context = query_pinecone(index_id, question)
    if not context:
        print(f"No relevant information found for question: {question}")
        return jsonify({"error": "No relevant information found"}), 404

    response = generate_response(context, question)
    print(f"Generated response: {response}")
    return jsonify({"response": response}), 200


# Helper functions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def index_document(text, chat_name):
    index = pinecone_client.Index(index_name)
    sentences = text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    embeddings = model.encode(sentences)

    vector_data = []
    for i, embedding in enumerate(embeddings):
        vector_id = str(uuid.uuid4())
        print(f"Embedding for sentence {i} (first 10 values): {embedding.tolist()[:10]}")  # Debug embedding
        vector_data.append({
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": {
                "text": sentences[i],
                "chat_name": chat_name
            }
        })
        

    batch_size = 50  # Reduce batch size if necessary
    try:
        for i in range(0, len(vector_data), batch_size):
            batch = vector_data[i:i+batch_size]
            print(f"Upserting batch {i // batch_size + 1} of {len(vector_data) // batch_size + 1}")
            index.upsert(vectors=batch)
        return index_name
    except Exception as e:
        print(f"Pinecone error: {e}")  # Simply print the error message
        raise

def store_metadata(chat_name, index_id):
    print(f"Storing metadata for chat: {chat_name}")
    db.collection('metadata').document(chat_name).set({'index_id': index_id})


def validate_question(question):
    is_valid = isinstance(question, str) and len(question.strip()) > 0
    print(f"Question validation result: {is_valid}")
    return is_valid


def get_index_id(chat_name):
    doc = db.collection('metadata').document(chat_name).get()
    if doc.exists:
        print(f"Found index_id for chat: {chat_name}")
        return doc.to_dict().get('index_id')
    else:
        print(f"No index_id found for chat: {chat_name}")
        return None


def query_pinecone(index_id, question):
    index = pinecone_client.Index(index_id)
    query_vector = model.encode([question])[0].tolist()
    print(f"Query vector for question (first 10 values): {query_vector[:10]}")  # Debug query vector
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    if results.matches:
        print(f"Found relevant match: {results.matches[0].metadata['text'][:100]}")  # Print first 100 chars
        return results.matches[0].metadata['text']
    else:
        print("No matches found")
        return ""

def generate_response(context, question):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    print(f"Sending prompt to Google Gemini: {prompt[:100]}...")  # Debug prompt
    response = model.generate_content(prompt)
    return response.text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
