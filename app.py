import os
import requests
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings
from docx import Document
import numpy as np
from googleapiclient.discovery import build
import google.generativeai as genai


# Import deep_translator for Google Translate
from deep_translator import GoogleTranslator

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Extract FAQ from DOCX
def extract_faq_from_docx(file_path):
    doc = Document(file_path)
    faq_dict = {}
    current_question = None
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.endswith('?'):  # Assuming questions end with a question mark
            current_question = text
            faq_dict[current_question] = ""
        elif current_question:
            faq_dict[current_question] += text + " "
    
    # Strip trailing spaces from answers
    for question in faq_dict:
        faq_dict[question] = faq_dict[question].strip()
    
    return faq_dict

# Preprocess FAQ data using GoogleGenerativeAIEmbeddings
def preprocess_faq(faq_data):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    questions = list(faq_data.keys())
    answers = [faq_data[q] for q in questions]
    
    question_embeddings = [embeddings.embed_query(q) for q in questions]
    
    return questions, answers, question_embeddings, embeddings

# Get FAQ suggestions
def get_faq_suggestions(user_query, questions, question_embeddings, embeddings, top_k=5):
    user_query_embedding = embeddings.embed_query(user_query)
    cos_scores = np.dot(question_embeddings, user_query_embedding)
    top_k_indices = np.argsort(cos_scores)[-top_k:][::-1]
    
    return [(questions[idx], cos_scores[idx]) for idx in top_k_indices]

# Get answer from FAQ data
def get_answer(user_query, questions, answers, question_embeddings, embeddings):
    user_query_embedding = embeddings.embed_query(user_query)
    cos_scores = np.dot(question_embeddings, user_query_embedding)
    best_match_idx = np.argmax(cos_scores)
    
    return answers[best_match_idx]

# Translation function using deep_translator
def translate_text(text, target_language):
    try:
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated_text
    except Exception as e:
        return text

# Load text file and return content
def get_text_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return None

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    return splitter.split_text(text)

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_filters():
    filters = set()
    with open('filter_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            filters.add(line.strip())
    return filters

filters = load_filters()

def contains_filtered_content(text):
    for filter_word in filters:
        if filter_word in text.lower():
            return True
    return False

# Create a conversational chain
def get_conversational_chain():
    prompt_template = """
    You are an expert on the JJM Operational Guidelines. Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "The answer is not available in the context." Do not provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0, max_tokens=1000, top_p=0.98, top_k=50, stop_sequences=["\n"])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Global initialization flag
initialized = False

@app.before_request
def before_request():
    global initialized
    if not initialized:
        file_path = "JJM_Operational_Guidelines.txt"
        faq_file_path = 'test.docx'

        raw_text = get_text_content(file_path)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        
        # Load and preprocess FAQ data
        faq_data = extract_faq_from_docx(faq_file_path)
        global questions, answers, question_embeddings, embeddings
        questions, answers, question_embeddings, embeddings = preprocess_faq(faq_data)
        
        initialized = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data['question']
    selected_language = data['language']

    # Attempt to get answer using Gemini API
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response['output_text']
    
    translated_answer = translate_text(answer, selected_language)
    
    if "The answer is not available in the context." in answer:
        # Get FAQ suggestions
        suggestions = get_faq_suggestions(user_question, questions, question_embeddings, embeddings)
        suggested_questions = [q for q, _ in suggestions]
        translated_suggestions = [translate_text(q, selected_language) for q in suggested_questions]
        
        # Perform YouTube search for related videos
        youtube_videos = youtube_search(user_question, selected_language)
        
        return jsonify({
            "answer": translated_answer,
            "available": False,
            "faq_available": True,
            "suggestions": translated_suggestions,
            "videos": youtube_videos
        })
    else:
        # Perform YouTube search for related videos
        youtube_videos = youtube_search(user_question, selected_language)
        
        return jsonify({
            "answer": translated_answer,
            "available": True,
            "faq_available": False,
            "videos": youtube_videos
        })

def youtube_search(query: str, language: str):
    api_key = os.getenv('YOUTUBE_API_KEY')
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    
    translated_query = translate_text(query, language) + " Jal jeevan mission "
    
    search_params = {
        'part': 'snippet',
        'q': translated_query,
        'key': api_key,
        'maxResults': 5,
        'type': 'video'
    }
    
    search_response = requests.get(search_url, params=search_params)
    search_results = search_response.json()

    video_ids = [item['id']['videoId'] for item in search_results['items']]
    
    video_params = {
        'part': 'snippet,statistics',
        'id': ','.join(video_ids),
        'key': api_key
    }
    
    video_response = requests.get(video_url, params=video_params)
    video_results = video_response.json()
    
    videos = []
    for item in video_results.get('items', []):
        title = translate_text(item['snippet']['title'], language)
        # Check if title contains filtered content
        if not contains_filtered_content(title):
            video_data = {
                'title': title,
                'url': f"https://www.youtube.com/watch?v={item['id']}",
            }
            videos.append(video_data)
    
    return videos

@app.route('/faq_answer', methods=['POST'])
def faq_answer():
    data = request.get_json()
    user_question = data['question']
    selected_language = data['language']
    
    # Get answer from FAQ data
    answer = get_answer(user_question, questions, answers, question_embeddings, embeddings)
    translated_answer = translate_text(answer, selected_language)
    
    return jsonify({"answer": translated_answer})

@app.route('/youtube_search', methods=['POST'])
def youtube_search_route():
    data = request.get_json()
    question = data.get('question', '')
    language = data.get('language', 'en')
    youtube_videos = youtube_search(question, language)
    return jsonify({'youtube_videos': youtube_videos})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
