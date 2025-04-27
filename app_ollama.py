# app_ollama.py (No Vision)

import streamlit as st
import os
import time
from pathlib import Path
import io # May not be needed now, but harmless

# --- Page Config (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Agri AI Assistant", layout="wide")

# LangChain components
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma # Consider switching to langchain_chroma later
from langchain.prompts import PromptTemplate

# Optional: Translation
# Ensure 'transformers' and 'sentencepiece' are installed: pip install "transformers[sentencepiece]"
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Don't use st.warning here before the main app layout starts
    print("WARNING: Transformers library not found. Translation feature disabled. Install with: pip install \"transformers[sentencepiece]\"")
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None # Define it as None if import fails

# --- Configuration ---
# LLM Configuration
OLLAMA_MODEL_NAME = "mistral:7b-instruct-q4_K_M" # Ensure this model is pulled in Ollama
TEMPERATURE = 0.7

# RAG Configuration
PDF_DATA_PATH = "data/"
VECTOR_DB_PATH = "vectorstores/db_chroma"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Local embedding model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 3 # Number of context chunks to retrieve

# Translation Configuration (Optional)
TRANSLATION_MODEL_EN_HI = "Helsinki-NLP/opus-mt-en-hi" # English to Hindi

# --- Helper Functions (LLM, Embeddings, RAG, Translation) ---

@st.cache_resource # Cache the LLM client instance
def load_llm():
    """Loads the LangChain Ollama client"""
    st.info(f"Initializing Ollama client for model: {OLLAMA_MODEL_NAME}...")
    st.info("Ensure the Ollama application server is running.")
    try:
        llm = Ollama(
            model=OLLAMA_MODEL_NAME,
            temperature=TEMPERATURE,
            # base_url="http://localhost:11434" # Default Ollama API URL
        )
        # Optional: Add a quick connection test here if desired
        st.success(f"Ollama client initialized for model {OLLAMA_MODEL_NAME}.")
        st.success("GPU acceleration is handled by the Ollama server.")
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama client: {e}")
        st.error("Ensure the Ollama application is running and necessary packages (e.g., langchain-community) are installed.")
        st.stop()

@st.cache_resource
def load_embeddings():
    """Loads the Sentence Transformer embeddings"""
    st.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.info("Embedding model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

def data_ingestion(pdf_folder_path, vector_db_path, embeddings):
    """Loads PDFs, splits text, creates embeddings, and stores in Chroma"""
    vectorstore_cls = Chroma
    vector_store_exists = os.path.exists(vector_db_path) and os.listdir(vector_db_path)

    if not vector_store_exists:
        st.info(f"No existing vector store found or empty at {vector_db_path}. Ingesting data...")
        if not os.path.exists(pdf_folder_path) or not os.listdir(pdf_folder_path):
             st.error(f"PDF data folder is empty or missing: {pdf_folder_path}. Please create it and add PDF files.")
             st.stop()
        pdf_files = list(Path(pdf_folder_path).glob('./*.pdf'))
        if not pdf_files:
             st.error(f"No PDF files found in {pdf_folder_path}. Please add PDF documents.")
             st.stop()
        st.info(f"Found {len(pdf_files)} PDF files for ingestion.")
        loader = DirectoryLoader(pdf_folder_path, glob='./*.pdf', loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
        try:
            documents = loader.load()
        except Exception as load_err:
             st.error(f"Error loading PDFs from {pdf_folder_path}: {load_err}")
             st.exception(load_err)
             st.stop()
        if not documents: st.error(f"No documents were successfully loaded from PDFs in {pdf_folder_path}."); st.stop()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        if not texts: st.error("Failed to split documents into text chunks."); st.stop()
        st.info(f"Loaded {len(documents)} documents, split into {len(texts)} chunks.")
        st.info("Creating vector store (this may take several minutes for large documents)...")
        try:
            vectorstore = vectorstore_cls.from_documents(texts, embeddings, persist_directory=vector_db_path)
            st.success(f"Vector store created and persisted at {vector_db_path}")
            return vectorstore
        except Exception as vs_err: st.error(f"Failed to create vector store: {vs_err}"); st.exception(vs_err); st.stop()
    else:
        st.info(f"Loading existing vector store from {vector_db_path}")
        try:
            vectorstore = vectorstore_cls(persist_directory=vector_db_path, embedding_function=embeddings)
            st.success("Existing vector store loaded.")
            return vectorstore
        except Exception as vs_load_err: st.error(f"Failed to load existing vector store: {vs_load_err}"); st.error(f"Try deleting the vector store directory '{vector_db_path}' and restarting."); st.exception(vs_load_err); st.stop()

# Custom Prompt Template for RAG
rag_prompt_template_str = """SYSTEM: You are an AI assistant providing agricultural advice. Use the following pieces of context ONLY to answer the user's question. If you don't know the answer from the context, just say that you don't know, don't try to make up an answer. Provide concise and actionable advice.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
RAG_PROMPT = PromptTemplate(template=rag_prompt_template_str, input_variables=["context", "question"])

@st.cache_resource # Cache the QA chain
def create_rag_chain(_llm, _vectorstore):
    """Creates the RetrievalQA chain"""
    st.info("Creating RAG chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type='stuff',
            retriever=_vectorstore.as_retriever(search_kwargs={'k': SEARCH_K}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': RAG_PROMPT}
        )
        st.success("RAG chain created.")
        return qa_chain
    except Exception as e:
        st.error(f"Failed to create RAG chain: {e}")
        st.exception(e)
        st.stop()

# --- Prompt Engineering Module ---
def generate_crop_report_prompt(crop, stage, weather, observations):
  """Generates a prompt for a simulated field report"""
  return f"""
  SYSTEM: You are an agricultural expert simulating a field report.
  TASK: Generate a brief field report based on the following details. Focus on potential risks and next steps. Be concise and practical for a farmer.
  DETAILS:
  - Crop: {crop}
  - Growth Stage: {stage}
  - Recent Weather: {weather}
  - Observations: {observations}
  REPORT:
  """

def generate_pest_suggestion_prompt(crop, region, weather_forecast):
  """Generates a prompt for pest suggestions"""
  return f"""
  SYSTEM: You are an AI assistant predicting potential pest outbreaks based on general knowledge.
  TASK: Based on the crop, region, and weather forecast, list 2-3 likely pests and suggest one simple preventative or early monitoring measure for each. Assume typical conditions for the region if specific data is missing.
  DETAILS:
  - Crop: {crop}
  - Region: {region}
  - Weather Forecast: {weather_forecast}
  PREDICTION and SUGGESTIONS:
  """

# --- Optional: Translation Functions ---
@st.cache_resource
def load_translator(model_name=TRANSLATION_MODEL_EN_HI):
    """Loads a Hugging Face translation pipeline"""
    if not TRANSFORMERS_AVAILABLE or not hf_pipeline:
        return None
    st.info(f"Loading translator model: {model_name}")
    try:
        translator = hf_pipeline("translation", model=model_name)
        st.success("Translator loaded.")
        return translator
    except Exception as e:
        st.warning(f"Could not load translator model '{model_name}': {e}")
        st.warning("Translation feature will be unavailable.")
        return None

def translate_text(text, translator):
    """Translates text using the loaded pipeline"""
    if translator and text:
        try:
            result = translator(text, max_length=512)
            if result and isinstance(result, list) and 'translation_text' in result[0]:
                return result[0]['translation_text']
            else: st.error(f"Unexpected translation result format: {result}"); return f"[Translation Format Error] {text}"
        except Exception as e: st.error(f"Translation failed: {e}"); return f"[Translation Exception] {text}"
    elif not translator:
         if text: st.warning("Translator not loaded, cannot translate.")
         return text
    else: return ""

# --- Main App Logic ---

# Title and Markdown moved after set_page_config
st.title("🌾 Generative AI Assistant for Sustainable Agriculture")
st.markdown("Powered by Local LLMs (**Ollama**) and RAG")

# Initialization (runs once or when cache expires)
llm = load_llm()
embeddings = load_embeddings()
# Create vector store directory if it doesn't exist
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
vectorstore = data_ingestion(PDF_DATA_PATH, VECTOR_DB_PATH, embeddings)
qa_chain = create_rag_chain(llm, vectorstore)
translator_hi = load_translator(TRANSLATION_MODEL_EN_HI)
# Vision parts removed

# --- Task Selection Sidebar ---
st.sidebar.header("Select Task")
# Vision option removed
task_options = ["Ask Questions (RAG)", "Generate Report/Suggestion"]
# Removed vision_model_loaded check
selected_task = st.sidebar.selectbox("Choose what you want to do:", task_options, key="task_selector") # Added key

# --- Main Interaction Area (Conditional Display based on selection) ---

if selected_task == "Ask Questions (RAG)":
    st.header("Ask Questions about Crop Diseases, Pests, Soil")
    st.info("The AI will answer based on the information contained in the ingested PDF documents.")
    user_query = st.text_input("Enter your question in English:", key="rag_query")
    # Disable checkbox if translator failed to load
    translate_output = st.checkbox("Translate answer to Hindi?", key="rag_translate", disabled=(translator_hi is None))
    if translator_hi is None and TRANSFORMERS_AVAILABLE:
         st.warning("Hindi translation model failed to load. Translation unavailable.")
    elif not TRANSFORMERS_AVAILABLE:
         st.warning("Transformers library not installed. Translation unavailable.")


    if st.button("Get Answer", key="rag_submit"):
        if user_query:
            with st.spinner("Searching documents and generating answer via Ollama..."):
                start_time = time.time()
                try:
                    result = qa_chain.invoke({"query": user_query})
                    end_time = time.time()
                    st.subheader("Answer:")
                    answer = result.get("result", "Error: No answer found in result.")
                    st.write(answer)

                    if translate_output and translator_hi:
                         with st.spinner("Translating to Hindi..."):
                             hindi_answer = translate_text(answer, translator_hi)
                             st.subheader("उत्तर (Hindi):")
                             st.write(hindi_answer)

                    st.write(f"Response generated in {end_time - start_time:.2f} seconds.")

                    with st.expander("Show Sources (Retrieved Document Chunks)"):
                        source_docs = result.get('source_documents', [])
                        if source_docs:
                            for doc in source_docs:
                                source_name = doc.metadata.get('source', 'N/A')
                                page_num = doc.metadata.get('page', 'N/A')
                                if page_num != 'N/A': page_num += 1 # Adjust page num if 0-indexed
                                st.markdown(f"**Source:** {os.path.basename(source_name)}, Page: {page_num}")
                                st.markdown(f"> {doc.page_content[:500]}...")
                        else: st.write("No source documents information available.")
                except Exception as e: st.error(f"An error occurred during RAG: {e}"); st.exception(e)
        else: st.warning("Please enter a question.")

elif selected_task == "Generate Report/Suggestion":
    st.header("Generate Simulated Reports or Suggestions")
    st.info("Uses prompt engineering to generate text based on your inputs.")
    gen_type = st.radio("Select Generation Type:", ["Field Report", "Pest Suggestion"], horizontal=True, key="gen_type_radio")
    # Disable checkbox if translator failed to load
    translate_output_gen = st.checkbox("Translate output to Hindi?", key="gen_translate", disabled=(translator_hi is None))
    if translator_hi is None and TRANSFORMERS_AVAILABLE:
         st.warning("Hindi translation model failed to load. Translation unavailable.")
    elif not TRANSFORMERS_AVAILABLE:
         st.warning("Transformers library not installed. Translation unavailable.")

    if gen_type == "Field Report":
        st.subheader("Field Report Details")
        # Using unique keys for widgets within the same 'if' block
        crop = st.text_input("Crop Name:", "Tomato", key="report_crop")
        stage = st.text_input("Growth Stage:", "Flowering", key="report_stage")
        weather = st.text_input("Recent Weather:", "Hot and Humid, recent rain", key="report_weather")
        observations = st.text_area("Observations:", "Some yellowing leaves...", key="report_obs")
        if st.button("Generate Report", key="gen_report"):
             if all([crop, stage, weather, observations]):
                 prompt = generate_crop_report_prompt(crop, stage, weather, observations)
                 with st.spinner("Generating report via Ollama..."):
                     start_time = time.time()
                     try:
                        response = llm.invoke(prompt)
                        end_time = time.time()
                        st.subheader("Generated Report:")
                        st.write(response)
                        if translate_output_gen and translator_hi:
                            with st.spinner("Translating to Hindi..."):
                                hindi_response = translate_text(response, translator_hi)
                                st.subheader("रिपोर्ट (Hindi):")
                                st.write(hindi_response)
                        st.write(f"Report generated in {end_time - start_time:.2f} seconds.")
                     except Exception as e: st.error(f"Error generating report via Ollama: {e}"); st.exception(e)
             else: st.warning("Please fill in all details for the report.")

    elif gen_type == "Pest Suggestion":
        st.subheader("Pest Suggestion Details")
        # Using unique keys for widgets
        crop = st.text_input("Crop Name:", "Cotton", key="pest_crop")
        region = st.text_input("Region:", "Central India", key="pest_region")
        weather_forecast = st.text_input("Weather Forecast:", "Monsoon approaching...", key="pest_weather")
        if st.button("Generate Suggestions", key="gen_pest"):
            if all([crop, region, weather_forecast]):
                prompt = generate_pest_suggestion_prompt(crop, region, weather_forecast)
                with st.spinner("Generating suggestions via Ollama..."):
                     start_time = time.time()
                     try:
                        response = llm.invoke(prompt)
                        end_time = time.time()
                        st.subheader("Generated Suggestions:")
                        st.write(response)
                        if translate_output_gen and translator_hi:
                            with st.spinner("Translating to Hindi..."):
                                hindi_response = translate_text(response, translator_hi)
                                st.subheader("सुझाव (Hindi):")
                                st.write(hindi_response)
                        st.write(f"Suggestions generated in {end_time - start_time:.2f} seconds.")
                     except Exception as e: st.error(f"Error generating suggestions via Ollama: {e}"); st.exception(e)
            else: st.warning("Please fill in all details for pest suggestions.")

# Vision elif block completely removed

# --- Footer/Info Sidebar ---
st.sidebar.markdown("---")
st.sidebar.info(f"LLM Server: Ollama")
st.sidebar.info(f"Model: {OLLAMA_MODEL_NAME}")
st.sidebar.info(f"Embedding: {EMBEDDING_MODEL_NAME}")
st.sidebar.info(f"Vector DB: {VECTOR_DB_PATH}")
# Vision model info removed
