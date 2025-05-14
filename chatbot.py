# app_ollama.py (Chatbot UI, No Vision)

import streamlit as st
import os
import time
from pathlib import Path
import io

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
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: Transformers library not found. Translation feature disabled.")
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None

# --- Configuration ---
OLLAMA_MODEL_NAME = "mistral:7b-instruct-q4_K_M"
TEMPERATURE = 0.7
PDF_DATA_PATH = "data/"
VECTOR_DB_PATH = "vectorstores/db_chroma"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 3
TRANSLATION_MODEL_EN_HI = "Helsinki-NLP/opus-mt-en-hi"

# --- Cached Resource Loading Functions (Minimal Output) ---
# These will now be called by initialize_app() only when needed

@st.cache_resource
def load_llm_cached():
    """Loads the LangChain Ollama client - minimal output for caching"""
    print(f"Attempting to load Ollama model: {OLLAMA_MODEL_NAME}") # Log to console
    try:
        llm = Ollama(model=OLLAMA_MODEL_NAME, temperature=TEMPERATURE)
        # Simple check
        llm.invoke("Respond with OK")
        print("Ollama client loaded successfully.")
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama client: {e}")
        st.error("Ensure the Ollama application is running and the model '{OLLAMA_MODEL_NAME}' is pulled.")
        st.stop() # Stop execution if LLM fails

@st.cache_resource
def load_embeddings_cached():
    """Loads the Sentence Transformer embeddings - minimal output"""
    print(f"Attempting to load embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

# Note: data_ingestion still needs to show progress during the potentially long first run
# We won't cache the *entire* function, but the result (vectorstore) might be implicitly cached if passed around.
# Let's keep its messages for now during the first load. We will call it inside initialize_app.
def data_ingestion(pdf_folder_path, vector_db_path, embeddings):
    """Loads PDFs, splits text, creates embeddings, and stores in Chroma"""
    vectorstore_cls = Chroma
    vector_store_exists = os.path.exists(vector_db_path) and os.listdir(vector_db_path)

    if not vector_store_exists:
        st.info(f"No existing vector store found at {vector_db_path}. Ingesting data...")
        # ... (rest of the ingestion logic with st.info/error as before) ...
        if not os.path.exists(pdf_folder_path) or not os.listdir(pdf_folder_path):
             st.error(f"PDF data folder is empty or missing: {pdf_folder_path}. Please create it and add PDF files.")
             st.stop()
        pdf_files = list(Path(pdf_folder_path).glob('./*.pdf'))
        if not pdf_files: st.error(f"No PDF files found in {pdf_folder_path}. Please add PDF documents."); st.stop()
        st.info(f"Found {len(pdf_files)} PDF files for ingestion.")
        loader = DirectoryLoader(pdf_folder_path, glob='./*.pdf', loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
        try: documents = loader.load()
        except Exception as load_err: st.error(f"Error loading PDFs: {load_err}"); st.exception(load_err); st.stop()
        if not documents: st.error(f"No documents loaded from PDFs."); st.stop()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        if not texts: st.error("Failed to split documents."); st.stop()
        st.info(f"Loaded {len(documents)} docs, split into {len(texts)} chunks.")
        st.info("Creating vector store (can take time)...")
        try:
            vectorstore = vectorstore_cls.from_documents(texts, embeddings, persist_directory=vector_db_path)
            st.success(f"Vector store created at {vector_db_path}")
            return vectorstore
        except Exception as vs_err: st.error(f"Failed to create vector store: {vs_err}"); st.exception(vs_err); st.stop()
    else:
        # Only show loading message if not initialized yet
        if 'initialized' not in st.session_state:
             st.info(f"Loading existing vector store from {vector_db_path}")
        try:
            vectorstore = vectorstore_cls(persist_directory=vector_db_path, embedding_function=embeddings)
            # Only show success if not initialized yet
            if 'initialized' not in st.session_state:
                 st.success("Existing vector store loaded.")
            return vectorstore
        except Exception as vs_load_err: st.error(f"Failed to load vector store: {vs_load_err}"); st.exception(vs_load_err); st.stop()

# LangChain imports needed at the top of the file:
from langchain.chains import LLMChain # Add this import
from langchain.chains.combine_documents.stuff import StuffDocumentsChain # Add this import
# RetrievalQA is already imported

@st.cache_resource # Cache the QA chain
def create_rag_chain_cached(_llm, _vectorstore):
    """Creates the RetrievalQA chain - minimal output"""
    print("Attempting to create RAG chain...")

    # Define the prompt template (ensure it's defined correctly here or accessible)
    rag_prompt_template_str = """SYSTEM: You are an AI assistant providing agricultural advice. Use the following pieces of context ONLY to answer the user's question. If you don't know the answer from the context, just say that you don't know, don't try to make up an answer. Provide concise and actionable advice.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:"""
    RAG_PROMPT = PromptTemplate(
        template=rag_prompt_template_str, input_variables=["context", "question"]
    )

    try:
        # 1. Create the LLMChain explicitly using the prompt
        llm_chain = LLMChain(llm=_llm, prompt=RAG_PROMPT)
        # CHANGE HERE: Use .input_keys
        print(f"LLMChain input keys: {llm_chain.input_keys}") # Debug print

        # 2. Create the StuffDocumentsChain, ensuring 'context' is the document variable
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context" # Explicitly name the document variable
        )
        # CHANGE HERE: Use .input_keys
        print(f"StuffDocumentsChain input keys: {combine_documents_chain.input_keys}") # Debug print


        # 3. Create the RetrievalQA chain using the combine_documents_chain
        qa_chain = RetrievalQA(
            retriever=_vectorstore.as_retriever(search_kwargs={'k': SEARCH_K}),
            combine_documents_chain=combine_documents_chain, # Pass the explicit chain
            return_source_documents=True
        )

        print("RAG chain created successfully.")
        return qa_chain

    except Exception as e:
        st.error(f"Failed to create RAG chain: {e}")
        st.exception(e)
        st.stop()
@st.cache_resource
def load_translator_cached(model_name=TRANSLATION_MODEL_EN_HI):
    """Loads a Hugging Face translation pipeline - minimal output"""
    if not TRANSFORMERS_AVAILABLE or not hf_pipeline: return None
    print(f"Attempting to load translator model: {model_name}")
    try:
        translator = hf_pipeline("translation", model=model_name)
        print("Translator loaded successfully.")
        return translator
    except Exception as e:
        # Use st.warning here as it's optional
        st.warning(f"Could not load translator model '{model_name}': {e}")
        return None

import re # Ensure this is imported at the top of your file

def translate_text(text, translator):
    """Translates text sentence by sentence using the loaded pipeline"""
    if not translator or not text:
        if not translator and text:
            # Use st.warning if in Streamlit context, otherwise print
            if hasattr(st, 'warning'): st.warning("Translator not loaded, cannot translate.")
            else: print("WARNING: Translator not loaded, cannot translate.")
        return text or ""

    try:
        # Split text into sentences, trying to preserve delimiters
        # This regex looks for sentence-ending punctuation followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s for s in sentences if s.strip()] # Remove empty strings

        if not sentences: # If splitting results in nothing (e.g., single word input)
            sentences = [text]

        translated_sentences = []
        # Use st.progress for longer translations if needed in Streamlit context
        # progress_bar = None
        # if hasattr(st, 'progress') and len(sentences) > 3:
        #     progress_bar = st.progress(0, text="Translating sentences...")

        for i, sentence in enumerate(sentences):
            if not sentence.strip(): # Skip any remaining empty/whitespace-only sentences
                continue
            try:
                # Keep max_length reasonable for individual sentences to avoid issues
                result = translator(sentence.strip(), max_length=256) # Max length per sentence
                if result and isinstance(result, list) and 'translation_text' in result[0]:
                    translated_sentences.append(result[0]['translation_text'])
                else:
                    print(f"ERROR: Failed to translate sentence: '{sentence[:50]}...' | Result: {result}")
                    translated_sentences.append(f"[ हिस्सा अनुवादित नहीं किया जा सका: {sentence[:30]}... ]") # Untranslated part marker
            except Exception as e_sent:
                print(f"ERROR: Exception translating sentence '{sentence[:50]}...': {e_sent}")
                translated_sentences.append(f"[ अनुवाद में त्रुटि: {sentence[:30]}... ]") # Error marker

            # if progress_bar:
            #     progress_bar.progress((i + 1) / len(sentences))

        # if progress_bar:
        #     progress_bar.empty() # Clear progress bar

        return " ".join(translated_sentences) # Join translated sentences back

    except Exception as e:
        print(f"ERROR: Overall sentence-based translation failed: {e}")
        # Use st.exception if in Streamlit context
        if hasattr(st, 'exception'): st.exception(e)
        return f"[Sentence Translation Exception] {text}" # Fallback to original

# --- Prompt Engineering Functions (No change needed) ---
def generate_crop_report_prompt(crop, stage, weather, observations):
  """Generates a prompt for a simulated field report"""
  return f"""
  SYSTEM: You are an experienced agricultural field expert. Your task is to write a concise field observation report.
  The report should be professional, factual, and focus on potential risks or important next steps for the farmer based ONLY on the details provided below.
  Do not ask questions. Do not provide disclaimers. Do not write code. Generate only the report content.

  FIELD OBSERVATION DETAILS:
  - Crop Type: {crop}
  - Current Growth Stage: {stage}
  - Recent Weather Conditions: {weather}
  - Farmer's Key Observations: {observations}

  GENERATED FIELD REPORT:
  """

def generate_pest_suggestion_prompt(crop, region, weather_forecast):
  """Generates a prompt for pest suggestions"""
  return f"""
  SYSTEM: You are an agricultural entomologist providing pest risk advisories.
  TASK: Based ONLY on the provided crop, region, and weather forecast, list 2-3 specific pests that are likely to be a concern. For each pest, suggest one simple, actionable preventative measure or an early monitoring technique a farmer can use.
  Be concise. Do not ask clarifying questions. Do not write code. Only provide the pest list and suggestions.

  CONTEXTUAL INFORMATION:
  - Crop: {crop}
  - Geographical Region: {region}
  - Upcoming Weather Forecast: {weather_forecast}

  PEST PREDICTIONS AND PREVENTATIVE SUGGESTIONS:
  """

# --- Initialization Function ---
def initialize_app():
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing models and data... This might take a minute on first run."):
            # Ensure vector store dir exists
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            # Load resources using cached functions
            st.session_state.llm = load_llm_cached()
            st.session_state.embeddings = load_embeddings_cached()
            # Data ingestion might still print messages if it needs to build the store
            st.session_state.vectorstore = data_ingestion(PDF_DATA_PATH, VECTOR_DB_PATH, st.session_state.embeddings)
            st.session_state.qa_chain = create_rag_chain_cached(st.session_state.llm, st.session_state.vectorstore)
            st.session_state.translator_hi = load_translator_cached()
            # Initialize chat history specifically for the chatbot view
            st.session_state.messages = []
            # Mark as initialized
            st.session_state.initialized = True
        st.success("Initialization complete!")
        # Short delay then rerun to clear the init messages
        time.sleep(1.5)
        st.rerun()
    # Ensure components are always available in session state after init
    # This prevents errors if loading failed silently before st.stop()
    required_keys = ['llm', 'embeddings', 'vectorstore', 'qa_chain', 'translator_hi', 'messages']
    if not all(key in st.session_state for key in required_keys):
         # This case should ideally be caught by st.stop() in the load functions
         st.error("Initialization failed unexpectedly. Please check logs and restart.")
         st.stop()


# --- Main App ---

# Title and Markdown
st.title("🌾 Generative AI Assistant for Sustainable Agriculture")
st.markdown("Powered by Local LLMs (**Ollama**) and RAG")

# Run initialization logic
initialize_app()

# --- Task Selection Sidebar ---
st.sidebar.header("Select Task")
task_options = ["Chat about Crops (RAG)", "Generate Report/Suggestion"] # Renamed RAG task
selected_task = st.sidebar.selectbox(
    "Choose what you want to do:",
    task_options,
    key="task_selector"
)

# Add translation toggle to sidebar for chat
if selected_task == "Chat about Crops (RAG)":
     st.session_state.translate_output = st.sidebar.checkbox(
         "Translate Bot answers to Hindi?",
         key="chat_translate",
         disabled=(st.session_state.translator_hi is None)
     )
     if st.session_state.translator_hi is None and TRANSFORMERS_AVAILABLE:
         st.sidebar.warning("Hindi translation unavailable (model load failed).")
     elif not TRANSFORMERS_AVAILABLE:
          st.sidebar.warning("Transformers library not installed. Translation unavailable.")

# --- Main Interaction Area ---

if selected_task == "Chat about Crops (RAG)":
    st.header("Chat about Crop Diseases, Pests, Soil")
    st.info("Ask questions about the topics covered in the provided PDF documents.")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources if they exist for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                 with st.expander("Show Sources"):
                      for source in message["sources"]:
                           st.markdown(f"**Source:** {source.get('name', 'N/A')}, Page: {source.get('page', 'N/A')}")
                           st.markdown(f"> {source.get('content', '')[:500]}...")

    # Accept user input using chat_input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources_info = [] # To store sources for this response
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    # Use the QA chain
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    end_time = time.time()
                    print(f"RAG response generated in {end_time - start_time:.2f} seconds.") # Console log

                    full_response = result.get("result", "Sorry, I encountered an issue getting an answer.")

                    # Process sources
                    source_docs = result.get('source_documents', [])
                    if source_docs:
                        for doc in source_docs:
                            source_name = doc.metadata.get('source', 'N/A')
                            page_num = doc.metadata.get('page', 'N/A')
                            if page_num != 'N/A': page_num += 1 # Adjust page num if 0-indexed
                            sources_info.append({
                                "name": os.path.basename(source_name),
                                "page": page_num,
                                "content": doc.page_content
                            })

                    # Translate if requested
                    if st.session_state.get('translate_output', False) and st.session_state.translator_hi:
                         with st.spinner("Translating..."):
                              translated_response = translate_text(full_response, st.session_state.translator_hi)
                              # Display translated response
                              message_placeholder.markdown(translated_response)
                              # Optionally display original english in expander
                              with st.expander("Original English Answer"):
                                   st.markdown(full_response)
                              # Keep original for history if needed, or translated? Let's store original + sources
                              # full_response remains original english here for history storage

                    else:
                         # Display original response
                         message_placeholder.markdown(full_response)

                    # Display sources below the response text
                    if sources_info:
                        with st.expander("Show Sources"):
                             for source in sources_info:
                                st.markdown(f"**Source:** {source.get('name', 'N/A')}, Page: {source.get('page', 'N/A')}")
                                st.markdown(f"> {source.get('content', '')[:500]}...")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.exception(e)
                    full_response = "Sorry, I encountered an error while processing your request."
                    message_placeholder.markdown(full_response)

            # Add assistant response (original english) and sources to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response, # Store original english
                "sources": sources_info
                })


elif selected_task == "Generate Report/Suggestion":
    st.header("Generate Simulated Reports or Suggestions")
    st.info("Uses prompt engineering to generate text based on your inputs.")
    gen_type = st.radio("Select Generation Type:", ["Field Report", "Pest Suggestion"], horizontal=True, key="gen_type_radio")
    # Moved translation toggle to sidebar for chat, maybe remove here or keep? Let's remove for simplicity.
    # translate_output_gen = st.checkbox("Translate output to Hindi?", key="gen_translate", disabled=(st.session_state.translator_hi is None))

    if gen_type == "Field Report":
        st.subheader("Field Report Details")
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
                        response = st.session_state.llm.invoke(prompt)
                        end_time = time.time()
                        st.subheader("Generated Report:")
                        st.write(response)
                        # Add translation if needed
                        st.write(f"Report generated in {end_time - start_time:.2f} seconds.")
                     except Exception as e: st.error(f"Error generating report via Ollama: {e}"); st.exception(e)
             else: st.warning("Please fill in all details for the report.")

    elif gen_type == "Pest Suggestion":
        st.subheader("Pest Suggestion Details")
        crop = st.text_input("Crop Name:", "Cotton", key="pest_crop")
        region = st.text_input("Region:", "Central India", key="pest_region")
        weather_forecast = st.text_input("Weather Forecast:", "Monsoon approaching...", key="pest_weather")
        if st.button("Generate Suggestions", key="gen_pest"):
            if all([crop, region, weather_forecast]):
                prompt = generate_pest_suggestion_prompt(crop, region, weather_forecast)
                with st.spinner("Generating suggestions via Ollama..."):
                     start_time = time.time()
                     try:
                        response = st.session_state.llm.invoke(prompt)
                        end_time = time.time()
                        st.subheader("Generated Suggestions:")
                        st.write(response)
                        # Add translation if needed
                        st.write(f"Suggestions generated in {end_time - start_time:.2f} seconds.")
                     except Exception as e: st.error(f"Error generating suggestions via Ollama: {e}"); st.exception(e)
            else: st.warning("Please fill in all details for pest suggestions.")


# --- Footer/Info Sidebar ---
st.sidebar.markdown("---")
st.sidebar.info(f"LLM Server: Ollama")
st.sidebar.info(f"Model: {OLLAMA_MODEL_NAME}")
st.sidebar.info(f"Embedding: {EMBEDDING_MODEL_NAME}")
st.sidebar.info(f"Vector DB: {VECTOR_DB_PATH}")
# Translation available info
if TRANSFORMERS_AVAILABLE:
    if st.session_state.get('translator_hi'):
        st.sidebar.info("Translation: Hindi Enabled")
    else:
         st.sidebar.warning("Translation: Hindi Model Load Failed")
else:
    st.sidebar.warning("Translation: Transformers Lib Missing")
