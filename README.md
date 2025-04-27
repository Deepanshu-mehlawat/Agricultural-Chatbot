# Agri AI Assistant 🌾

A Generative AI-powered assistant for farmers, designed to provide sustainable agriculture solutions using local Large Language Models (LLMs) via Ollama and Retrieval-Augmented Generation (RAG).

## Features

*   **RAG Q&A:** Ask questions in natural language (English) about agricultural topics (crop diseases, pests, soil health, etc.). The assistant answers based on information retrieved from provided PDF documents.
*   **Chat Interface:** Interact with the RAG system through a user-friendly chatbot interface.
*   **Report/Suggestion Generation:** Generate simulated field reports or pest suggestions based on user inputs using prompt engineering.
*   **GPU Acceleration:** Leverages local GPU resources via the Ollama server for faster LLM inference.
*   **Optional Hindi Translation:** Can translate the assistant's responses into Hindi using a local translation model.

## Technology Stack

*   **LLM Server:** [Ollama](https://ollama.com/) (Running Mistral 7B Instruct Q4_K_M by default)
*   **Orchestration:** [LangChain](https://python.langchain.com/)
*   **Embeddings:** [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
*   **Vector Store:** [ChromaDB](https://www.trychroma.com/)
*   **UI Framework:** [Streamlit](https://streamlit.io/)
*   **Translation:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (Helsinki-NLP models)
*   **Environment:** Python 3.10, Conda

## Setup Instructions

Follow these steps to set up and run the Agri AI Assistant locally.

### Prerequisites

1.  **Operating System:** Tested primarily on Windows. Should be adaptable for macOS/Linux.
2.  **Anaconda or Miniconda:** Required for managing the Python environment and dependencies. Download from [anaconda.com](https://www.anaconda.com/download).
3.  **Git:** Required for cloning the repository. ([git-scm.com](https://git-scm.com/downloads))
4.  **NVIDIA GPU (Recommended for Performance):** While the app can run on CPU, GPU acceleration via Ollama significantly improves performance. Ensure you have:
    *   An NVIDIA GPU compatible with Ollama.
    *   Up-to-date NVIDIA drivers installed.
    *   *(Note: Installing the full CUDA Toolkit globally is **not** strictly required for running with Ollama, as Ollama handles its own CUDA dependencies, but it was part of the development setup for this project during earlier testing phases).*
5.  **Ollama:** This application relies on a running Ollama server.
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Ensure the Ollama application is running in the background (check system tray icon).

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

2.  **Create Conda Environment:**
    ```bash
    conda create -n agri_assistant python=3.10 -y
    conda activate agri_assistant
    ```
    *(Replace `agri_assistant` with your preferred environment name if desired)*

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the Ollama Model:**
    *   Make sure the Ollama application is running.
    *   Open a *new* terminal (no need to activate the conda env for this step) and run:
        ```bash
        ollama pull mistral:7b-instruct-q4_K_M
        ```
    *   *(This downloads the specific Mistral 7B model used by default. You can modify `OLLAMA_MODEL_NAME` in the script if you use a different model).*

5.  **Prepare Data:**
    *   Create a folder named `data` in the root of the project directory.
    *   Place relevant agricultural PDF documents (e.g., fact sheets, guides) inside the `data` folder. The RAG system will use these as its knowledge base.

6.  **Vector Store:**
    *   A folder named `vectorstores/db_chroma` will be created automatically the first time you run the app to store the document embeddings.

### Running the Application

1.  **Ensure Ollama is Running:** Check that the Ollama application/server is active.
2.  **Activate Conda Environment:**
    ```bash
    conda activate agri_assistant
    ```
3.  **Run Streamlit App:**
    ```bash
    streamlit run app_ollama.py
    ```
4.  The application should open automatically in your default web browser.

## Usage Guide

1.  **Task Selection:** Use the sidebar on the left to choose between:
    *   **Chat about Crops (RAG):** Interact with the RAG system in a chat format. Ask questions related to the content of the PDFs in the `data` folder.
    *   **Generate Report/Suggestion:** Generate simulated text based on prompt templates.
2.  **Chat Interface:**
    *   Type your question into the input box at the bottom and press Enter.
    *   The assistant's response (based on retrieved documents) will appear.
    *   You can expand the "Show Sources" section under the bot's message to see snippets from the source PDFs used to generate the answer.
    *   Use the "Translate Bot answers to Hindi?" checkbox in the sidebar (if the translation model loaded successfully) to toggle translation.
3.  **Generate Report/Suggestion:**
    *   Select the type of generation ("Field Report" or "Pest Suggestion").
    *   Fill in the required details in the input fields.
    *   Click the "Generate..." button.
    *   The generated text will appear below.

## Configuration

You can modify some default settings within the `app_ollama.py` script:

*   `OLLAMA_MODEL_NAME`: Change the target model served by Ollama.
*   `TEMPERATURE`: Adjust the LLM's creativity/randomness.
*   `PDF_DATA_PATH`, `VECTOR_DB_PATH`: Change data and vector store locations.
*   `EMBEDDING_MODEL_NAME`: Use a different Sentence Transformer model.
*   `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SEARCH_K`: Modify RAG parameters.
*   `TRANSLATION_MODEL_EN_HI`: Change the Hindi translation model.

## Challenges Faced (Development Notes)

*   Initial attempts to use direct compilation libraries (`llama-cpp-python`, `ctransformers`) with CUDA GPU acceleration on Windows encountered significant build environment challenges (CMake errors, CUDA toolset detection issues, compiler mismatches).
*   Resolving these required careful setup of Visual Studio Build Tools, global CUDA Toolkit installation, specific environment variables (`CMAKE_ARGS`, `CUDAToolkit_ROOT`), and using the correct x64 Developer Command Prompt.
*   Switching to **Ollama** as the LLM backend dramatically simplified the setup process, reliably providing GPU acceleration by abstracting away the complex build requirements.

## Future Work

*   Expand the knowledge base with more diverse agricultural documents.
*   Implement more sophisticated RAG techniques (e.g., re-ranking, query transformation).
*   Support multilingual input queries.
*   Add evaluation metrics for response quality and relevance.
*   Improve error handling and user feedback in the UI.
*   Re-integrate vision capabilities for image-based disease diagnosis (requires finding/training a suitable model and reliable setup).
*   Refine prompt templates for better report/suggestion generation.

*(Optional: Add License Information Here)*
