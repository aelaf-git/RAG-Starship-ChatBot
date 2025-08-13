Advanced RAG Chatbot: From Notebook to Deployed Web App
This repository contains the complete source code and documentation for a sophisticated Retrieval-Augmented Generation (RAG) system. The project demonstrates the end-to-end process of building a chatbot that can answer questions and summarize information from user-provided PDF documents.
The project is presented in two distinct formats:
RAG-Starship_ChatBot.ipynb: A Jupyter Notebook detailing the step-by-step development, experimentation, and validation of the core RAG pipeline.
app.py: A user-friendly, interactive web application built with Streamlit, designed for easy use and deployment.
![alt text](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
<!-- Replace with your live deployment link! -->
🌟 Key Features
Interactive Q&A: Chat with your documents and get answers grounded in the provided text.
Dynamic Document Summarization: Generate concise summaries of entire PDF documents using an advanced map-reduce technique.
Multi-Document Support: Upload any PDF via the web interface to create a custom knowledge base on the fly.
High-Precision Retrieval: Utilizes a hybrid retrieval approach with a fast FAISS vector search followed by a more accurate Cross-Encoder reranker to ensure the most relevant context is used.
Fast & Powerful LLM: Powered by the high-performance llama3-8b-8192 model running on the Groq API for near-instantaneous generation.
Dual Implementations: Includes both a detailed development notebook for learning and a polished Streamlit app for demonstration.
⚙️ Tech Stack & Architecture
This project is built using the LangChain framework in Python, orchestrating several key components to create the RAG and Summarization pipelines.
Q&A Flow Architecture
code
Code
User Query -> Streamlit UI -> FAISS Retriever -> CrossEncoder Reranker -> Prompt Template -> Groq (Llama 3) -> Answer
Summarization Flow Architecture
code
Code
PDF Document -> LangChain Loader -> Text Chunks -> Map-Reduce Chain -> Groq (Llama 3) -> Final Summary
Framework: LangChain
Document Loader: PyPDFLoader
Text Splitter: RecursiveCharacterTextSplitter
Embedding Model: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
Vector Store: FAISS (CPU)
Reranker: HuggingFaceCrossEncoder (ms-marco-MiniLM-L-6-v2)
LLM: Llama 3 (8B) via langchain-groq
UI/Deployment: Streamlit & Streamlit Community Cloud
🛠️ Setup and Installation
Follow these steps to set up and run the project locally.
1. Clone the Repository
code
Bash
git clone https://github.com/your-username/RAG-Starship_ChatBot.git
cd RAG-Starship_ChatBot```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
# .venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate
3. Install Dependencies
Install all the required Python packages from the requirements.txt file.
code
Bash
pip install -r requirements.txt
4. Set Up API Keys
The application requires an API key from Groq to use the Llama 3 model.
Get your key: Sign up for free at the Groq Console.
You need to configure the key differently depending on which version of the project you are running:
For the Jupyter Notebook (.ipynb):
Create a file named .env in the root of the project folder and add your key:
code
Code
GROQ_API_KEY="your_actual_api_key_here"
For the Streamlit App (app.py):
Create a folder named .streamlit and inside it, create a file named secrets.toml. Add your key there:
code
Toml
GROQ_API_KEY="your_actual_api_key_here"
🚀 Usage Instructions
Version 1: Jupyter Notebook (RAG-Starship_ChatBot.ipynb)
This notebook is perfect for understanding the core logic, experimenting with different components, and seeing the output of each step.
Make sure you have set up your .env file as described above.
Place the Starship.pdf file in the root directory.
Open the project in a code editor like Visual Studio Code.
Launch RAG-Starship_ChatBot.ipynb.
Select the Python interpreter corresponding to your .venv virtual environment.
Run the cells sequentially from top to bottom to see the entire RAG pipeline in action.
Version 2: Streamlit Web App (app.py)
This is the user-friendly, interactive version of the project.
Ensure you have set up your .streamlit/secrets.toml file.
Run the application from your terminal:
code
Bash
streamlit run app.py
Your web browser will open with the application running locally.
How to use the app:
a. Use the sidebar to upload any PDF document you wish to analyze.
b. Once the file is processed, choose a mode: "Q&A (Chat)" or "Summarization".
c. In Q&A mode, type your questions into the chat input at the bottom and press Enter.
d. In Summarization mode, click the "Generate Summary" button to get a full summary of the document.
🔬 How It Works: A Deeper Dive
The application's intelligence comes from its carefully constructed RAG and summarization pipelines.
1. Ingestion and Chunking
When a PDF is provided, PyPDFLoader extracts the text. This text is then broken down by the RecursiveCharacterTextSplitter into small, overlapping chunks. Chunking is essential because LLMs have a limited context window, and overlapping preserves semantic meaning that might be lost at chunk boundaries.
2. Indexing and Storage (for Q&A)
Each text chunk is converted into a numerical vector (an "embedding") using the all-MiniLM-L6-v2 model from HuggingFace. These vectors capture the semantic meaning of the text. All vectors are stored in a FAISS index, a highly efficient library for similarity searching. This entire index is our searchable knowledge base.
3. Retrieval and Reranking (for Q&A)
When a user asks a question:
Step 1 (Retrieval): The user's question is converted into a vector. FAISS then performs a rapid search to find the chunks with the most similar vectors. This gives us a list of potentially relevant documents.
Step 2 (Reranking): To improve precision, these initial documents are passed to a CrossEncoder model. Unlike the first model, a Cross-Encoder examines the question and a document simultaneously, providing a much more accurate relevance score. It then re-sorts the documents and returns the top 2, ensuring only the most relevant context is passed to the LLM.
4. Generation (for Q&A)
The final, reranked context and the original user question are inserted into a carefully crafted PromptTemplate. This prompt instructs the LLM (llama3-8b-8192) to answer the question only based on the provided context, which minimizes hallucinations and ensures grounded answers.
5. Summarization
For summarization, a different strategy is used: Map-Reduce.
Map Step: The LLM is asked to create a summary for each individual text chunk.
Reduce Step: All the individual chunk summaries are then combined and given to the LLM at once, with instructions to create a final, unified summary from them. This allows the model to summarize documents of any length without exceeding its context window.
⚠️ Limitations and Known Issues
API Key Required: The application will not function without a valid Groq API key.
Initial Processing Time: The very first time the app is run (or when a new document is uploaded), creating the embeddings and vector store can take a moment, especially for large documents. Subsequent interactions are fast thanks to caching.
Resource Intensive: The embedding and reranking models run on the CPU and can consume significant RAM during processing.
Answer Quality: The quality of the chatbot's answers is entirely dependent on the quality and content of the uploaded PDF document.