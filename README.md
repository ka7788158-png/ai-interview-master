# AI Interview Master 🧠

## Overview
**AI Interview Master** is an AI-powered interview preparation and resume analysis tool. Built using LangChain and Google's Gemini models, it utilizes a Retrieval-Augmented Generation (RAG) pipeline to analyze a candidate's resume (PDF) against a target Job Description. 

It generates a comprehensive, personalized preparation report including match scores, skill gap analysis, tailored technical and behavioral questions, and a day-by-day study roadmap.

## Features
- **PDF Resume Parsing:** Extracts and processes resume content using `PyPDFLoader`.
- **RAG Architecture:** Embeds resume chunks into a Chroma vector store for context-aware retrieval.
- **Gemini 2.5 Flash Integration:** Leverages advanced LLMs to generate highly tailored interview strategies.
- **Interactive UI:** A clean, easy-to-use Streamlit web interface.

## Project Structure
- `app.py`: The Streamlit frontend handling UI layout, file uploads, and displaying the generated report.
- `ai_agent.py`: The core RAG pipeline and backend logic using LangChain and Google Generative AI.
- `requirements.txt`: Python dependencies required to run the application.
- `.env` (Not included): Environment variables file containing the Google API keys.

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd ai-interview-master
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `chromadb` installed as well for the vector store functionality).*

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY="your_actual_google_api_key_here"
   ```

4. **Run the Application:**
   Launch the Streamlit interface locally:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- **Frontend:** Streamlit
- **Framework:** LangChain
- **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Vector Database:** ChromaDB
- **Document Processing:** PyPDF

## Author
**Kavya Agrawal** *Data Scientist & Machine Learning Engineer*
