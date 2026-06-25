import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_interview_report(resume_file, job_desc:str, self_desc:str) -> str:
    """
    Takes an uploaded resume PDF, Job Description, and Self-Description.
    Uses RAG to extract relevant resume details and generates a structured interview prep report.
    """
    # We need to save it to a temporary file so PyPDFLoader can read it.

    with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as temp_file:
        temp_file.write(resume_file.getvalue())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # splitting resume into small chunks 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = splitter.split_documents(docs)

        # creating vector database
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

        vectorstore = Chroma.from_documents(chunks, embeddings)
        # Create a Retriever to fetch the most relevant parts of the resume 
        # based on the job description
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 4. SET UP THE LLM & PROMPT (LangChain Core)
        llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.2 )

        prompt = ChatPromptTemplate([
            ("system", """You are an expert AI Career Coach and Technical Interviewer. 
            Analyze the candidate's Resume Context and Self-Description against the provided Job Description.
            
            Provide a structured markdown report containing:
            1. A Match Score (e.g., 85%).
            2. Key Skill Gaps (What the JD asks for but the resume lacks).
            3. 3-5 Tailored Technical Interview Questions with brief tips on how to answer them.
            4. 2 Behavioral Questions.
            5. A short day-by-day preparation roadmap."""),
            ("human", """
            Target Job Description:
            {job_desc}
            
            Candidate Self-Description:
            {self_desc}
            
            Relevant Resume Context (Retrieved via RAG):
            {context}
            """)
        ])

        # Built and Invoke the Runnable Chain
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # The RAG Chain: 
        # 1. Passes job_desc & self_desc right through.
        # 2. Uses the job_desc to search the VectorDB (Retriever) and formats the results into 'context'.
        # 3. Feeds everything into the Prompt.
        # 4. Sends the Prompt to the LLM.
        # 5. Parses the LLM output as a simple string.

        rag_chain = (
            {"context": retriever | format_docs, 
             "job_desc": RunnablePassthrough(), 
             "self_desc": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the chain using the Job Description as the primary query for the retriever
        result = rag_chain.invoke(job_desc)

        return result
    
    finally:
        # Clean up the temporary PDF file to save space
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)