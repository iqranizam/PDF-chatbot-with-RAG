import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import gradio as gr

# Load environment variables
load_dotenv()

class PDFChatbot:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.chat_history = []
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Set up text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Set up prompt template
        self.prompt_template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Keep the answer concise.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
    def load_pdf(self, file_path):
        """Load and process PDF file"""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Split documents
            docs = self.text_splitter.split_documents(pages)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Set up RAG chain
            prompt = ChatPromptTemplate.from_template(self.prompt_template)
            
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            return True, "PDF loaded successfully!"
        except Exception as e:
            return False, f"Error loading PDF: {str(e)}"
    
    def ask_question(self, question):
        """Answer question based on loaded PDF"""
        if not self.chain:
            return "Please load a PDF first."
        
        try:
            answer = self.chain.invoke(question)
            self.chat_history.append((question, answer))
            return answer
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_history = []
        return "Chat history cleared."

# Gradio Interface
def setup_gradio_interface():
    chatbot = PDFChatbot()
    
    def process_pdf(file_obj):
        success, message = chatbot.load_pdf(file_obj.name)
        if success:
            return message, None  # Clear chat history on new PDF load
        else:
            return message, None
    
    def respond(message, chat_history):
        answer = chatbot.ask_question(message)
        chat_history.append((message, answer))
        return "", chat_history
    
    with gr.Blocks(title="PDF Chatbot") as demo:
        gr.Markdown("# PDF Chatbot with RAG")
        gr.Markdown("Upload a PDF and ask questions about its content.")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload PDF", type="file")
                upload_status = gr.Textbox(label="Status", interactive=False)
                upload_button = gr.Button("Process PDF")
                clear_button = gr.Button("Clear Chat")
            
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(label="Conversation")
                user_input = gr.Textbox(label="Your Question", placeholder="Type your question here...")
                submit_button = gr.Button("Submit")
        
        # Event handlers
        upload_button.click(
            process_pdf,
            inputs=file_input,
            outputs=[upload_status, chatbot_interface]
        )
        
        submit_button.click(
            respond,
            inputs=[user_input, chatbot_interface],
            outputs=[user_input, chatbot_interface]
        )
        
        user_input.submit(
            respond,
            inputs=[user_input, chatbot_interface],
            outputs=[user_input, chatbot_interface]
        )
        
        clear_button.click(
            chatbot.clear_chat,
            outputs=upload_status
        ).then(
            lambda: None,
            None,
            chatbot_interface,
            queue=False
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch Gradio interface
    interface = setup_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
