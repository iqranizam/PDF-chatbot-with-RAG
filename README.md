# PDF Chatbot with RAG (LangChain + OpenAI)

A chatbot that answers questions from uploaded PDFs using **Retrieval-Augmented Generation (RAG)**.

## Features
- Upload and query PDFs (research papers, reports, etc.)
- Accurate AI-powered answers with source context
- Gradio web interface
- FAISS/Chroma vector database

## Tech Stack
- **LLM**: OpenAI GPT-3.5/4 or Llama 2
- **Framework**: LangChain
- **Vector DB**: FAISS/Chroma
- **UI**: Gradio

## Installation
1. Clone repo:
```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add OpenAI API key to `.env`:
```env
OPENAI_API_KEY="your-key-here"
```

## Usage
```bash
python app.py
```
Open http://localhost:7860 to use the chatbot.

## How It Works
1. PDF → Text chunks → Vector embeddings
2. Semantic search finds relevant text
3. LLM generates answers from retrieved text

## Project Structure
```
app.py          # Main application
requirements.txt
.env            # API keys (gitignored)
```

## License
MIT
```

**Pro Tips for GitHub:**
1. Replace `yourusername` with your GitHub handle
2. Upload a demo GIF/screenshot (rename placeholder URL)
3. For best formatting:
   - Use **`.md` extension**
   - Keep line breaks between sections
   - Use ` ``` ` for code blocks

This version is:
✅ **Concise** (fits GitHub preview)  
✅ **Formatted** (Markdown-compatible)  
✅ **Actionable** (clear install/usage steps)
