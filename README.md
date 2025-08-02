# 🤖 AI Document Analyzer

A powerful AI-powered document analysis system that allows users to upload PDF documents and ask intelligent questions about their content using advanced language models.

## ✨ Features

- 📄 **PDF Document Upload** - Drag & drop or browse to upload PDF files
- 🤖 **AI-Powered Analysis** - Uses Google Gemini for intelligent document analysis
- 💬 **Multi-turn Conversations** - Ask follow-up questions with context memory
- 🔍 **Clause Visualization** - Highlights key decision-relevant text with transparency
- 📚 **Conversation History** - View and navigate through previous conversations
- 🎨 **Modern UI** - Responsive design with beautiful animations
- 🚀 **Real-time Processing** - Fast document indexing and query responses

## 🛠️ Tech Stack

- **Backend:** FastAPI, LangChain, ChromaDB
- **AI:** Google Gemini 1.5 Flash, HuggingFace Embeddings
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Document Processing:** PyMuPDF
- **Vector Storage:** ChromaDB with HuggingFace embeddings

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google API Key for Gemini

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-document-analyzer.git
cd ai-document-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

4. **Run the application:**
```bash
python main.py
```

5. **Open your browser:** Navigate to `http://localhost:8000`


### Render

1. Connect your GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn main:app --host=0.0.0.0 --port=$PORT`
4. Add environment variable: `GOOGLE_API_KEY`

## 📖 Usage

1. **Upload Documents**: Drag and drop PDF files or click to browse
2. **Ask Questions**: Type natural language questions about your documents
3. **View Results**: Get AI-powered analysis with supporting clauses
4. **Follow-up**: Ask contextual follow-up questions
5. **History**: View detailed conversation history with navigation

### Example Queries

- "What are the key terms and conditions?"
- "What is covered under this policy?"
- "What about coverage for a 45-year-old male?"
- "Are there any exclusions I should know about?"

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY` - Required. Your Google API key for Gemini
- `HOST` - Optional. Server host (default: 127.0.0.1)
- `PORT` - Optional. Server port (default: 8000)

### Features Toggle

The application automatically detects uploaded documents and enables features:
- Document upload validation
- Duplicate file prevention
- Multi-turn conversation support
- Clause visualization with highlighting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini for AI capabilities
- LangChain for document processing pipeline
- HuggingFace for embeddings
- FastAPI for the robust backend framework

---

**Made with ❤️ Amruthesh Hiremath**
