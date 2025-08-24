# AI Reports Assistant

Your intelligent document analysis companion powered by AI.

## 🚀 Quick Start (No Docker Required)

### Prerequisites
- **Python 3.8+** - [Download here](https://python.org)
- **Node.js 16+** - [Download here](https://nodejs.org)
- **OpenAI API Key** - Set as environment variable `OPENAI_API_KEY`

### Option 1: Automatic Start (Recommended)

#### Windows Users:
```bash
# Double-click the batch file
start_project.bat
```

#### All Platforms:
```bash
# Run the Python start script
python start_project.py
```

This will automatically:
- ✅ Install all dependencies
- 🚀 Start the backend server on port 8000
- 🌐 Start the frontend on port 3000
- 🌍 Open your browser to the application

### Option 2: Manual Start

#### 1. Start Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn webapi.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start Frontend (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

#### 3. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🏗️ Project Structure

```
reports_ai_agent/
├── backend/                 # Python FastAPI backend
│   ├── webapi/            # Main API application
│   ├── agents/            # AI agents (Router, Needle, TableQA, Summary)
│   ├── core/              # Core utilities (PDF reader, text splitter)
│   ├── indexer/           # Document indexing
│   └── retrieval/         # Document retrieval systems
├── frontend/               # Next.js frontend
│   ├── components/        # React components
│   ├── pages/            # Next.js pages
│   └── styles/           # CSS and Tailwind styles
├── data/                  # Sample documents
├── start_project.py       # Python start script
├── start_project.bat      # Windows batch start script
└── README.md             # This file
```

## 🔧 Features

### 🤖 AI Agents
- **Router Agent**: Intelligently routes queries to appropriate agents
- **Needle Agent**: Direct fact lookup and document search
- **TableQA Agent**: Answers questions about tabular data
- **Summary Agent**: Generates document summaries

### 📱 Modern UI
- **Responsive Design**: Works on all devices
- **Theme Toggle**: Switch between light and dark modes
- **Floating Windows**: Expand AI responses for better readability
- **Real-time Chat**: Interactive conversation interface

### 📚 Document Processing
- **PDF Support**: Read and analyze PDF documents
- **Text Splitting**: Intelligent document chunking
- **Vector Search**: FAISS-based similarity search
- **Hybrid Retrieval**: Combines multiple search strategies

## 🌟 Key Technologies

- **Backend**: FastAPI, Python, LangChain, OpenAI
- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion
- **AI**: OpenAI GPT models, embeddings, vector search
- **Database**: FAISS for vector storage

## 🔑 Environment Variables

Set your OpenAI API key:
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Mac/Linux
export OPENAI_API_KEY=your_api_key_here
```

## 📖 Usage

1. **Upload Documents**: Place PDF files in the `data/` directory
2. **Start Application**: Use the start scripts above
3. **Ask Questions**: Chat with your AI assistant about your documents
4. **Get Insights**: Receive intelligent analysis and summaries

## 🛠️ Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn webapi.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill processes on ports 3000 and 8000
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:3000 | xargs kill -9
```

**Python Dependencies**
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

**Node Dependencies**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**OpenAI API Issues**
- Verify your API key is set correctly
- Check your OpenAI account has sufficient credits
- Ensure the API key has the correct permissions

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all prerequisites are installed
4. Verify your OpenAI API key is valid

---

**Happy Document Analysis! 🚀📚✨**
