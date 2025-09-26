# Meta Llama Workshop Demo - Customer Support Agent

A real-world customer support agent built with LangGraph, featuring local LLM classification, RAG pipeline, and web search capabilities.

## 🎯 **Features**

- **Local LLM Classification**: Uses Ollama for customer inquiry classification
- **RAG Pipeline**: Vector search through internal knowledge base
- **Web Search**: Real-time web search using Tavily API
- **Smart Routing**: Automatic escalation to specialists based on search quality
- **Interactive Demo**: Command-line interface with interrupts
- **Web Interface**: FastAPI-based chatbot UI
- **Jupyter Notebooks**: Complete workshop materials

## 🛠️ **Prerequisites**

### 1. **Ollama Setup (Required for Local LLM)**

Install Ollama:
```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai/download
```

Pull the required model:
```bash
ollama pull llama3.2:3b
```

Start Ollama service:
```bash
ollama serve
```

### 2. **API Keys**

You'll need API keys for:
- **Upstage API**: [Get key here](https://console.upstage.ai/)
- **Tavily API**: [Get key here](https://tavily.com/)

## 🚀 **Quick Start**

### 1. **Environment Setup**

Follow the setup guide in `환경설정 가이드.md` for detailed instructions.

Quick setup:
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone git@github.com:sunwoong-upstage/meta-llama-workshop-demo.git
cd meta-llama-workshop-demo
uv init
uv venv --python 3.11
uv add langgraph langchain-core openai==1.52.2 faiss-cpu numpy tavily-python python-dotenv "httpx<0.28.0" jupyter ipykernel langchain-upstage langchain langgraph-cli[inmem]
```

### 2. **Environment Variables**

Create `.env` file:
```env
UPSTAGE_API_KEY=your_upstage_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LOCAL_LLM_MODEL=llama3.2:3b
```

### 3. **Run the Demo**

**Interactive Command Line:**
```bash
uv run python graph_build.py
```

**Web Interface:**
```bash
uv run python fastapi_app.py
# Open http://localhost:8000
```

**Jupyter Notebook:**
```bash
uv run jupyter notebook
# Open notebooks/Customer_Support_Agent_Complete.ipynb
```

**LangGraph Dev Mode:**
```bash
uv run langgraph dev
# Open http://localhost:8123
```

## 📊 **GPU Monitoring (M4 Mac)**

Monitor GPU usage during local LLM inference:

### **Option 1: Activity Monitor**
```bash
# Open Activity Monitor → Window → GPU History
open -a "Activity Monitor"
```

### **Option 2: Terminal Commands**
```bash
# GPU usage
sudo powermetrics --samplers gpu_power -n 1 -i 1000

# Alternative: System stats
iostat 1
```

### **Option 3: Install monitoring tools**
```bash
# Install htop with GPU support
brew install htop

# Install GPU monitoring
brew install nvtop  # For detailed GPU stats
```

## 🏗️ **Architecture**

### **Local LLM Classification**
- **Model**: `llama3.2:3b` via Ollama
- **Purpose**: Classify customer inquiries into categories
- **Fallback**: Keyword-based classification if LLM fails
- **Categories**: Technical, Billing, Account, General, Urgent

### **RAG Pipeline**
- **Embeddings**: Upstage embedding models
- **Vector DB**: FAISS (in-memory)
- **Search**: Semantic similarity search
- **Threshold**: 25% similarity for knowledge base results

### **Web Search**
- **Provider**: Tavily API
- **Query Optimization**: LLM-powered query rewriting
- **Threshold**: 50% relevance for web results

### **Smart Routing**
- **Direct Answer**: High-quality search results found
- **Escalation**: Low-quality results → specialist assignment
- **Specialist Assignment**: Based on LLM classification

## 📁 **Project Structure**

```
├── graph_build.py          # Main LangGraph implementation
├── fastapi_app.py          # Web interface
├── 환경설정 가이드.md        # Korean setup guide
├── langgraph.json          # LangGraph configuration
├── data/
│   └── customer_data.csv   # Sample customer data
├── notebooks/
│   └── Customer_Support_Agent_Complete.ipynb
└── static/                 # Web UI assets
    ├── index.html
    ├── style.css
    └── script.js
```

## 🔧 **Configuration**

### **Model Settings**
- **LLM**: `solar-pro2` (Upstage)
- **Local LLM**: `llama3.2:3b` (Ollama)
- **Temperature**: 0 (deterministic)
- **Max Tokens**: Dynamic based on task

### **Search Thresholds**
- **Knowledge Base**: 25% similarity
- **Web Search**: 50% relevance
- **Escalation**: Below thresholds

### **Specialist Categories**
- **Technical**: API, system issues
- **Billing**: Payment, subscriptions
- **Account**: Login, security
- **General**: Information requests
- **Urgent**: Critical issues

## 🧪 **Testing**

Try these example questions:
- "How do I reset my password?"
- "I'm having API connection issues"
- "What are your billing policies?"
- "My system is down - urgent help needed"

## 🚨 **Troubleshooting**

### **Ollama Issues**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve

# Check model
ollama run llama3.2:3b "Hello"
```

### **GPU Not Being Used**
- Ensure Ollama is using GPU: Check Activity Monitor → GPU tab
- M4 Macs use unified memory - GPU usage shows in "GPU" section
- Model should load into GPU memory automatically

### **API Issues**
- Check `.env` file exists and has correct keys
- Verify API key validity
- Check network connectivity

## 📚 **Learn More**

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Upstage API Docs](https://developers.upstage.ai/)
- [Tavily API Docs](https://docs.tavily.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)

## 🤝 **Contributing**

This is a workshop demo project. Feel free to:
- Experiment with different models
- Add new specialist categories
- Improve the web interface
- Enhance the knowledge base

## 📄 **License**

Educational use only - Meta Llama Workshop Demo
