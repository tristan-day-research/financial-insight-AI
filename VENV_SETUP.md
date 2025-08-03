# Virtual Environment Setup for Financial Insight AI

## ✅ Virtual Environment is Ready!

Your virtual environment has been successfully created and configured with all dependencies installed.

## Quick Start

### Activate the Virtual Environment

**Option 1: Use the convenience script**
```bash
./activate_venv.sh
```

**Option 2: Manual activation**
```bash
source venv/bin/activate
```

### Deactivate the Virtual Environment
```bash
deactivate
```

## What's Installed

Your virtual environment includes:

### Core AI/ML Libraries
- `langchain==0.1.0` - AI framework for building applications
- `openai==1.12.0` - OpenAI API client
- `faiss-cpu==1.7.4` - Vector similarity search
- `chromadb==0.4.22` - Vector database
- `sentence-transformers==2.2.2` - Text embeddings

### Data Processing
- `pandas==2.1.4` - Data manipulation
- `numpy==1.24.3` - Numerical computing
- `sqlalchemy==2.0.25` - Database ORM

### Financial APIs
- `sec-edgar-downloader==5.0.1` - SEC filings downloader
- `yfinance==0.2.12` - Yahoo Finance data

### Web Interface
- `streamlit==1.29.0` - Web app framework
- `plotly==5.17.0` - Interactive charts

### Development Tools
- `pytest==7.4.4` - Testing framework
- `black==23.12.1` - Code formatter
- `flake8==7.0.0` - Linter
- `mypy==1.8.0` - Type checker

## Best Practices

### 1. Always Activate Before Working
```bash
source venv/bin/activate
```

### 2. Install New Dependencies
```bash
pip install <package_name>
pip freeze > requirements.txt  # Update requirements
```

### 3. Run Your Application
```bash
python quickstart.py
# or
streamlit run llm_chat/src/ui/streamlit_app.py
```

### 4. Development Workflow
```bash
# Activate environment
source venv/bin/activate

# Run tests
pytest

# Format code
black .

# Type checking
mypy .

# Linting
flake8
```

## Project Structure

```
Financial Insight AI/
├── venv/                    # Virtual environment (ignored by git)
├── knowledge_base/          # Core RAG system
├── llm_chat/               # Web interface
├── requirements.txt         # Dependencies
├── setup.py                # Package configuration
├── activate_venv.sh        # Convenience script
└── VENV_SETUP.md          # This file
```

## Troubleshooting

### If you get "command not found: python"
Make sure you've activated the virtual environment:
```bash
source venv/bin/activate
```

### If you get import errors
Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### If you need to recreate the environment
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file for API keys:
```bash
cp env.example .env
# Edit .env with your API keys
```

## Next Steps

1. Set up your API keys in `.env`
2. Run `python quickstart.py` to test the system
3. Explore the `knowledge_base/` and `llm_chat/` directories
4. Check out the Streamlit interface

Happy coding! 🚀 