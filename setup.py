"""
Setup script for Financial Knowledge Base RAG System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="financial-knowledge-base",
    version="0.1.0",
    author="Financial Knowledge Base Team",
    author_email="contact@company.com",
    description="Enterprise AI system for financial document analysis and insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/financial-knowledge-base",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "ui": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
            "pandas>=1.5.0",
        ],
        "full": [
            "openai>=1.0.0",
            "langchain>=0.1.0",
            "faiss-cpu>=1.7.0",
            "chromadb>=0.4.0",
            "pinecone-client>=2.2.0",
            "sentence-transformers>=2.2.0",
            "pypdf>=3.15.0",
            "python-docx>=0.8.11",
            "beautifulsoup4>=4.12.0",
            "requests>=2.31.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "knowledge-base=knowledge_base.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "knowledge_base": [
            "config/*.py",
            "data/*",
            "logs/*",
        ],
    },
    keywords=[
        "financial",
        "ai",
        "rag",
        "nlp",
        "machine-learning",
        "document-analysis",
        "sec-filings",
        "financial-analysis",
    ],
    project_urls={
        "Bug Reports": "https://github.com/company/financial-knowledge-base/issues",
        "Source": "https://github.com/company/financial-knowledge-base",
        "Documentation": "https://github.com/company/financial-knowledge-base/blob/main/README.md",
    },
)