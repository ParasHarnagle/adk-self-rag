# ADK-SELFRAG

Self-RAG implementation using Google ADK (Agent Development Kit).

## Overview

This project implements a Self-RAG (Retrieval Augmented Generation) agent using Google's Agent Development Kit. The agent can process documents, retrieve relevant information, and generate responses based on the retrieved context.

## Project Structure

```
ADK-SELFRAG/
├── .env                   # Environment variables configuration
├── .env.example           # Example environment file template
├── .gitignore             # Git ignore file
├── .python-version        # Python version specification
├── LICENSE                # Project license
├── README.md              # This documentation file
├── main.py                # Main entry point for the application
├── pdf_uploader.py        # Utility to upload PDFs to vector database
├── pyproject.toml         # Project dependencies and configuration
├── uv.lock                # Lock file for uv package manager
├── .venv/                 # Virtual environment directory
└── self-rag/              # Main package directory
    ├── __init__.py        # Package initialization
    ├── __pycache__/       # Python cache directory
    ├── agent.py           # Core agent implementation
    ├── custom_agent.py    # Custom agent extensions
    ├── prompts.py         # Prompt templates for agents
    └── tools/             # Tools directory
        ├── __init__.py    # Tools package initialization
        ├── __pycache__/   # Python cache directory
        └── tools.py       # Tool implementations
```

## Prerequisites

- Python 3.9+
- Pinecone account with API key
- OpenAI API Key
- Google ADK setup with Google AI API Key
- uv package manager installed

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jeyong-shin/adk-selfrag.git
   cd adk-selfrag
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Configure your environment variables:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file with your API keys:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   GOOGLE_API_KEY=your_api_key
   OPENAI_API_KEY=your_api_key
   PINECONE_API_KEY=your_api_key
   PINECONE_INDEX_NAME=your_index_name
   PINECONE_NAMESPACE=your_namespace
   ```

## Usage

### Running the Web Application

To start the web interface for the agent:

```bash
uv run adk web
```

This will launch a local web server where you can interact with your agent.

### Uploading Documents

Before running the agent, you may want to upload documents to your Pinecone index:

```bash
python pdf_uploader.py path/to/your/document.pdf namespace_name
```

This utility will:
1. Extract text from the PDF
2. Split the text into manageable chunks
3. Generate embeddings for each chunk
4. Upload the embeddings to your Pinecone index

## Agent Architecture

The project contains several agent implementations:

- `agent.py`: Contains core LlmAgent implementations and the root agent
- `custom_agent.py`: Contains custom agent extensions for specific use cases
- `tools.py`: Implements tools that agents can use to perform actions

## License

See the [LICENSE](LICENSE) file for details.

## Troubleshooting

If you encounter any issues:

1. Ensure all API keys are correctly set in the `.env` file
2. Verify that your Pinecone index is properly configured
3. Check that you're using a compatible Python version
4. Make sure the uv package manager is installed and working correctly
