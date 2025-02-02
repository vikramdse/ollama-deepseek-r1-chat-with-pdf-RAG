# PDF Q&A

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content. The app uses semantic search and LLM-powered question answering to provide accurate responses based on the document content.

## Features

- PDF document upload and processing
- Semantic chunking for better context understanding
- Vector-based similarity search using FAISS
- LLM-powered question answering using Ollama
- Chat-like interface with conversation history
- Efficient document caching to avoid reprocessing

## Technology Stack

- Python 3.x
- Streamlit
- LangChain
- FAISS for vector storage
- Ollama for embeddings and LLM
- PDFPlumber for PDF processing

## Prerequisites

Before running the application, make sure you have the following installed:

1. Python 3.x
2. [Ollama](https://ollama.com/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-qa.git
cd pdf-qa
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Pull the required Ollama model:

```bash
ollama pull deepseek-r1:1.5b
```

The project currently uses the DeepSeek-R1 1.5B parameter model. If you want to use a larger parameter model for potentially better performance, you can:

- Pull a different model
- Update the `MODEL_NAME` constant in `app.py` to match your chosen model

```python
MODEL_NAME = "deepseek-r1:14b"  # example
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a PDF document using the file uploader

4. Start asking questions about your document in the chat interface

## Project Structure

```
ollama-deepseek-r1-chat-with-pdf-RAG/
├── app.py              # Main application file
├── pdfs/              # Temporary directory for PDF processing
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Key Components

- `DocumentProcessor`: Handles PDF processing, including loading, splitting, and indexing documents
- `QuestionAnswerer`: Manages the question-answering system using the Ollama LLM
- Vector store: Uses FAISS for efficient similarity search
- Semantic chunker: Splits documents into meaningful chunks for better context understanding

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

MIT License

Copyright (c) 2025 Vikram Salunkhe
[vikramsalunkhe.com](https://www.vikramsalunkhe.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

- This project uses the DeepSeek-R1 1.5B model via Ollama
- Built with Streamlit and LangChain
