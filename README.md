# Multimodal AI Assistant

A powerful Streamlit-based application that combines multiple AI modalities (text, image, audio, and video processing) into a single chat interface using local Ollama models and other open-source AI tools.

![Multimodal AI Assistant](https://raw.githubusercontent.com/username/multimodal-ai-assistant/main/screenshot.png)

## Features

- **Multiple Input Modalities**:
  - **Text**: Process natural language queries using LLMs
  - **Image**: Analyze images with vision-language models
  - **Audio**: Transcribe speech and perform semantic search
  - **Video**: Extract frames and audio for comprehensive video analysis

- **Integrated AI Experience**:
  - Chat with context awareness across modalities
  - Follow-up questions about previously uploaded media
  - Semantic search through audio transcriptions

- **Local Processing**:
  - All models run locally via Ollama or directly
  - No data sent to external APIs
  - Works offline once dependencies are installed

## Requirements

### System Dependencies

- [Ollama](https://ollama.ai/) - For running LLM and vision models locally
- [FFmpeg](https://ffmpeg.org/) - For audio extraction from videos

### Python Requirements

- streamlit
- requests
- Pillow
- torch
- faiss-cpu (or faiss-gpu)
- sentence-transformers
- openai-whisper
- opencv-python
- numpy

## Installation

1. **Install System Dependencies**:
   ```bash
   # Install Ollama (see https://ollama.ai/ for OS-specific instructions)
   
   # Install FFmpeg
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html or use Chocolatey
   choco install ffmpeg
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/InfinityM10/Multimodel-Chatbot.git
   cd multimodal-ai-assistant
   ```

3. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

4. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download Required Ollama Models**:
   ```bash
   ollama pull llama3
   ollama pull llava
   # Optional additional models
   ollama pull mistral
   ollama pull mixtral
   ollama pull gemma
   ollama pull phi3
   ollama pull bakllava
   ollama pull llava:13b
   ```

## Usage

1. **Start Ollama Service** (if not running):
   ```bash
   ollama serve
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run all.py
   ```

3. **Access the Application**:
   Open your browser and navigate to `http://localhost:8501`

## Features in Detail

### Text Processing
- Uses Ollama models for text completion
- Supports various open-source models (llama3, mistral, mixtral, gemma, phi3)
- Maintains context awareness with previous messages

### Image Analysis
- Processes images using vision-language models (llava, bakllava)
- Recognizes objects, scenes, text in images, and more
- Stores processed images for follow-up questions

### Audio Processing
- Transcribes speech using Whisper
- Creates embeddings for semantic search
- Enables querying previous audio content

### Video Analysis
- Extracts key frames from videos
- Transcribes audio from video content
- Combines visual and audio analysis for comprehensive understanding

### Semantic Search
- Uses FAISS for efficient vector similarity search
- Creates embeddings with SentenceTransformer
- Enables searching through previous audio/text content

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
|  Streamlit UI     |     |  Local AI Models  |     |  Vector Database  |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                       |                         |
        |                       |                         |
        v                       v                         v
+-------------------------------------------------------------------+
|                                                                   |
|                        Core Application                           |
|                                                                   |
+-------------------------------------------------------------------+
        |                 |                  |                 |
        v                 v                  v                 v
+-------------+  +----------------+  +--------------+  +---------------+
|             |  |                |  |              |  |               |
| Text Engine |  | Vision Engine  |  | Audio Engine |  | Video Engine  |
|             |  |                |  |              |  |               |
+-------------+  +----------------+  +--------------+  +---------------+
```

## Development Roadmap

- [ ] Add support for PDF document processing
- [ ] Implement RAG (Retrieval-Augmented Generation) for knowledge base
- [ ] Add memory management for long conversations
- [ ] Support for fine-tuning local models
- [ ] Implement agent-based workflows

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to open-source models
- [Streamlit](https://streamlit.io/) for the interactive web interface
- [Whisper](https://github.com/openai/whisper) for speech transcription
- [SentenceTransformers](https://www.sbert.net/) for text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

---

Made with by Manohar V
