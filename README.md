# AI Agent — Go + Genkit / Google ADK

AI agent implementations in Go, demonstrating Text Generation, Tool Calling, Structured Output, Flows, and RAG.

## Implementations

### genkit-go/

Full-featured AI agent built with [Firebase Genkit](https://firebase.google.com/docs/genkit).

**Features:**
- Text Generation (Gemini / Ollama)
- Tool Calling (Calculator with 4 arithmetic operations)
- Structured Output (Sentiment Analysis, Joke Generation)
- REST API Server (5 endpoints)
- RAG with in-memory vector store (cosine similarity search)
- Dual model support: Google AI Gemini + Ollama (llama3.2)

**API Endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Free-form chat with tool access |
| `/api/rag` | POST | RAG-based Q&A |
| `/api/tellJoke` | POST | Structured joke generation |
| `/api/analyzeSentiment` | POST | Sentiment analysis |
| `/health` | GET | Health check |

### adk-go/

ReAct-pattern agent built with [Google ADK](https://google.github.io/adk-go/).

**Features:**
- ReAct loop (Reasoning + Acting)
- AI knowledge search tool (connects to mcp/ai_knowledge)
- Web search tool (connects to mcp/external_api)
- Gemini 2.0 Flash for planning and generation

## Quick Start

```bash
# genkit-go
cd genkit-go
cp .env.example .env  # Set GEMINI_API_KEY
make demo              # Run demo mode
make server            # Run API server

# adk-go
cd adk-go
cp .env.example .env   # Set GEMINI_API_KEY
make run               # Run CLI mode
```

## Tech Stack

- **Language:** Go
- **Frameworks:** Firebase Genkit, Google ADK
- **Models:** Gemini 2.0 Flash, Ollama (llama3.2)
- **Embedding:** text-embedding-004
- **Vector Search:** In-memory cosine similarity
