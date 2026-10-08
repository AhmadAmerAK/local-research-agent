# Local Research Agent
A simple locally hosted Research Agent that performs a research query and returns back a summarised cited result.

## Installation

### Prerequisites

- Python 3.11+
- 6GB+ VRAM GPU, please refer to this [link](https://aiagentskit.com/blog/best-gpu-for-ai/) for better understanding of GPUs suitable for local hosting
- OpenAlex and Tavily API keys
- Ollama installed locally
- Create a local `.env` file from the example:

```bash
cp .env.example .env
```

### Search API Keys

- Create accounts on both [OpenAlex](https://openalex.org) and [Tavily](https://www.tavily.com/) and choose your desired plans. There are free tier plans for both search APIs.
- Copy your API keys from: [OpenAlex](https://openalex.org/settings/api-key), [Tavily](https://app.tavily.com/home)
- Put the real keys in `.env`:

```bash
OPENALEX_API_KEY= YOUR_OPENALEX_KEY
TAVILY_API_KEY= YOUR_TAVILY_KEY
```

### Ollama Setup

Install Ollama if needed:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama in a separate terminal:

```bash
ollama serve
```

Pull the configured model:

```bash
ollama pull qwen3:4b-instruct
```

The value of `OLLAMA_MODEL` in `.env` must match a model available from `ollama list`.

### Python Environment

```bash
# Create environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```