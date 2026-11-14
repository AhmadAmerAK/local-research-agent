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

## Run the application

```bash
agent-env.venv/bin/python src/gui/gui_main.py
```

For the command-line interface:

```bash
agent-env.venv/bin/python src/main.py "Research the rise in ADHD diagnosis"
```

Use your activated environment's `python` instead if you followed the `.venv` installation below.

Each GUI window owns one background worker and one Strands research agent. Follow-up questions
reuse its conversation and evidence. Starting a new application creates an isolated session;
previous saved research does not automatically enter its context.

The agent selects its research route each turn. Python exposes only that route's search provider
and enforces a per-question search budget of `MAX_SEARCH_QUERIES * MAX_RESEARCH_ROUNDS`.
There is no Python search-round loop. Detailed `AGENTS.md` rules and existing skills are loaded
on demand; essential scope, citation and provider rules always apply.

Evidence and original provider responses are saved under `src/agent/memory/runs/<session-id>/`.
Individual answers are saved in its `turns/` subdirectory. Native context reduction can compact
conversation history; the agent can still retrieve original evidence in pages using stable source IDs.
Closing the window requests cancellation and waits for the worker to close its session safely.
In-flight HTTP requests and retry waits cooperate with cancellation; request deadlines also bound stalls.

## Tests

```bash
QT_QPA_PLATFORM=offscreen agent-env.venv/bin/python -m pytest tests -q
```

The regression suite uses the installed Strands loop with scripted model responses and mocked
providers; it does not require live Ollama inference or consume research API credits.

## Reliability checks

Requests have explicit timeouts and cooperative cancellation, including stalled Ollama
streams. Reports also check sentence-sized claims against original retrieved evidence,
label conflicts and uncertainty in the same response, and save the checks as `grounding.json`.
These conservative checks do not establish semantic correctness of paraphrased claims.

See [Phase 2.5 verification and limitations](docs/phase-2.5-reliability.md) for measured
shutdown behaviour, the grounding evaluation, and reproduction commands. Session restoration
after restarting the application remains out of scope.
