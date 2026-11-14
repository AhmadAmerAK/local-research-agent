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


## Configuration

Set options in `.env` (see [.env.example](.env.example)); existing process environment values
take precedence. Restart the application after changing settings. Blank numeric values use
the defaults below. Every numeric setting must be a positive integer: zero, negative values,
floating-point values and infinity do not disable limits. Direct `Settings` construction uses
the same validity checks. Operational budgets have no arbitrary application upper ceilings.

### Research budgets

| Setting | Default | Meaning |
|---|---:|---|
| `MAX_SEARCH_QUERIES` | 2 | Multiplied by `MAX_RESEARCH_ROUNDS` for the per-question search-call budget. |
| `MAX_RESEARCH_ROUNDS` | 1 | Legacy multiplier, not a Python search loop. A value such as 5 is accepted. |
| `MAX_SOURCES` | 5 | Maximum distinct sources cited in a final answer; does not discard other retrieved evidence. |
| `MIN_REQUIRED_SOURCES` | 3 | Minimum distinct valid sources actually cited in a new report. Must not exceed `MAX_SOURCES`. |
| `AGENT_MAX_TURNS` | search budget + 12 | Maximum model calls per question, including correction retries and an output-limit continuation. |

Three substantive, individually cited perspectives remain an assignment requirement independent
of source count. For example, `MIN_REQUIRED_SOURCES=1` permits three perspectives supported by
one source; `MIN_REQUIRED_SOURCES=4` requires at least four cited sources. The agent must still
acknowledge insufficient evidence instead of inventing sources. Conversational follow-ups retain
their existing concise-answer policy: at least one valid supporting citation, without repeating
the full report or its source minimum. The guide and report skill use these same rules.

An explicit `AGENT_MAX_TURNS` overrides the derived default. It can stop research before the
search-call budget is consumed: routing, guide loading, evidence reads and final answers also
use model calls. The extra 12 in the default preserves the previous allowance for that work;
it is a configurable heuristic, not a model capability limit. Each question renews the budgets.
HTTP retry attempts remain separately bounded and do not count as additional search tool calls.

### Context and output

| Setting | Default | Meaning |
|---|---:|---|
| `CONTEXT_WINDOW` | 8192 | Passed to Strands `context_window_limit` and Ollama `num_ctx`. |
| `OUTPUT_TOKENS` | 3072 | Ollama `num_predict` for every conversational model call, including continuation. |

`OUTPUT_TOKENS` must be smaller than `CONTEXT_WINDOW` so input has room. This check does not
guarantee every prompt fits: native Strands context reduction still manages conversation history.
The application imposes no model-independent context ceiling; the selected model/server and
hardware must support the requested size. Accepting a value is not proof of full GPU residency.
Defaults, temperature (0.15), keep-alive (5 minutes), Ollama service settings and autonomous
architecture are unchanged. An output-limit continuation is attempted at most once, and only
when the question's model-call budget has room.

### Search and evidence payloads

| Setting | Default | Meaning |
|---|---:|---|
| `SEARCH_RESULTS_PER_QUERY` | `MAX_SOURCES` | Requested records per provider call, independent of the report source cap. |
| `MAX_QUERY_CHARS` | 1500 | Application limit after whitespace normalisation; oversized queries return an explicit shorten/split error before spending search budget. |
| `SEARCH_SNIPPET_CHARS` | 700 | Maximum preview characters per source in native search results and the compact source cache. |
| `EVIDENCE_PAGE_CHARS` | 2000 | Characters of original provider JSON returned by `read_evidence(source_id=..., offset=...)`. |
| `EVIDENCE_CATALOGUE_PAGE_SIZE` | 20 | Records per `read_evidence(offset=...)` catalogue page when source ID is omitted. |

Search results include the requested/effective record limit and preview character limit.
The supported API maxima are [100 OpenAlex records per page](https://help.openalex.org/api/paging/)
and [20 Tavily results per request](https://docs.tavily.com/documentation/api-reference/endpoint/search).
Larger `SEARCH_RESULTS_PER_QUERY` values are capped to that provider's page maximum and reported
as such. Search tools do not automatically paginate the provider API; extra searches remain the
agent's choice within its budget. `MAX_SOURCES` can exceed either page maximum across searches.

The 1500-character query default follows [Tavily's concise-query guidance](https://docs.tavily.com/documentation/best-practices/best-practices-search),
not a universal provider-enforced maximum. It is configurable. Queries are never silently cut.
Blank queries and requests denied by provider policy also spend no search-call budget.

Search previews are intentionally bounded. Full retrieved JSON remains in the evidence store;
shortening a preview does not delete it or change source IDs. `read_evidence` returns `next_offset`
until all original JSON or catalogue records have been read. Offsets count characters for an
original record and records for a catalogue. Negative offsets are rejected. The raw JSON is the
original provider result, not necessarily full article text. Grounding checks continue to read
original evidence rather than previews. Citation URLs are preserved in full: unlike previews,
shortening source identifiers can break links or merge distinct evidence records.

Larger record counts, snippets and pages increase prompt pressure. Choose finite budgets suitable
for the configured context and use paged reads for detail; no setting automatically tunes VRAM.

### Fixed safeguards and policies retained

Three perspectives, valid source IDs, research-only scope and per-turn provider isolation remain
policy, not tuning options. A report gets at most one structural correction and one token-limit
continuation to prevent repeated regeneration. Tavily retains basic search without generated
answers, images or full-page extraction, preserving the existing integration and request cost.

Cancellation, timeouts and retry behaviour are unchanged: provider requests have a 30-second
total deadline (including up to three attempts), Ollama requests have a 180-second total deadline,
and read inactivity limits remain 10/60 seconds respectively. Connect/pool limits are 5 seconds;
write limits are 10/15 seconds. Shutdown hooks are bounded to 5 seconds and cancellation polling
uses 50 milliseconds. These keep stalled requests and retries finite.

Compact metadata still caps titles at 300 characters, author names at 90 characters and authors
at four. OpenAlex abstract decoding retains its defensive nonnegative-position check below
10,000. These do not erase the original provider JSON. Standalone normalisation helpers retain
their 2200-character preview default for compatibility; the application's research tools always
use `SEARCH_SNIPPET_CHARS`. Grounding's conservative matching rules are unchanged and are not
an inference setting or a guarantee of factual accuracy.

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

