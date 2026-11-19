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
acknowledge insufficient evidence instead of inventing sources. Conversational follow-ups must
answer the latest question directly with at least one valid supporting citation, using relevant
conversation/evidence and searching autonomously when needed. They do not repeat the full report
or its source minimum. If the user explicitly asks for another full report, the model sets
`full_report=true` on the existing routing tool and full report validation applies. The current
response format is included in the native route result and turn instructions; the guide and
report skill use the same policy. No additional agent or orchestration stage is involved.

`MAX_SOURCES` counts distinct valid IDs cited in the prepared narrative, not retrieved records,
repeat citations, source-only attribution labels, or model-generated bibliographies. Factual
claims remain in the narrative even if prefixed with `Source:`. The same existing validation hook checks
this limit before its single correction opportunity. If that correction still fails, the
rejected draft is labelled incomplete with the exact violation and retains its genuine
references, even if those exceed the cap. It is not accepted as a compliant report. Only unknown
IDs become `[UNVERIFIED]`; a real retrieved source is never relabelled merely to shorten the
reference list. All retrieved evidence remains available for follow-ups.

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

Opt-in managed Bedrock KB and local-only control: see [RAG setup and commands](docs/RAG.md).
Existing research remains the default. RAG uses local Ollama generation with no web fallback.
The RAG path has its own narrow citation check; the research grounding checks below apply
to the existing research route.

Requests have explicit timeouts and cooperative cancellation, including stalled Ollama
streams. Reports also check sentence-sized claims against original retrieved evidence,
label conflicts and uncertainty in the same response, and save the checks as `grounding.json`.

Structural headings, separators and citation-only footers are excluded from claim extraction.
Cited factual assertions remain checked even when styled as headings. A detected contradiction
stays beside the actual claim as **Unsupported by cited evidence**, with its reason and source
IDs. Missing citations, unavailable evidence, unknown IDs and conflicting cited sources also
remain marked inline. Semantic abstentions use short `[Review C#]` markers and one explanatory
notice instead of repeating a long warning on every sentence. Each claim's original text,
citations, status, reason, evidence excerpt and stable claim ID remain in Agent Activity and
the saved JSON. Unresolved semantic support is not treated as factual correctness; valid
paraphrases and unsupported inferences can both require review. No verifier model or additional
validation/rewrite loop is added.

To repeat the exact two-question scenario through real PyQt widgets, Ollama and the selected research provider:

```bash
QT_QPA_PLATFORM=offscreen agent-env.venv/bin/python -B scripts/verify_adhd_gui.py
```

This opt-in check uses the existing `.env` unchanged and consumes research API requests. It sends
`why is there a surge in the ADHD diagnoses`, followed by `how can I manage my ADHD symptoms then`.
Requests, rendered text, screenshots, activity and a summary are saved under the printed `/tmp`
directory. It checks the same agent/worker, retained question history, direct follow-up format,
response-card count, Qt responsiveness and orderly shutdown. It does not establish medical or
research accuracy of the model's answers.

## GUI trace export

The GUI's **Results to retrieve** field starts from `BEDROCK_NUMBER_OF_RESULTS`
(default 5). With RAG selected, change it between questions to request 1–100 results
on the next retrieval, without restarting. Each question uses the value present
when Send is pressed; the value is recorded in the trace and `evidence.json`.
This session control does not write to `.env` or change CLI settings.

The right-hand Agent Activity panel shows `KB_RETRIEVAL_RESULTS` with complete
returned chunks and metadata, `KB_CONTEXT` with the exact selected/clipped excerpts,
and `MODEL_INPUT` with the system prompt and messages passed to the local model.
Raw model text, metadata and citation-validation events also appear there.

Closing the GUI waits for worker shutdown, then saves the entire session to
`src/agent/memory/runs/gui-<timestamp>-<id>/trace.json`. The file contains ordered,
timestamped events, the exact visible panel text in `rendered_text`, and links to
question-level run directories containing `evidence.json`. Its path appears in
the panel's initial `SESSION_TRACE` entry and is printed on successful close.
If saving fails, the window stays open and reports the error so closing can be retried.
These traces include document text and remain local in the ignored runs directory.

KB citations use linked original document titles and returned page metadata when
available. `[KB…]` markers are clickable for HTTP(S) sources. No excerpt line numbers,
invented page numbers or LLM-generated source titles are added.

One valid KB citation can support related statements across paragraphs or a list;
citations are not required in every paragraph. The model must still use relevant
retrieved evidence. The structural check rejects missing or unknown source IDs,
but does not prove semantic support. Citation failures report a validation problem
rather than claiming that the KB lacks evidence; `KB_VALIDATION` includes the reason.

RAG reserves `OUTPUT_TOKENS` and 512 tokens for framing, then packs excerpts using
an input estimate: two ASCII bytes per token, with non-ASCII UTF-8 bytes counted
individually. Serialized source metadata and JSON escaping count toward that budget.
This is an estimate, not the selected model's tokenizer; long excerpts still have
the existing 2,000-byte cap and any clipping is shown in the trace.

