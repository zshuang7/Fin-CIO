# Financial Multi-Agent System

A multi-agent stock analysis system powered by **DeepSeek**, **LiteLLM**, and the **Agno** framework.

## Architecture

```
.
├── agents/
│   ├── __init__.py
│   └── team_config.py      # Agent A (Researcher), Agent B (NewsScout), Agent C (ReportManager)
├── tools/
│   ├── __init__.py
│   ├── finance_engine.py   # yfinance wrapper (Agno Toolkit)
│   └── report_engine.py    # Excel & PDF report generation
├── reports/                # Generated reports land here (auto-created)
├── .env                    # API keys (never commit this)
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Researcher** | Fetches 3-year financials, P/E, EPS, dividends | `FinanceEngine` (yfinance) |
| **NewsScout** | Retrieves latest headlines & market sentiment | `DuckDuckGoTools` |
| **ReportManager** | Synthesises BUY/HOLD/SELL & saves reports | `ReportEngine` (Excel + PDF) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

The `.env` file is pre-configured. You can edit it to swap models:

```env
DEEPSEEK_API_KEY=your_key_here
MODEL_ID=deepseek/deepseek-chat          # or deepseek/deepseek-reasoner, openai/gpt-4o
FMP_API_KEY=your_fmp_key_here            # Financial Modeling Prep (analyst consensus vote)
```

### 3. Run

**Interactive mode (recommended):**
```bash
python main.py
```

**Single query:**
```bash
python main.py "Analyze TSLA"
python main.py "Analyze AAPL"
```

**Debug mode:**
```bash
python main.py "Analyze NVDA" --debug
```

## Workflow

When you type `Analyze TSLA`, the team coordinator:

1. Sends the Researcher to fetch 3-year financials + key metrics
2. Sends the NewsScout to fetch 5 recent headlines + sentiment
3. Passes both to the ReportManager, which:
   - Writes a BUY / HOLD / SELL recommendation
   - Saves `reports/TSLA_data.xlsx` (multi-sheet Excel)
   - Saves `reports/TSLA_report_<timestamp>.pdf`

## Swapping to OpenAI

Change `.env`:
```env
OPENAI_API_KEY=sk-...
MODEL_ID=openai/gpt-4o
```

No code changes needed — LiteLLM handles the routing.
