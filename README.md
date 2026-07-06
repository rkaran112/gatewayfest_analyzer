# FestPulse — GATEWAYS 2025 Analytics Dashboard

A single-page Streamlit dashboard that analyzes participation, sentiment, and revenue data from the GATEWAYS 2025 fest (Christ University, Dept. of Computer Science).

## What it does

FestPulse loads a CSV of fest registrations and feedback and turns it into an interactive analytics terminal with four tabs:

- **Cartography** — a choropleth map of participant density across Indian states, event/college participation bar charts, and a state × revenue treemap.
- **The Signal** — sentiment analysis of free-text feedback (via TextBlob polarity/subjectivity), a custom emotion classifier (Joy, Trust, Frustration, Anger, Fear) and root-cause theme tagger (Logistics, Mentorship, Scheduling, Judging, Registration, Networking) built on keyword lexicons, a "signal confidence" score, a risk radar per event/state/college, word clouds by sentiment, rating distributions, and a searchable feedback table.
- **Command** — an engagement funnel (registered → gave feedback → rated 5 stars), revenue breakdowns by event and event type, an auto-generated "Oracle" text report of key insights, a "happiest delegation" callout, and a revenue-goal progress tracker (hardcoded goal of ₹150,000).
- **Benchmark** — a configurable weighted composite score (participation / rating / sentiment / revenue) to rank colleges, states, or events, with radar and scatter comparisons and an auto-generated benchmark brief.

Sidebar filters (state, event, event type, rating, sentiment) apply across all tabs, and filtered data can be exported as CSV.

## Tech stack

- Python
- [Streamlit](https://streamlit.io/) — UI/dashboard framework
- pandas — data loading and aggregation
- Plotly (express + graph_objects) — interactive charts (choropleth, treemap, heatmaps, radar, funnel, etc.)
- TextBlob + NLTK — sentiment polarity/subjectivity and stopword filtering
- WordCloud + Matplotlib — sentiment word clouds
- Pillow

## Setup

```bash
pip install -r requirements.txt
```

The app also downloads the NLTK `stopwords` corpus automatically on first run (`nltk.download("stopwords", quiet=True)`).

## Usage

The dataset CSV (`C5-FestDataset - fest_dataset - C5-FestDataset - fest_dataset.csv`) must sit in the same directory as `app.py` — the filename is hardcoded in `app.py` as `DATA_FILE`. Expected columns: `Student Name, College, Phone Number, Place, State, Event Name, Event Type, Amount Paid, Feedback on Fest, Rating`.

Run the dashboard with:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Status

**Complete for its current scope.** The app is a single, working `app.py` with no stub functions, TODOs, or obviously missing pieces — all four tabs are fully implemented and the data-loading path has basic error handling (missing file / load errors surface as Streamlit error messages instead of crashing).

Known limitations, not necessarily bugs to fix, but worth knowing before extending:
- The dataset path/filename and the ₹150,000 revenue goal are hardcoded in `app.py` rather than configurable.
- Sentiment/emotion/theme detection use simple keyword lexicons and TextBlob polarity rather than a trained NLP model, so results are a heuristic approximation, not ground truth.
- The app is tailored specifically to the GATEWAYS 2025 dataset's column names; reusing it for another fest's data would require matching that schema.
- No automated tests are included.
