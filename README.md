pt_politics_mvp/
├── data/
│   ├── raw_pdfs/             # Drop your PDFs here
│   ├── taxonomy.json         # List of categories: ["Saúde", "Educação", "Fiscal"]
│   └── database/             # SQLite or Postgres
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   └── pdf_processor.py  # marker-pdf wrapper
│   ├── processing/
│   │   ├── segmenter.py      # Splits text into chunks/proposals
│   │   └── cleaner.py        # Basic text cleanup
│   ├── analysis/             # <--- THE CORE WORK HAPPENS HERE
│   │   ├── categorizer.py    # MVP: Assigns topics (Health, Tax, etc.)
│   │   ├── neutralizer.py    # MVP: Summarizes/Rewrites proposals neutrally
│   │   ├── populism.py       # FUTURE: Empty file (Placeholder)
│   │   └── constitution.py   # FUTURE: Empty file (Placeholder)
│   └── synthesis/
│       └── macro_summary.py  # FUTURE: Empty file (Placeholder)
├── main.py                   # Runs the simplified flow
└── requirements.txt