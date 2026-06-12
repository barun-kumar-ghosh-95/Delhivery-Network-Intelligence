# Optimizing Delivery ETAs with Graph-Based Network Intelligence

This repository contains the complete codebase, dashboard, and documentation for the graph-based network intelligence system built to optimize delivery ETAs for the logistics network.

## 🚀 Live Dashboard (Localhost)

The production-ready web application is configured to run locally. You can access the fully functional, interactive BI dashboard by clicking the link below (ensure the Streamlit server is running):

> **[🌍 OPEN DASHBOARD: http://localhost:8501](http://localhost:8501)**

## Project Structure
- `dashboard/`: Streamlit source code for the interactive BI application.
- `src/`: Core Python modules for data processing, graph feature engineering (Node2Vec, PageRank), and machine learning (LightGBM).
- `configs/`: Pipeline configuration variables.
- `scripts/`: Execution scripts, including pipeline runner and PPT generator.
- `reports/`: Strategy memos and final documentation.

## Running the Dashboard Locally
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run dashboard/app.py
   ```

## Included in Final Submission Package
- Full Source Code (clean, formatted, and strictly professional)
- Fully functional 6-page interactive BI Dashboard
- Presentation Deck (`presentation.pptx`)
- Strategy Memo and Technical Docs
- Subsampled Data for Reproducibility
