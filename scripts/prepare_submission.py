"""
Prepare Submission Script
=========================
Programmatically structures and packages the project into a competition-ready ZIP.
"""

import os
import shutil
import zipfile
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIR = os.path.join(PROJECT_ROOT, "Delhivery_Network_Intelligence")

def build_structure():
    print("[Packager] Creating submission folder structure...")
    folders = [
        "",
        "dashboard",
        "dashboard/pages",
        "dashboard/components",
        "src",
        "src/data",
        "src/graph",
        "src/features",
        "src/models",
        "configs",
        "scripts",
        "models",
        "results",
        "visualizations",
        "screenshots",
        "docs",
        "data_sample"
    ]
    
    for folder in folders:
        path = os.path.join(SUB_DIR, folder)
        os.makedirs(path, exist_ok=True)

def copy_source_code():
    print("[Packager] Copying source code, dashboards, and configs...")
    
    # Src modules
    src_base = os.path.join(PROJECT_ROOT, "src")
    for root, dirs, files in os.walk(src_base):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), src_base)
                dest = os.path.join(SUB_DIR, "src", rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(os.path.join(root, file), dest)
                
    # Dashboard
    dash_base = os.path.join(PROJECT_ROOT, "dashboard")
    for root, dirs, files in os.walk(dash_base):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), dash_base)
                dest = os.path.join(SUB_DIR, "dashboard", rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(os.path.join(root, file), dest)

    # Configs
    cfg_base = os.path.join(PROJECT_ROOT, "configs")
    for file in os.listdir(cfg_base):
        if file.endswith(".yaml") or file.endswith(".yml"):
            shutil.copy2(os.path.join(cfg_base, file), os.path.join(SUB_DIR, "configs", file))
            
    # Scripts
    scr_base = os.path.join(PROJECT_ROOT, "scripts")
    for file in os.listdir(scr_base):
        if file.endswith(".py") and file != "prepare_submission.py":
            shutil.copy2(os.path.join(scr_base, file), os.path.join(SUB_DIR, "scripts", file))
            
    # Requirements.txt
    shutil.copy2(os.path.join(PROJECT_ROOT, "requirements.txt"), os.path.join(SUB_DIR, "requirements.txt"))

def copy_trained_models():
    print("[Packager] Copying serialized model artifacts...")
    model_src = os.path.join(PROJECT_ROOT, "data", "models")
    if os.path.exists(model_src):
        for file in os.listdir(model_src):
            if file.endswith(".pkl"):
                shutil.copy2(os.path.join(model_src, file), os.path.join(SUB_DIR, "models", file))

def copy_processed_results():
    print("[Packager] Copying processed results and analytics summaries...")
    processed_src = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # Copy main outputs
    if os.path.exists(processed_src):
        for file in os.listdir(processed_src):
            if file.endswith(".json") or file.endswith(".csv"):
                shutil.copy2(os.path.join(processed_src, file), os.path.join(SUB_DIR, "results", file))
                
        # Copy graph analysis reports
        analysis_src = os.path.join(processed_src, "analysis")
        if os.path.exists(analysis_src):
            dest_analysis = os.path.join(SUB_DIR, "results", "analysis")
            os.makedirs(dest_analysis, exist_ok=True)
            for file in os.listdir(analysis_src):
                shutil.copy2(os.path.join(analysis_src, file), os.path.join(dest_analysis, file))

def create_data_sample():
    print("[Packager] Sampling raw delivery data to keep submission size reasonable...")
    raw_csv = os.path.join(PROJECT_ROOT, "delivery_data.csv")
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv, nrows=1000)
        df.to_csv(os.path.join(SUB_DIR, "data_sample", "delivery_data_sample.csv"), index=False)
        print("[Packager] Saved 1,000 row data sample.")
    else:
        print("[Packager] Warning: delivery_data.csv not found, skipping sampling.")

def write_docs_templates():
    print("[Packager] Writing strategic report, deck, and strategy memo markdown templates...")
    
    # Memo
    memo_path = os.path.join(PROJECT_ROOT, "reports", "executive_memo.md")
    if os.path.exists(memo_path):
        shutil.copy2(memo_path, os.path.join(SUB_DIR, "strategy_memo.md"))
        
    # Draft Report and PPT markdown templates inside docs/
    report_desc = "# REPORT: Graph-Based Network Intelligence\n(See Phase 4 in Submission Report for exact guidelines)"
    with open(os.path.join(SUB_DIR, "docs", "report_guidelines.md"), "w") as f:
        f.write(report_desc)
        
    ppt_desc = "# PRESENTATION: Pitch Deck Slides\n(See Phase 5 in Submission Report for exact guidelines)"
    with open(os.path.join(SUB_DIR, "docs", "presentation_deck.md"), "w") as f:
        f.write(ppt_desc)

def create_zip():
    zip_path = os.path.join(PROJECT_ROOT, "Delhivery_Network_Intelligence.zip")
    print(f"[Packager] Compiling ZIP package at {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(SUB_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                zipf.write(file_path, rel_path)
                
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[Packager] Zip creation complete! Size: {size_mb:.2f} MB")

def main():
    if os.path.exists(SUB_DIR):
        shutil.rmtree(SUB_DIR)
        
    build_structure()
    copy_source_code()
    copy_trained_models()
    copy_processed_results()
    create_data_sample()
    write_docs_templates()
    
    # Write the README inside the subfolder
    shutil.copy2(os.path.join(PROJECT_ROOT, "README.md"), os.path.join(SUB_DIR, "README.md"))
    
    # Copy presentation
    ppt_path = os.path.join(PROJECT_ROOT, "presentation.pptx")
    if os.path.exists(ppt_path):
        shutil.copy2(ppt_path, os.path.join(SUB_DIR, "presentation.pptx"))
        
    # Copy render.yaml
    render_path = os.path.join(PROJECT_ROOT, "render.yaml")
    if os.path.exists(render_path):
        shutil.copy2(render_path, os.path.join(SUB_DIR, "render.yaml"))
        
    create_zip()
    print("[Packager] READY FOR SUBMISSION PREPARATION!")

if __name__ == "__main__":
    main()
