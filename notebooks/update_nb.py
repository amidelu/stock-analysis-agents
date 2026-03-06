import json

notebook_path = "/Volumes/Storage/Development/AI/stock-analysis-agents/notebooks/kaggle_submission_complete.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "source" in cell:
        for i, line in enumerate(cell["source"]):
            # Change titles and texts
            if "## Google Gemini ADK Capstone Project" in line:
                cell["source"][i] = line.replace(
                    "Google Gemini", "Google Gemini & Groq"
                )
            if "Google's Agent Development Kit (ADK) and Gemini models." in line:
                cell["source"][i] = line.replace(
                    "Gemini models", "Gemini & Groq models"
                )
            if 'model=Gemini(model="gemini-2.0-flash-exp")' in line:
                cell["source"][i] = line.replace(
                    'model=Gemini(model="gemini-2.0-flash-exp")',
                    "model=get_llm_model() # Dynamically switch between Gemini or Groq",
                )
            if "15. ✅ **Long Context** - Leveraging Gemini's 2M token window" in line:
                cell["source"][i] = line.replace(
                    "Leveraging Gemini's 2M token window",
                    "Leveraging Gemini & Groq large token windows",
                )
            if "- Google Gemini 2.0 Flash (Exp)" in line:
                cell["source"][i] = line.replace(
                    "- Google Gemini 2.0 Flash (Exp)",
                    "- Google Gemini 2.0 Flash (Exp) / Groq Llama/GPT",
                )

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
