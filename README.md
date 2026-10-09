# Agriculture Domain Question Answering

Fine-tuned `google/flan-t5-base` on 20,000 agricultural Q&A pairs.

## Results

| Model | ROUGE-L | Exact Match | BERTScore F1 |
|---|---|---|---|
| TF-IDF Retrieval | 0.506 | 0.187 | 0.911 |
| flan-t5-base | 0.327 | 0.000 | 0.878 |

## Links
- 🤗 Model: https://huggingface.co/Toronga/flan-t5-agri-qa
- 🚀 Live Demo: (pending Streamlit deployment)

## Setup
```bash
pip install -r requirements.txt
streamlit run src/app.py
