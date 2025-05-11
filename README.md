# 🩺 DiaBuddy: A Diabetes Health Care Assistant using LLMs (LLaMA-3 + LoRA + RAG)

**Team 31**
🧑‍💻 Vaibhav Vinay Ranashoor
🧑‍💻 Rohan Jain
_CSE 574 – (Spring 2025)_
University at Buffalo

---

## 🔬 Project Overview

DiaBuddy is a **privacy-preserving, LLM-based diabetes assistant** built with:

- 🦙 **LLaMA-3 8B Instruct model**
- 🧠 **LoRA-based fine-tuning** on 4K+ curated QA pairs
- 📚 **RAG (Retrieval-Augmented Generation)** using a 2032-page trusted medical PDF
- 📊 Full evaluation using BLEU, ROUGE-L, BERTScore, Hallucination Rate, Cosine Similarity, and t-SNE

Our goal was to **enable lightweight, clinically coherent diabetic care chatbots** that run on a single GPU (Colab Pro, A100) and respect data locality—**no cloud APIs**.

---

## 🗂️ Project Structure

```
.
├── data/
│   ├── diabetes_qa_llama2_all_till_date_no_dup.jsonl   # Final curated SFT dataset
│   ├── diabetes_qa_test_data.jsonl                     # 50-QA test set
│   └── small_rag_doc.pdf
│   └── small_rag_doc.pdf                               # 2032-page RAG data source
|
├── docs/
│   ├── CSE_574_Project_Final_Report_Team_31.docx
│   └── Final_presentation_team_31.pptx
│
├── src/
│   ├── Diabetes_LLM_Finetune_llama_3.ipynb             # LoRA fine-tuning notebook
│   └── Diabetes_Lllama3_finetune_and_RAG.ipynb         # RAG + Evaluation notebook
│
└── README.md                                           # ← You're here!
```

---

## 🔧 Setup Instructions (Google Colab Only)

> 💻 You need a **Google Colab Pro** subscription with **NVIDIA A100 GPU** access.

### ✅ 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 📦 2. Install Dependencies

```bash
!pip install --upgrade pip
!pip install transformers peft accelerate langchain faiss-cpu bitsandbytes \
sentence-transformers datasets evaluate bert_score
```

---

## 🧠 Part 1: Fine-Tuning LLaMA-3 using LoRA (SFT)

### 🧾 LoRA Configuration

- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- LoRA: `r=8`, `alpha=16`, `dropout=0.1`
- Optimizer: AdamW, lr=2e-5
- Epochs: 3
- Trained on 4277 diabetes-specific Q\&A pairs

### 📁 Fine-tuned Artifacts (Saved Separately in Google Drive)

You can download the pretrained LoRA adapter weights and tokenizer files here:
🔗 [Google Drive – LoRA Adapter Weights](https://drive.google.com/drive/folders/1bF4VCPMRd-TgX9WRt3Z1FS_rRCuYi0Ud?usp=sharing)

Contents:

```
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
README.md
```

To use them:

```python
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained("<downloaded-folder-path>")
model = PeftModel.from_pretrained(base_model, "<downloaded-folder-path>")
```

---

## 💡 To Reproduce This Project

> Clone this repo and run notebooks in Colab.

### 🔁 Step-by-Step

1. Upload contents of `data/` to your Google Drive
2. Download LoRA adapter weights from this link:
   🔗 [https://drive.google.com/drive/folders/1bF4VCPMRd-TgX9WRt3Z1FS_rRCuYi0Ud?usp=sharing](https://drive.google.com/drive/folders/1bF4VCPMRd-TgX9WRt3Z1FS_rRCuYi0Ud?usp=sharing)
   and place them in your Drive folder
3. Open `Diabetes_LLM_Finetune_llama_3.ipynb` → Fine-tune (optional if using above)
4. Open `Diabetes_Lllama3_finetune_and_RAG.ipynb` → Run:

   - PDF → Chunking + FAISS
   - Generate responses (RAG + Non-RAG)
   - Evaluate BLEU, ROUGE-L, BERTScore, Cosine Sim, t-SNE, Hallucination

---

## 🔍 Part 2: Retrieval-Augmented Generation (RAG)

### 📘 Data Source

- `small_rag_doc.pdf` (2032 pages): Compiled official diabetes guidelines
- Chunked into 500-token segments using LangChain
- Embeddings via `all-MiniLM-L6-v2`
- Indexed using `FAISS`

### 🔄 RAG Pipeline

```python
# Retrieve top-K context chunks
context = retriever.similarity_search(query, k=20)

# Concatenate context + user query → generate response
response = model.generate(context + query)
```

---

## 📈 Evaluation Pipeline

| Metric            | Library              | Base Model           | Fine-Tuned + RAG                   |
| ----------------- | -------------------- | -------------------- | ---------------------------------- |
| **BLEU**          | 🤗 evaluate          | ✅ 0.49%             | ✅ 0.81%                           |
| **ROUGE-L**       | 🤗 evaluate          | ✅ 6.86%             | ✅ 8.23%                           |
| **BERTScore**     | 🤗 bert-score        | ✅ 0.834             | ✅ 0.841                           |
| **Cosine Sim.**   | Sentence-Transformer | ✅ 0.474             | ✅ 0.448 (↓ due to length/fluency) |
| **Hallucination** | BART-MNLI            | ✅ 0% contradictions | ✅ 0%, but more "Neutral"          |
| **t-SNE**         | sklearn              | 📉 Overlapping       | ✅ Category clusters               |

### 📊 Hallucination Evaluation (BART-MNLI)

| Model | Entailment | Neutral | Contradiction (Hallucination) |
| ----- | ---------- | ------- | ----------------------------- |
| Base  | 46         | 4       | **0**                         |
| RAG   | 41         | 9       | **0**                         |

### 🧠 Key Insights

- RAG improves factuality but increases verbosity → more neutrals
- No contradictions = No dangerous hallucinations 🚫
- Curated datasets + strong prompt engineering = high alignment 💡

---

## 💡 To Reproduce This Project

> Clone this repo and run notebooks in Colab.

### 🔁 Step-by-Step

1. Upload contents of `data/` and `finetuned/` to your Google Drive
2. Open `Diabetes_LLM_Finetune_llama_3.ipynb` → Fine-tune if needed
3. Open `Diabetes_Lllama3_finetune_and_RAG.ipynb` → Run:

   - PDF → Chunking + FAISS
   - Generate responses (RAG + Non-RAG)
   - Evaluate: BLEU, ROUGE-L, BERTScore, CosSim, t-SNE, Hallucination Count

---

## 🔐 Privacy and Safety Notes

- We **do not send any prompts to OpenAI or external APIs**
- Fully on-device (Colab A100), LoRA finetuning, private PDF ingestion
- Model not a medical device; for educational/research use only

---

## 📌 Future Work

- 📲 Mobile deployment via quantization and ONNX
- 🌐 Multilingual support (MedDialog-CN)
- 🧾 Real-time CGM integration
- 🩺 Clinical validation using expert review

---

## 📚 References

See full list in [`docs/CSE_574_Project_Final_Report_Team_31.docx`](docs/CSE_574_Project_Final_Report_Team_31.docx)

---

## 🧑‍🔬 Contributions

| Team Member | Contributions                                                               |
| ----------- | --------------------------------------------------------------------------- |
| **Vaibhav** | Finetuning, RAG Setup, Evaluation, Result Analysis, Dataset Curation        |
| **Rohan**   | Prompting, Dataset Preparation, LoRA Config, Hallucination Detection, t-SNE |
