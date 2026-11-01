# AI PDF Resume Analyzer & Recruitment Assistant 🚀

An automated HR evaluation and candidate management system built with Python, Streamlit, SQLite, and multi-model AI APIs (Groq & Gemini). This platform allows recruiters to upload candidate PDF resumes, automatically evaluate and rank applicants against job descriptions, track data locally, and trigger automated candidate email updates.

---

## ✨ Live App
🌐 **Try the deployed application:** [AI PDF Resume Analyzer](https://ai-pdf-resume-analyzer-crokxrefvw8euwpxhtwkts.streamlit.app)

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Backend:** Python 3.x
* **AI & LLM Integration:** Groq API & Google Gemini API (`google-genai`)
* **Database:** SQLite (`candidates.db`)
* **Document Handling & Data Processing:** `pypdf`, `pandas`, `openpyxl`
* **Email Services:** Python `smtplib` (Gmail SMTP)

---

## ⚡ Key Features

* 📄 **PDF Extraction & Parsing:** Automatically extracts structured text and candidate metadata from uploaded PDF resumes.
* 🤖 **AI Candidate Scoring:** Uses LLM reasoning to grade, rank, and match candidate qualifications against customized job descriptions.
* 📊 **SQLite Candidate Database:** Persists applicant scoring history, email logs, and top-candidate flags inside a local database.
* ✉️ **Automated SMTP Outreach:** Generates and sends automated status emails directly to candidates.
* 📈 **Excel Export:** Exports sorted candidate evaluation metrics into cleanly formatted `.xlsx` spreadsheets using `openpyxl`.

---

## 📦 Local Installation & Setup

If you want to run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ummara606/ai-pdf-resume-analyzer.git](https://github.com/Ummara606/ai-pdf-resume-analyzer.git)
   cd ai-pdf-resume-analyzer
