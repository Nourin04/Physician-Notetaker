# Physician Notetaker

👋 Welcome! This project implements an AI-powered pipeline for **medical transcription understanding, NLP-based summarization, sentiment analysis**, and **SOAP note generation** from doctor-patient conversations.

This was built and deployed using the **Groq API with LLaMA 3** for inference, served through a **Flask app**, and deployed on **Render**.

---

## 🩺 Sample Use Case
### Deployed Link: https://physician-notetaker-j2fd.onrender.com/

![image](https://github.com/user-attachments/assets/7f38381b-1d57-48d4-9260-a9d04b446a3d)
![image](https://github.com/user-attachments/assets/1bcdb7ec-003e-406f-9843-fb054fab0a2e)
![image](https://github.com/user-attachments/assets/18cc6b5b-9654-45b4-99ba-d13823e45d46)




A physician-patient conversation is passed to the system, and the app:

* Extracts key medical details (like symptoms, diagnosis, treatment, etc.)
* Analyzes the patient's **sentiment** and **intent**
* Generates a **structured SOAP note**

---

## 🧠 Tasks Implemented

### ✅ Task 1: Medical NLP Summarization

**Objective:** Extract structured medical data from a free-flowing conversation transcript.

**Delivered:**

* Named Entity Extraction: `Symptoms`, `Diagnosis`, `Treatment`, `Prognosis`, etc.
* Conversion to JSON with fields: `Patient_Name`, `Current_Status`, etc.
* Done via carefully crafted prompts to the Groq-hosted LLaMA 3 model.

**Sample Input:**

```
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
```
**Sample Output:**

```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}
```

**Q: How would you handle ambiguous or missing medical data?**
**A:** The LLM is prompted to explicitly state "Not mentioned" for missing fields. This avoids hallucination and enforces completeness.

**Q: What pre-trained models are ideal for medical summarization?**
**A:** Models like `BioGPT`, `ClinicalBERT`, and `LLaMA 3` (for instruction-tuned generation) are suitable depending on compute constraints. `medSpaCy` is also a strong choice for biomedical NER but would require task-specific training for accurate extraction.

---

### ✅ Task 2: Sentiment & Intent Analysis

**Objective:** Detect the emotional tone and intent of the patient's message.

**Delivered:**

* Sentiment Classification: `Anxious`, `Neutral`, `Reassured`
* Intent Detection: e.g., `Seeking reassurance`, `Expressing concern`, etc.

**Sample Output:**

```json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}
```

**Q: How to fine-tune BERT for sentiment?**
**A:** Label a medical-specific sentiment dataset, tokenize using BERT tokenizer, and fine-tune on `cross-entropy` loss for 3–5 epochs.

**Q: What datasets can be used?**
**A:** `i2b2`, `MTSamples`, or synthetic datasets from patient-provider dialogues.

---

### ✅ Task 3: SOAP Note Generation (Bonus)

**Objective:** Automatically generate a clinician-style SOAP note from the transcript.

**Delivered:**

* Structured fields: `Subjective`, `Objective`, `Assessment`, and `Plan`
* Each with nested fields like `Chief_Complaint`, `Diagnosis`, `Follow_Up`, etc.

**Sample Output:**

```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident..."
  },
  "Objective": {
    "Physical_Exam_Findings": "Full range of motion...",
    "Clinical_Facts": "No signs of damage..."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury",
    "Severity": "Mild"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy",
    "Medication": "Painkillers",
    "Follow_up_Plan": "Return if pain worsens"
  }
}
```

**Q: How to train an NLP model for SOAP mapping?**
**A:** Fine-tune an LLM (e.g., T5 or LLaMA) on annotated clinical dialogue-to-SOAP data. Alternatively, use instruction-tuned prompting with a large model like `LLaMA 3`.

**Q: What techniques improve accuracy?**
**A:** Use structured prompts, few-shot examples, and rule-based post-processing to enforce output schema.

---

## 🔬 Experimental Notes

* ✅ **Synthetic data fine-tuning was attempted** on a multi-label BERT model for sentiment and intent classification. However, performance degraded slightly on real-world samples.

  * Conclusion: **Fine-tuning needs a larger real dataset + more training epochs** for generalization.

* ⚠️ For **NER**, libraries like `medSpaCy` and `scispaCy` can provide domain-aware biomedical entities, but require **custom training or adaptation** for conversational text.

---

## 🧪 Technologies Used

| Component       | Tech Stack               |
| --------------- | ------------------------ |
| LLM Inference   | Groq API + LLaMA 3 (70B) |
| Backend         | Flask (Python)           |
| Frontend        | HTML + JavaScript        |
| Deployment      | Render                   |
| Format Handling | JSON                     |

---

## 🚀 How to Run Locally

1. Clone the repo

```bash
git clone https://github.com/your-username/physician-notetaker
cd physician-notetaker
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your `.env` with the Groq API key:

```env
GROQ_API_KEY=your_key_here
```

4. Run Flask app

```bash
python app.py
```

5. Open your browser at `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
├── app.py                 # Flask backend + API routes
├── groq_config.py         # All Groq API prompt functions
├── templates/
│   └── index.html         # Simple frontend UI
├── .env                   # Secret API key (not committed)
└── requirements.txt       # Dependencies
```

---

## ✍️ Authors & Acknowledgements

* 💡 Prompt Engineering: Done manually for Groq's LLaMA 3 to enforce structured outputs
* 🧪 Tested and evaluated on synthetic and real-world transcripts
* ⚙️ Inspired by EMR automation in clinical NLP research

---



**🧠 Final Note:** This project demonstrates the power of **instruction-tuned LLMs** for clinical applications. While it’s not a replacement for doctors, it can significantly assist in **documentation, summarization, and triage.**
