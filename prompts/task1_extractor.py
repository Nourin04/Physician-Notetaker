from groq_config import client

system_prompt = """
You are an experienced NLP and healthcare expert working on medical transcription summarization.

You will be given a transcript between a doctor and a patient. Your task is to extract a structured JSON object with the following 6 fields:

1. "Patient_Name" – Use "Janet Jones" as default if name is not mentioned.
2. "Symptoms" – List clearly mentioned or implied symptoms.
3. "Diagnosis" – Use medical reasoning to infer a likely diagnosis.
4. "Treatment" – Include all mentioned treatments, and intelligently infer standard treatments (e.g., painkillers for pain).
5. "Current_Status" – Describe current health based on latest patient comment.
6. "Prognosis" – Predict the likely recovery or outcome based on symptoms and treatment, even if not explicitly stated.

💡 Use proper medical terminology and return a valid JSON object ONLY — no notes or explanation.
"""

def extract_info(conversation):
    user_prompt = f"Extract the following fields from this transcript:\n{conversation}"

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
