
import gradio as gr
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import fitz  # PyMuPDF
import torch
import soundfile as sf
import numpy as np
import re
import pytesseract
from PIL import Image
import io
import os
import json
from num2words import num2words
from sentence_transformers import SentenceTransformer, util

# ── Windows Tesseract paths ──────────────────────────────────────────────────
# Update these paths if your Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r'D:\exam\Tesseract-OCR\tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r'D:\exam\Tesseract-OCR\tessdata'

# ── Model loading ────────────────────────────────────────────────────────────
print("Loading AI models... please wait...")

# Better QA model — flan-t5-large gives much more accurate answers than base
qa_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",   # upgraded from flan-t5-base
    max_new_tokens=200
)

processor   = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model   = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder     = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Safer speaker embedding — pick index 0 and guard against dataset size
_xvec_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
)
_safe_idx = min(7306, len(_xvec_dataset) - 1)
speaker_embeddings = torch.tensor(
    _xvec_dataset[_safe_idx]["xvector"]
).unsqueeze(0)

stt              = pipeline("automatic-speech-recognition", model="openai/whisper-small")
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

student_answers: dict = {}

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_questions(pdf_file) -> list[str]:
    """Extract questions from uploaded PDF (with OCR fallback)."""
    path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
    doc  = fitz.open(path)
    text = ""
    for page in doc:
        page_text = page.get_text().strip()
        if len(page_text) < 10:
            print(f"Page {page.number}: blank — using OCR")
            pix      = page.get_pixmap(dpi=300)
            img      = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"

    print("\n--- EXTRACTED TEXT ---\n", text, "\n----------------------\n")

    lines      = [l.strip() for l in text.split("\n") if l.strip()]
    questions  = []
    current_q  = ""

    for line in lines:
        is_new_q = bool(re.match(
            r'^(Question\s*\d+|Q\d+|\d+[\.\)]|[a-zA-Z][\)\.])', line, re.IGNORECASE
        )) or line.startswith((
            "What", "Define", "Explain", "Compare", "Examine", "How",
            "Describe", "List", "State", "Write", "Discuss", "Differentiate"
        ))

        if is_new_q:
            if current_q:
                questions.append(current_q.strip())
            current_q = line
        else:
            current_q = (current_q + " " + line).strip() if current_q else line

    if current_q:
        questions.append(current_q.strip())

    if not questions:
        questions = ["Warning: No questions detected. Please check the PDF format."]

    return questions


def clean_text_for_speech(text: str) -> str:
    text = re.sub(r'(\d+\.?\d*)([a-zA-Z]+)', r'\1 \2', text)
    text = re.sub(r'\bhr\b',  'hours',      text, flags=re.IGNORECASE)
    text = re.sub(r'\bmin\b', 'minutes',    text, flags=re.IGNORECASE)
    text = re.sub(r'\bkg\b',  'kilograms',  text, flags=re.IGNORECASE)
    text = re.sub(r'\bcm\b',  'centimeters',text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    text = text.replace("-", " ")
    # SpeechT5 chokes on very long inputs — truncate to ~500 chars
    return text[:500]


def speak(text: str) -> str:
    """Convert text to speech and return path to .wav file."""
    clean = clean_text_for_speech(text)
    inputs = processor(text=clean, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=vocoder
        )
    out_path = "output.wav"
    sf.write(out_path, speech.numpy(), samplerate=16000)
    return out_path


def generate_answer_key(questions: list[str]) -> dict:
    """
    Generate a model answer key using flan-t5-large.
    Uses a clear, structured prompt and deduplication to improve accuracy.
    """
    answer_key = {}

    for i, q in enumerate(questions):
        key = f"Q{i+1}"
        try:
            # Clean, focused prompt — avoids repetition and hallucination
            prompt = (
                f"Answer the following exam question in 2-3 clear, accurate sentences. "
                f"Do not repeat yourself.\n\nQuestion: {q}\n\nAnswer:"
            )
            result = qa_generator(
                prompt,
                max_new_tokens=150,
                do_sample=False,          # greedy → more factual
                repetition_penalty=2.0    # strong penalty to avoid repeated phrases
            )
            raw_answer = result[0]["generated_text"].strip()

            # Remove the prompt echo if model repeats it
            if "Answer:" in raw_answer:
                raw_answer = raw_answer.split("Answer:")[-1].strip()

            # Deduplicate sentences
            sentences  = re.split(r'(?<=[.!?])\s+', raw_answer)
            seen       = set()
            unique_sents = []
            for s in sentences:
                norm = s.strip().lower()
                if norm and norm not in seen:
                    seen.add(norm)
                    unique_sents.append(s.strip())
            answer = " ".join(unique_sents) or raw_answer

        except Exception as e:
            print(f"Error generating answer for {key}: {e}")
            answer = "Error generating answer"

        answer_key[key] = {"question": q, "answer": answer}
        print(f"  {key} answer generated: {answer[:80]}...")

    with open("auto_answer_key.json", "w") as f:
        json.dump(answer_key, f, indent=2)

    print("Answer key saved to auto_answer_key.json")
    return answer_key


def grade_answers_auto() -> str:
    """Grade student answers against the auto-generated answer key."""
    if not os.path.exists("student_answers.json"):
        return "❌ No student answers found. Complete the exam first."
    if not os.path.exists("auto_answer_key.json"):
        return "❌ No answer key found. Load an exam PDF first."

    with open("student_answers.json", "r") as f:
        student = json.load(f)
    with open("auto_answer_key.json", "r") as f:
        answer_key = json.load(f)

    if not student:
        return "❌ Student answers file is empty."

    results      = []
    total_marks  = 0
    max_possible = len(student) * 10

    for qnum, data in student.items():
        student_ans = data.get("answer", "").strip()
        model_data  = answer_key.get(qnum, {})
        model_ans   = model_data.get("answer", "").strip()

        if not model_ans:
            results.append(f"{qnum}: ⚠️  No model answer available")
            continue
        if not student_ans:
            results.append(f"{qnum}: ⚠️  No student answer recorded")
            continue

        emb1  = similarity_model.encode(student_ans, convert_to_tensor=True)
        emb2  = similarity_model.encode(model_ans,   convert_to_tensor=True)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        pct   = round(score * 100, 1)

        # Marks out of 10
        if   score > 0.85: marks = 10
        elif score > 0.75: marks =  8
        elif score > 0.60: marks =  6
        elif score > 0.45: marks =  4
        elif score > 0.30: marks =  2
        else:              marks =  0

        verdict = (
            "✅ Correct"   if score > 0.75 else
            "🔶 Partial"   if score > 0.45 else
            "❌ Incorrect"
        )
        total_marks += marks

        results.append(
            f"{qnum}: {verdict} | Marks: {marks}/10  (Similarity: {pct}%)\n"
            f"  🎓 Student : {student_ans}\n"
            f"  📘 Expected: {model_ans}\n"
        )

    avg_pct = round((total_marks / max_possible) * 100, 1) if max_possible else 0
    results.append(
        f"\n{'='*50}\n"
        f"Total Marks  : {total_marks} / {max_possible}\n"
        f"Overall Score: {avg_pct}%"
    )
    report = "\n".join(results)

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return report


def grade_answers_with_key(answer_key_file) -> str:
    """Grade using a teacher-supplied JSON answer key."""
    if not os.path.exists("student_answers.json"):
        return "❌ No student answers found."

    with open("student_answers.json", "r") as f:
        student = json.load(f)

    try:
        path = answer_key_file.name if hasattr(answer_key_file, "name") else answer_key_file
        with open(path, "r") as f:
            answer_key = json.load(f)
    except Exception as e:
        return f"❌ Invalid answer key file: {e}"

    results     = []
    total_marks = 0

    for qnum, data in student.items():
        student_ans = data.get("answer", "").strip()
        model_ans   = answer_key.get(qnum, {}).get("answer", "").strip()

        if not model_ans:
            results.append(f"{qnum}: ⚠️  No model answer provided")
            continue

        emb1  = similarity_model.encode(student_ans, convert_to_tensor=True)
        emb2  = similarity_model.encode(model_ans,   convert_to_tensor=True)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        pct   = round(score * 100, 1)

        if   score > 0.85: marks = 10
        elif score > 0.75: marks =  8
        elif score > 0.60: marks =  6
        elif score > 0.45: marks =  4
        elif score > 0.30: marks =  2
        else:              marks =  0

        verdict = (
            "✅ Correct"   if score > 0.75 else
            "🔶 Partial"   if score > 0.45 else
            "❌ Incorrect"
        )
        total_marks += marks
        results.append(
            f"{qnum}: {verdict} | Marks: {marks}/10  (Similarity: {pct}%)\n"
            f"  🎓 Student : {student_ans}\n"
            f"  📘 Expected: {model_ans}\n"
        )

    max_possible = len(student) * 10
    avg_pct      = round((total_marks / max_possible) * 100, 1) if max_possible else 0
    results.append(
        f"\n{'='*50}\n"
        f"Total Marks  : {total_marks} / {max_possible}\n"
        f"Overall Score: {avg_pct}%"
    )
    report = "\n".join(results)
    with open("grade_report.txt", "w") as f:
        f.write(report)
    return report


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Voice Exam Assistant") as app:
    gr.Markdown("# 🎙️ Voice Exam Assistant for Visually Impaired Students")

    questions_state = gr.State([])
    idx_state       = gr.State(0)

    # ── Part 1: Exam ─────────────────────────────────────────────────────────
    gr.Markdown("## 📄 Part 1: Take the Exam")

    with gr.Row():
        pdf_input = gr.File(label="Upload Exam PDF (Teacher)")
        load_btn  = gr.Button("Start Exam", variant="primary")

    progress_display = gr.Textbox(
        label="Exam Progress", interactive=False, value="Upload an exam to begin."
    )
    question_display = gr.Textbox(label="Current Question", interactive=False)
    audio_output     = gr.Audio(label="Listen to Question", autoplay=True)
    audio_input      = gr.Audio(
        sources=["microphone"], type="filepath", label="Speak Your Answer"
    )
    answer_display   = gr.Textbox(label="Your Transcribed Answer", interactive=False)

    # ── Part 2: Grading ───────────────────────────────────────────────────────
    gr.Markdown("---\n## 📊 Part 2: Grade the Exam (Teacher Only)")
    with gr.Row():
        grade_auto_btn = gr.Button("Grade with Auto Key", variant="secondary")
        answer_key_upload = gr.File(label="Or upload your own Answer Key (.json)")
        grade_manual_btn  = gr.Button("Grade with Uploaded Key", variant="secondary")

    grade_output = gr.Textbox(label="Grade Report", lines=15, interactive=False)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def load_exam(pdf):
        qs = extract_questions(pdf)
        student_answers.clear()

        # Remove stale answer files
        for fname in ("student_answers.json", "auto_answer_key.json"):
            if os.path.exists(fname):
                os.remove(fname)

        generate_answer_key(qs)

        if not qs or qs[0].startswith("Warning"):
            audio = speak("Warning: No questions found in the PDF.")
            return qs, 0, qs[0], audio, "No questions found."

        intro = f"Exam loaded. There are {len(qs)} questions. Please answer each question after it is read."
        speak(intro)                      # play intro (not returned to UI)
        audio = speak(qs[0])             # return first question audio
        progress = f"Question 1 of {len(qs)}"
        return qs, 0, qs[0], audio, progress


    def transcribe_and_advance(audio, questions, idx):
        if audio is None or not questions:
            return idx, questions[idx] if questions else "", None, "", f"Question {idx+1} of {len(questions)}"

        # Transcribe
        try:
            result      = stt(audio)
            transcribed = result["text"].strip()
        except Exception as e:
            transcribed = ""
            print(f"STT error: {e}")

        # Reject noise / too-short answers
        noise_words = {"you", "thank you", "thanks", "uh", "um", "hmm", ""}
        if len(transcribed) < 4 or transcribed.lower() in noise_words:
            audio_out = speak("I could not hear a proper answer. Please try again.")
            return idx, questions[idx], audio_out, "Try again — speak clearly.", \
                   f"Question {idx+1} of {len(questions)}"

        # Save answer
        if idx < len(questions):
            student_answers[f"Q{idx+1}"] = {
                "question": questions[idx],
                "answer":   transcribed
            }
            print(f"Saved Q{idx+1}: {transcribed}")

        # Advance
        next_idx = idx + 1
        if next_idx < len(questions):
            audio_out    = speak(questions[next_idx])
            progress     = f"Question {next_idx + 1} of {len(questions)}"
            return next_idx, questions[next_idx], audio_out, transcribed, progress
        else:
            with open("student_answers.json", "w") as f:
                json.dump(student_answers, f, indent=2)
            audio_out = speak("Exam complete. All your answers have been saved. Well done!")
            return next_idx, "✅ Exam complete! All answers saved.", audio_out, transcribed, "Done!"


    # ── Wire up ───────────────────────────────────────────────────────────────

    load_btn.click(
        load_exam,
        inputs=[pdf_input],
        outputs=[questions_state, idx_state, question_display, audio_output, progress_display]
    )

    audio_input.change(
        transcribe_and_advance,
        inputs=[audio_input, questions_state, idx_state],
        outputs=[idx_state, question_display, audio_output, answer_display, progress_display]
    )

    grade_auto_btn.click(
        grade_answers_auto,
        inputs=[],
        outputs=[grade_output]
    )

    grade_manual_btn.click(
        grade_answers_with_key,
        inputs=[answer_key_upload],
        outputs=[grade_output]
    )

app.launch(share=False)
