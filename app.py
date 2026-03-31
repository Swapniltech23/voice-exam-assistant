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

# IMPORTANT FOR WINDOWS
pytesseract.pytesseract.tesseract_cmd = r'D:\exam\Tesseract-OCR\tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r'D:\exam\Tesseract-OCR\tessdata'

print("Loading heavy AI models... This might take a minute...")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

speaker_embeddings = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)[7306]["xvector"]
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

student_answers = {}

def extract_questions(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        page_text = page.get_text().strip()
        if len(page_text) < 10:
            print("Blank page detected. Activating OCR vision...")
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"

    print("\n--- EXTRACTED EXAM TEXT ---")
    print(text)
    print("---------------------------\n")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    questions = []
    current_q = ""

    for l in lines:
        is_start_of_question = re.match(
            r'^(Question\s*\d+|Q\d+|0\.\d+|\d+[\.\)]|[a-zA-Z][\)\.])', l, re.IGNORECASE
        ) or l.startswith(("What", "Define", "Explain", "Compare", "Examine", "How"))

        if is_start_of_question:
            if current_q:
                questions.append(current_q)
            current_q = l
        else:
            if current_q:
                current_q += " " + l
            else:
                current_q = l

    if current_q:
        questions.append(current_q)

    if len(questions) == 0:
        questions = ["Warning: No questions detected. Please check the PDF format."]

    return questions

def clean_text_for_speech(text):
    text = re.sub(r'(\d+\.?\d*)([a-zA-Z]+)', r'\1 \2', text)
    text = re.sub(r'\bhr\b', 'hours', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmin\b', 'minutes', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkg\b', 'kilograms', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcm\b', 'centimeters', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    text = text.replace("-", " ")
    return text

def speak(text):
    text = clean_text_for_speech(text)
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=vocoder
        )
    sf.write("output.wav", speech.numpy(), samplerate=16000)
    return "output.wav"

def grade_answers(answer_key_file):
    if not os.path.exists("student_answers.json"):
        return "No student answers found. Complete the exam first."
    with open("student_answers.json", "r") as f:
        student = json.load(f)
    try:
        with open(answer_key_file.name, "r") as f:
            answer_key = json.load(f)
    except:
        return "Invalid answer key file. Must be a JSON file."

    results = []
    total_score = 0

    for qnum, data in student.items():
        student_ans = data["answer"]
        model_ans = answer_key.get(qnum, {}).get("answer", "")
        if not model_ans:
            results.append(f"{qnum}: No model answer provided")
            continue
        emb1 = similarity_model.encode(student_ans, convert_to_tensor=True)
        emb2 = similarity_model.encode(model_ans, convert_to_tensor=True)
        score = float(util.cos_sim(emb1, emb2)[0][0])
        percentage = round(score * 100, 1)
        verdict = "Correct" if score > 0.75 else "Partial" if score > 0.45 else "Incorrect"
        total_score += score
        results.append(
            f"{qnum}: {verdict} ({percentage}%)\n  Student: {student_ans}\n  Expected: {model_ans}\n"
        )

    avg = round((total_score / len(student)) * 100, 1) if student else 0
    results.append(f"\nOverall Score: {avg}%")
    report = "\n".join(results)
    with open("grade_report.txt", "w") as f:
        f.write(report)
    return report

# ---------------- GRADIO UI ----------------
with gr.Blocks() as app:
    gr.Markdown("# Voice Exam Assistant for Visually Impaired Students")

    questions_state = gr.State([])
    idx_state = gr.State(0)

    # --- SECTION 1: EXAM TAKING ---
    gr.Markdown("### Part 1: Take the Exam")

    with gr.Row():
        pdf_input = gr.File(label="1. Upload Exam PDF (Teacher)")
        load_btn = gr.Button("2. Start Exam")

    progress_display = gr.Textbox(label="Exam Progress", interactive=False, value="Upload an exam to begin.")
    question_display = gr.Textbox(label="Current Question", interactive=False)
    audio_output = gr.Audio(label="Listen to Question", autoplay=True)
    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Your Answer")
    answer_display = gr.Textbox(label="Your Transcribed Answer", interactive=False)

    # Hidden — no longer needed by student
    next_btn = gr.Button("Save Answer & Next Question", variant="primary", visible=False)

    gr.Markdown("---")

    # --- SECTION 2: GRADING ---
    gr.Markdown("### Part 2: Grade the Exam (Teacher Only)")
    answer_key_input = gr.File(label="Upload Answer Key (JSON)")
    grade_btn = gr.Button("Grade Exam", variant="secondary")
    grade_output = gr.Textbox(label="Final Grade Report", lines=10, interactive=False)

    # --- LOGIC ---
    def load_exam(pdf):
        qs = extract_questions(pdf)
        student_answers.clear()
        audio = speak(
            "Exam loaded. There are " + str(len(qs)) +
            " questions. After each question is read, please speak your answer clearly."
        )
        progress_text = f"Question 1 of {len(qs)}" if qs else "No questions found."
        # Play intro announcement first, then immediately read Q1
        speak(qs[0]) if qs else None
        audio = speak(qs[0]) if qs else None
        return qs, 0, qs[0] if qs else "No questions found", audio, progress_text

    def transcribe_and_advance(audio, questions, idx):
        if audio is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # Step 1 — transcribe
        try:
            result = stt(audio)
            transcribed = result["text"]
        except Exception as e:
            transcribed = f"Error: {str(e)}"

        # Step 2 — filter out noise / too short answers
        if len(transcribed.strip().split()) < 2:
            audio_out = speak("Sorry, I didn't catch that. Please answer again.")
            return idx, questions[idx], audio_out, "Didn't catch that. Please try again.", f"Question {idx+1} of {len(questions)}"

        # Step 3 — save answer
        if idx < len(questions):
            student_answers[f"Q{idx+1}"] = {
                "question": questions[idx],
                "answer": transcribed
            }
            print(f"Saved: Q{idx+1} -> {transcribed}")

        # Step 4 — auto advance
        next_idx = idx + 1
        if next_idx < len(questions):
            audio_out = speak(questions[next_idx])
            progress_text = f"Question {next_idx + 1} of {len(questions)}"
            return next_idx, questions[next_idx], audio_out, transcribed, progress_text
        else:
            with open("student_answers.json", "w") as f:
                json.dump(student_answers, f, indent=2)
            audio_out = speak("Exam complete. All your answers have been saved. Well done.")
            return next_idx, "Exam complete! All answers saved.", audio_out, transcribed, "Done!"

    # --- CONNECTIONS ---
    load_btn.click(
        load_exam,
        inputs=[pdf_input],
        outputs=[questions_state, idx_state, question_display, audio_output, progress_display]
    )

    # KEY CHANGE: auto-advance on transcription — no button click needed
    audio_input.change(
        transcribe_and_advance,
        inputs=[audio_input, questions_state, idx_state],
        outputs=[idx_state, question_display, audio_output, answer_display, progress_display]
    )

    grade_btn.click(
        grade_answers,
        inputs=[answer_key_input],
        outputs=[grade_output]
    )

app.launch()