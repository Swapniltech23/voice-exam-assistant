# Voice Exam Assistant for Visually Impaired Students

A fully voice-controlled exam system built using Hugging Face models.

## Live Demo
https://huggingface.co/spaces/Swapnil12sahu/voice-exam-assistant

## Problem
Visually impaired students in India need a human scribe for exams — 
expensive, hard to scale, and introduces human bias. 
This tool removes that dependency entirely.

## Solution
Three Hugging Face models chained together into one accessible pipeline:

| Model | Purpose |
|-------|---------|
| `openai/whisper-small` | Converts student's spoken answer to text |
| `microsoft/speecht5_tts` | Reads exam questions aloud to student |
| `sentence-transformers/all-MiniLM-L6-v2` | Auto-grades answers by meaning |

## Features
- Upload any exam PDF — questions extracted automatically
- OCR support for scanned PDFs
- Questions read aloud using AI text-to-speech
- Student speaks answers — transcribed automatically by Whisper
- Auto-advances to next question after each answer
- Teacher uploads answer key JSON — AI grades by semantic similarity
- Exports full grade report

## How to Run Locally
pip install -r requirements.txt
python app.py

## How to Test
1. Upload a question paper PDF
2. Click Start Exam
3. Listen to each question
4. Press mic button and speak your answer
5. App saves and moves to next question automatically
6. After exam, upload answer_key.json and click Grade Exam

## Answer Key Format
{
  "Q1": {"answer": "New Delhi is the capital of India"},
  "Q2": {"answer": "Mahatma Gandhi is the father of the nation"}
}

## Built With
- Hugging Face Transformers
- Gradio
- PyMuPDF + Tesseract OCR
- Python
