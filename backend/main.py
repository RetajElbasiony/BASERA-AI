import json
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CAMERA_OUTPUT_FILE = BASE_DIR / "camera" / "camera_output.json"

SPEECH_DIR = BASE_DIR / "speech"
SPEECH_DIR.mkdir(exist_ok=True)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="BASERA API",
    description="AI Accessibility Assistant",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/speech-files",
    StaticFiles(directory=SPEECH_DIR),
    name="speech-files"
)


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    message: str


class StudyRequest(BaseModel):
    message: str
    student_type: str = "general"
    subject: str = "general"


class AgentRequest(BaseModel):
    input_type: str = "text"
    text: str
    student_type: str = "general"
    subject: str = "general"


class OCRRequest(BaseModel):
    text: str
    student_type: str = "blind"
    subject: str = "general"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "BASERA backend is running.",
        "docs": "/docs"
    }


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    prompt = f"""
You are BASERA, an AI educational accessibility assistant.

User message:
{request.message}

Answer the user clearly and naturally.

Rules:
- Use simple language.
- Answer directly.
- If the user asks about programming, explain technical terms clearly.
- Keep important English technical terms when useful.
- Do not invent information.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "success": True,
            "response": response.text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {str(e)}"
        )


# =========================================================
# STUDY
# =========================================================

@app.post("/study")
def study(request: StudyRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    prompt = f"""
You are BASERA, an AI educational accessibility assistant.

Student type:
{request.student_type}

Subject:
{request.subject}

Student question:
{request.message}

Your job is to help the student understand the educational content.

Instructions:
- Explain clearly and simply.
- Adapt the explanation to the student's needs.
- Use Arabic naturally when appropriate.
- Keep important English technical terms.
- Give examples when useful.
- Do not invent information.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "success": True,
            "student_type": request.student_type,
            "subject": request.subject,
            "response": response.text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {str(e)}"
        )


# =========================================================
# CAMERA STUDY
# =========================================================

@app.get("/camera-study")
def camera_study():

    if not CAMERA_OUTPUT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="camera_output.json not found."
        )

    try:
        with open(
            CAMERA_OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            camera_data = json.load(f)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read camera output: {str(e)}"
        )


    # IMPORTANT:
    # Camera module uses "mode"
    # Older backend used "input_type"
    # We support both.

    input_type = (
        camera_data.get("input_type")
        or camera_data.get("mode")
        or ""
    )

    text = camera_data.get(
        "text",
        ""
    )

    objects = camera_data.get(
        "objects",
        []
    )

    sentence_complete = camera_data.get(
        "sentence_complete",
        False
    )


    # =====================================================
    # BUILD CAMERA INFORMATION
    # =====================================================

    camera_information = f"""
Input type:
{input_type}

Text:
{text}

Detected objects:
{json.dumps(objects, ensure_ascii=False)}

Sentence complete:
{sentence_complete}
"""


    prompt = f"""
You are BASERA, an AI accessibility assistant.

A camera/computer-vision module provided the following information:

{camera_information}

Your task:

1. Understand the information from the camera.
2. If the input contains a sign-language sentence, understand it and respond appropriately.
3. If objects were detected, explain what is relevant.
4. If text was detected, explain or answer it.
5. If the information is incomplete, do not invent missing information.
6. Use simple natural Arabic.
7. Keep important English technical terms when useful.
8. Give a useful educational response when possible.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "success": True,
            "input_type": input_type,
            "text": text,
            "objects": objects,
            "sentence_complete": sentence_complete,
            "response": response.text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {str(e)}"
        )


# =========================================================
# OCR
# =========================================================

@app.post("/ocr")
def ocr(request: OCRRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="OCR text is empty."
        )


    prompt = f"""
You are BASERA, an AI educational accessibility assistant.

The following text was extracted from an image using OCR.

OCR text:
{request.text}

Student type:
{request.student_type}

Subject:
{request.subject}

Your task:

1. Understand the extracted text.
2. If it is a question, answer it.
3. If it is educational content, explain it.
4. If it is an instruction, explain what the student needs to do.
5. Use simple natural Arabic.
6. Keep important English technical terms when useful.
7. If the OCR text is unclear or incomplete, clearly say that.
8. Do not invent missing text.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "success": True,
            "input_type": "ocr",
            "ocr_text": request.text,
            "student_type": request.student_type,
            "subject": request.subject,
            "response": response.text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini OCR error: {str(e)}"
        )


# =========================================================
# AGENT
# =========================================================

@app.post("/agent")
def agent(request: AgentRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Agent input cannot be empty."
        )


    prompt = f"""
You are BASERA, an AI educational accessibility agent.

Input type:
{request.input_type}

Student type:
{request.student_type}

Subject:
{request.subject}

User input:
{request.text}

Respond as an educational accessibility assistant.

Rules:
- Understand the user's intent.
- Give a direct and useful answer.
- Use simple natural Arabic when appropriate.
- Keep important English technical terms.
- Adapt the answer to the student's needs.
- Do not invent information.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return {
            "success": True,
            "input_type": request.input_type,
            "student_type": request.student_type,
            "subject": request.subject,
            "response": response.text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini agent error: {str(e)}"
        )


# =========================================================
# SPEECH AGENT
# =========================================================

@app.post("/speech-agent")
async def speech_agent(
    file: UploadFile = File(...)
):

    temp_filename = (
        SPEECH_DIR /
        f"input_{uuid.uuid4().hex}.wav"
    )


    try:

        # -------------------------------------------------
        # SAVE TEMPORARY AUDIO FILE
        # -------------------------------------------------

        with open(
            temp_filename,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # -------------------------------------------------
        # IMPORT SPEECH MODULE
        # -------------------------------------------------

        from speech.speech import (
            speech_to_text,
            text_to_speech
        )


        # -------------------------------------------------
        # SPEECH TO TEXT
        # -------------------------------------------------

        text, language = speech_to_text(
            str(temp_filename)
        )


        if not text.strip():

            return {
                "success": True,
                "transcript": "",
                "language": language,
                "response": "لم أتمكن من سماع كلام واضح.",
                "audio_url": None
            }


        # -------------------------------------------------
        # GEMINI
        # -------------------------------------------------

        prompt = f"""
You are BASERA, an AI educational accessibility assistant.

The student spoke the following:

{text}

Respond clearly and naturally.

Rules:
- Use Arabic when appropriate.
- Keep important English technical terms.
- Give a useful educational answer.
- Do not invent information.
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        answer = response.text


        # -------------------------------------------------
        # TEXT TO SPEECH
        # -------------------------------------------------

        output_filename = (
            f"response_{uuid.uuid4().hex}.mp3"
        )

        output_path = (
            SPEECH_DIR /
            output_filename
        )

        text_to_speech(
            answer,
            lang="ar",
            output_file=str(output_path)
        )


        # -------------------------------------------------
        # DELETE TEMP INPUT AUDIO
        # -------------------------------------------------

        if temp_filename.exists():
            temp_filename.unlink()


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {
            "success": True,
            "transcript": text,
            "language": language,
            "response": answer,
            "audio_url": f"/speech-files/{output_filename}"
        }


    except Exception as e:

        # Delete temporary audio if an error happens

        if temp_filename.exists():
            try:
                temp_filename.unlink()
            except Exception:
                pass


        raise HTTPException(
            status_code=500,
            detail=f"Speech agent error: {str(e)}"
        )
        # test edit by Mohammed
