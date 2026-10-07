from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import os
import json
import re
import time
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# You can change this from .env without changing Python code.
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_RESUME_LENGTH = 30000
MAX_JD_LENGTH = 25000
MAX_CHAT_CONTEXT = 18000

client = None

if OPENROUTER_API_KEY:
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )


# =========================================================
# FILE TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        return text.strip()

    except Exception as e:
        raise ValueError(f"Could not read PDF: {str(e)}")


def extract_text_from_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text.strip()

    except Exception as e:
        raise ValueError(f"Could not read image: {str(e)}")


def extract_uploaded_text(file):
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)

    if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return extract_text_from_image(file)

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore").strip()

    raise ValueError(
        "Unsupported file type. Please upload PDF, TXT, PNG, JPG, JPEG or WEBP."
    )


# =========================================================
# JSON HELPERS
# =========================================================

def clean_json_response(text):
    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    # Remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text.strip()


def parse_json_response(text):
    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)
    except Exception:
        return None


# =========================================================
# OPENROUTER
# =========================================================

def call_openrouter(
    prompt,
    max_tokens=3500,
    temperature=0.1
):
    if not client:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file."
        )

    last_error = None

    # Only a small number of retries.
    # This prevents the application from becoming unnecessarily slow.
    for attempt in range(2):

        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are ResumeMatch AI. "
                            "Return accurate, concise and structured answers. "
                            "Never invent user skills, projects, experience, "
                            "certifications, achievements or numbers."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "http://127.0.0.1:8000",
                    "X-Title": "ResumeMatch AI"
                }
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError("AI returned an empty response.")

            return content

        except Exception as e:
            last_error = e

            error_text = str(e).lower()

            temporary_error = any(
                keyword in error_text
                for keyword in [
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "rate limit",
                    "timeout",
                    "temporarily"
                ]
            )

            if temporary_error and attempt == 0:
                time.sleep(1)
                continue

            break
    print("OPENROUTER FINAL ERROR:", repr(last_error))
    raise RuntimeError(f"OpenRouter error: {str(last_error)}")


# =========================================================
# ANALYSIS PROMPT
# =========================================================

def build_analysis_prompt(resume, jd):

    return f"""
Analyze the following resume against the job description.

IMPORTANT RULES:

1. Use ONLY information present in the resume and job description.
2. Never invent skills, projects, experience, certifications, achievements,
   numbers, employers or technologies.
3. If a skill is missing from the resume, mark it as missing.
4. Do not tell the user to claim a skill they do not have.
5. Suggestions should be realistic for a fresher.
6. Keep arrays concise.
7. Maximum:
   - 12 matching skills
   - 12 missing skills
   - 10 related skills
   - 15 keywords
   - 15 requirements
   - 8 resume modifications
   - 8 projects
   - 8 action items
   - 10 keyword placement items
8. Match score must be based on evidence, not keyword stuffing.
9. ATS score is a formatting/readability estimate, not a guarantee.
10. Do not predict hiring probability.

Return ONLY valid JSON.

REQUIRED JSON:

{{
  "job_title": "",
  "overall_summary": "",

  "match_score": 0,

  "score_breakdown": {{
    "skills": 0,
    "keywords": 0,
    "projects": 0,
    "requirements": 0
  }},

  "ats_compatibility": {{
    "score": 0,
    "overall_status": "",
    "positive_checks": [],
    "warnings": [],
    "recommendations": []
  }},

  "top_actions": [
    {{
      "priority": 1,
      "action": "",
      "why": "",
      "type": "Resume"
    }}
  ],

  "skills_analysis": {{
    "matching": [],
    "missing": [],
    "related": [],
    "do_not_claim": []
  }},

  "skill_gap_dashboard": [
    {{
      "skill": "",
      "status": "Have",
      "importance": "High",
      "evidence": "",
      "recommended_action": ""
    }}
  ],

  "keyword_placement": [
    {{
      "keyword": "",
      "status": "Found",
      "current_location": "",
      "recommended_location": "",
      "reason": ""
    }}
  ],

  "important_keywords": [],

  "requirements": [
    {{
      "requirement": "",
      "status": "Found",
      "priority": "High",
      "evidence": ""
    }}
  ],

  "resume_modifications": [
    {{
      "section": "Summary",
      "action": "Modify",
      "current_content": "",
      "suggested_content": "",
      "reason": ""
    }}
  ],

  "project_relevance": [
    {{
      "project_name": "",
      "relevance": "High",
      "matching_requirements": [],
      "why_relevant": "",
      "suggested_change": ""
    }}
  ],

  "action_plan": [
    {{
      "priority": 1,
      "action": "",
      "why": "",
      "type": "Resume"
    }}
  ],

  "warnings": []
}}

ATS CHECKS SHOULD INCLUDE:

- Contact information
- Summary/profile
- Skills section
- Education
- Experience
- Projects
- Standard section headings
- Readability
- Tables/columns if detectable
- Excessive symbols
- Excessive graphics/images if detectable
- Missing important sections
- Long paragraphs

SKILL STATUS VALUES:

"Have"
"Missing"
"Related"
"Do Not Claim"

KEYWORD STATUS VALUES:

"Found"
"Missing"
"Found but weak"

RESUME:

-------------------------
{resume[:MAX_RESUME_LENGTH]}
-------------------------

JOB DESCRIPTION:

-------------------------
{jd[:MAX_JD_LENGTH]}
-------------------------
"""


# =========================================================
# DEFAULT ANALYSIS STRUCTURE
# =========================================================

def normalize_analysis(data):

    if not isinstance(data, dict):
        data = {}

    data.setdefault("job_title", "")
    data.setdefault("overall_summary", "")
    data.setdefault("match_score", 0)

    data.setdefault(
        "score_breakdown",
        {
            "skills": 0,
            "keywords": 0,
            "projects": 0,
            "requirements": 0
        }
    )

    data.setdefault(
        "ats_compatibility",
        {
            "score": 0,
            "overall_status": "",
            "positive_checks": [],
            "warnings": [],
            "recommendations": []
        }
    )

    data.setdefault("top_actions", [])

    data.setdefault(
        "skills_analysis",
        {
            "matching": [],
            "missing": [],
            "related": [],
            "do_not_claim": []
        }
    )

    data.setdefault("skill_gap_dashboard", [])
    data.setdefault("keyword_placement", [])
    data.setdefault("important_keywords", [])
    data.setdefault("requirements", [])
    data.setdefault("resume_modifications", [])
    data.setdefault("project_relevance", [])
    data.setdefault("action_plan", [])
    data.setdefault("warnings", [])

    return data


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ANALYZE
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        resume_text = request.form.get("resume", "").strip()
        jd_text = request.form.get("job_description", "").strip()

        # Resume upload
        resume_file = request.files.get("resume_file")

        if resume_file and resume_file.filename:
            uploaded_resume = extract_uploaded_text(resume_file)

            if uploaded_resume:
                resume_text = uploaded_resume

        # JD upload
        jd_file = request.files.get("jd_file")

        if jd_file and jd_file.filename:
            uploaded_jd = extract_uploaded_text(jd_file)

            if uploaded_jd:
                jd_text = uploaded_jd

        if not resume_text:
            return jsonify({
                "success": False,
                "error": "Please enter or upload your resume."
            }), 400

        if not jd_text:
            return jsonify({
                "success": False,
                "error": "Please enter or upload the job description."
            }), 400

        if len(resume_text) > MAX_RESUME_LENGTH:
            resume_text = resume_text[:MAX_RESUME_LENGTH]

        if len(jd_text) > MAX_JD_LENGTH:
            jd_text = jd_text[:MAX_JD_LENGTH]

        prompt = build_analysis_prompt(
            resume_text,
            jd_text
        )

        raw_response = call_openrouter(
            prompt,
            max_tokens=3500,
            temperature=0.1
        )

        analysis = parse_json_response(raw_response)

        if not analysis:

            # One retry using a simpler JSON instruction.
            retry_prompt = f"""
Convert the following AI output into valid JSON.

Return ONLY JSON.

Required top-level keys:
job_title,
overall_summary,
match_score,
score_breakdown,
ats_compatibility,
top_actions,
skills_analysis,
skill_gap_dashboard,
keyword_placement,
important_keywords,
requirements,
resume_modifications,
project_relevance,
action_plan,
warnings

AI OUTPUT:

{raw_response[:18000]}
"""

            repaired = call_openrouter(
                retry_prompt,
                max_tokens=3500,
                temperature=0
            )

            analysis = parse_json_response(repaired)

        if not analysis:
            return jsonify({
                "success": False,
                "error": "AI returned an invalid analysis. Please try again."
            }), 502

        analysis = normalize_analysis(analysis)

        return jsonify({
            "success": True,
            "resume": resume_text,
            "job_description": jd_text,
            "analysis": analysis
        })

    except Exception as e:

        print("ANALYZE ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# BULLET POINT IMPROVER
# =========================================================

@app.route("/improve-bullet", methods=["POST"])
def improve_bullet():

    try:

        data = request.get_json(silent=True) or {}

        bullet = data.get("bullet", "").strip()
        resume = data.get("resume", "").strip()
        job_description = data.get("job_description", "").strip()

        if not bullet:
            return jsonify({
                "success": False,
                "error": "Please enter a resume bullet point."
            }), 400

        prompt = f"""
Improve the following resume bullet point for the provided job description.

RULES:

1. Do not invent achievements.
2. Do not invent metrics.
3. Do not invent technologies.
4. Do not add experience that is not present.
5. Keep the original meaning.
6. Make it concise and professional.
7. Use relevant JD terminology only when truthful.
8. Prefer strong action verbs.
9. Return ONLY valid JSON.

Return:

{{
  "original": "",
  "improved": "",
  "why": "",
  "keywords_used": [],
  "warning": ""
}}

RESUME CONTEXT:
{resume[:10000]}

JOB DESCRIPTION:
{job_description[:10000]}

BULLET TO IMPROVE:
{bullet}
"""

        raw = call_openrouter(
            prompt,
            max_tokens=900,
            temperature=0.1
        )

        result = parse_json_response(raw)

        if not result:
            return jsonify({
                "success": False,
                "error": "Could not parse the AI response."
            }), 502

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:

        print("BULLET ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# AI CHAT ASSISTANT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(silent=True) or {}

        message = data.get("message", "").strip()
        resume = data.get("resume", "").strip()
        job_description = data.get("job_description", "").strip()
        analysis = data.get("analysis", {})

        if not message:
            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        analysis_text = json.dumps(
            analysis,
            ensure_ascii=False
        )

        prompt = f"""
You are ResumeMatch AI Assistant.

Answer the user's question using ONLY the resume,
job description and analysis below.

RULES:

1. Never invent skills.
2. Never invent experience.
3. Never invent projects.
4. Never invent certifications.
5. Never invent achievements or numbers.
6. If the user does not have a required skill, say it is missing.
7. Never advise the user to falsely claim a skill.
8. You may create learning plans for missing skills.
9. You may create 7-day, 30-day and 60-day plans.
10. You may create interview preparation based on the JD and resume.
11. You may suggest project improvements.
12. Do not predict whether the user will get hired.
13. Keep answers practical and beginner-friendly.
14. Use headings and bullet points when useful.

USER QUESTION:
{message}

RESUME:
{resume[:MAX_CHAT_CONTEXT]}

JOB DESCRIPTION:
{job_description[:MAX_CHAT_CONTEXT]}

LATEST ANALYSIS:
{analysis_text[:MAX_CHAT_CONTEXT]}
"""

        answer = call_openrouter(
            prompt,
            max_tokens=2200,
            temperature=0.2
        )

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:

        print("CHAT ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "model": OPENROUTER_MODEL
    })


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    print("")
    print("=" * 55)
    print("ResumeMatch AI")
    print("=" * 55)
    print(f"Model: {OPENROUTER_MODEL}")
    print("Server: http://127.0.0.1:8000")
    print("=" * 55)
    print("")

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )
