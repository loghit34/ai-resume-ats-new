import os
import json
import logging
from typing import Dict, List, Optional
import google.generativeai as genai

logger = logging.getLogger('ats_resume_scorer')

GEMINI_MODEL = 'gemini-1.5-flash'
_initialized = False

def _init_client():
    global _initialized
    if not _initialized:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        _initialized = True

def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    _init_client()
    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt
        )
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return response.text.strip()
    except Exception as exc:
        logger.error(f"Gemini API call failed: {exc}")
        # Try without response_mime_type if it failed due to configuration
        try:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt)
            return response.text.strip()
        except Exception as retry_exc:
            logger.error(f"Gemini API retry call failed: {retry_exc}")
            raise retry_exc

def _try_parse_json(text: str) -> dict | list | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

RESUME_SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume "
    "and return ONLY a valid JSON object. Do not include any explanations."
)

RESUME_USER_PROMPT = """Extract the following from this resume and return as JSON:
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "the full text of the Summary, Profile, About Me, Objective, or Professional Summary section at the top of the resume. Copy the ENTIRE paragraph exactly as written. If no such section exists, return an empty string.",
  "skills": ["list", "of", "skills"],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {{
      "title": "project name",
      "description": "what the project does and how it was built",
      "technologies": ["tech", "used"]
    }}
  ],
  "action_verbs": ["strong action verbs used in bullet points, e.g. developed, implemented, designed"],
  "keywords": ["important keywords and phrases from the resume for ATS matching"]
}}

Important instructions:
- For duration_months, calculate the number of months between start_date and end_date. If end_date is "Present" or "Current", calculate from start_date to now.
- For skills, extract ALL technical and soft skills mentioned anywhere in the resume.
- For action_verbs, find verbs that start bullet points or describe achievements.
- For keywords, extract noun phrases and technical terms relevant to ATS matching.
- Return ONLY valid JSON.

Resume Text:
{raw_text}"""

def parse_resume(raw_text: str) -> Dict:
    prompt = RESUME_USER_PROMPT.format(raw_text=raw_text)
    raw_response = _call_gemini(RESUME_SYSTEM_PROMPT, prompt)
    result = _try_parse_json(raw_response)
    if result is not None and isinstance(result, dict):
        return result
    raise ValueError(f"Gemini returned invalid or empty JSON response for resume: {raw_response[:200]}")

JD_SYSTEM_PROMPT = (
    "You are a job description parser. Extract information and "
    "return ONLY a valid JSON object. Do not include any explanations."
)

JD_USER_PROMPT = """Extract the following from this job description and return as JSON:
{{
  "job_title": "",
  "required_skills": ["list of must-have skills"],
  "preferred_skills": ["list of nice-to-have skills"],
  "experience_required": "",
  "education_required": "",
  "key_responsibilities": ["list of responsibilities"],
  "keywords": ["important keywords and phrases for ATS matching"]
}}

Job Description Text:
{raw_text}"""

def parse_job_description(raw_text: str) -> Dict:
    prompt = JD_USER_PROMPT.format(raw_text=raw_text)
    raw_response = _call_gemini(JD_SYSTEM_PROMPT, prompt)
    result = _try_parse_json(raw_response)
    if result is not None and isinstance(result, dict):
        return result
    raise ValueError(f"Gemini returned invalid or empty JSON response for job description: {raw_response[:200]}")

SIMILARITY_SYSTEM_PROMPT = (
    "You are an expert ATS screening system. Assess semantic similarity between a candidate's resume and job description. "
    "Return a JSON object containing a similarity score from 0.0 to 1.0 (where 1.0 is a perfect match and 0.0 is no match)."
)

SIMILARITY_USER_PROMPT = """Given the resume text and the job description text below, determine their semantic similarity score (0.0 to 1.0).
Consider skills, experience relevance, role alignment, and educational requirements.
Return ONLY a JSON object like: {{"similarity": 0.85}}

Resume:
{resume_text}

Job Description:
{jd_text}"""

def get_semantic_similarity(resume_text: str, jd_text: str) -> float:
    prompt = SIMILARITY_USER_PROMPT.format(resume_text=resume_text[:4000], jd_text=jd_text[:4000])
    try:
        raw_response = _call_gemini(SIMILARITY_SYSTEM_PROMPT, prompt)
        result = _try_parse_json(raw_response)
        if result is not None and isinstance(result, dict) and "similarity" in result:
            return float(result["similarity"])
    except Exception as exc:
        logger.warning(f"Failed to calculate similarity with Gemini: {exc}")
    return 0.5

SKILLS_GAP_SYSTEM_PROMPT = (
    "You are a talent acquisition specialist. Review the candidate's skills against a job description "
    "and identify gaps. Return ONLY a JSON list of missing skills."
)

SKILLS_GAP_USER_PROMPT = """Given the candidate's listed skills and the job description, identify up to 20 important skills, frameworks, or tools that are required or preferred in the Job Description but are missing or not clearly demonstrated in the candidate's skills.
Return ONLY a JSON list like: ["Docker", "Kubernetes", "AWS"]

Candidate Skills:
{resume_skills}

Job Description:
{jd_text}"""

def get_skills_gap(resume_skills: List[str], jd_text: str) -> List[str]:
    prompt = SKILLS_GAP_USER_PROMPT.format(
        resume_skills=", ".join(resume_skills),
        jd_text=jd_text[:4000]
    )
    try:
        raw_response = _call_gemini(SKILLS_GAP_SYSTEM_PROMPT, prompt)
        result = _try_parse_json(raw_response)
        if result is not None and isinstance(result, list):
            return [str(item) for item in result]
    except Exception as exc:
        logger.warning(f"Failed to get skills gap from Gemini: {exc}")
    return []

SKILLS_VALIDATION_SYSTEM_PROMPT = (
    "You are an ATS skills validator. Verify if the candidate's skills are supported by evidence "
    "in their projects or experience details. Return ONLY a JSON object."
)

SKILLS_VALIDATION_USER_PROMPT = """You are given a list of candidate skills, their projects, and experience descriptions.
For each skill, determine if there is supporting evidence in either the projects or experience description.
A skill is validated if it or a closely related synonym is mentioned in the projects or experience descriptions.

Return ONLY a JSON object with this exact format:
{{
  "validated_skills": [
     {{"skill": "Python", "projects": ["Project Name A", "Experience Section"], "similarity": 1.0}}
  ],
  "unvalidated_skills": ["CSS", "Docker"]
}}

Candidate Skills:
{skills}

Projects:
{projects}

Experience Descriptions:
{experience}"""

def validate_skills_with_projects(skills: List[str], projects: List[Dict], experience_entries: List[Dict]) -> Dict:
    if not skills:
        return {
            'validated_skills':      [],
            'unvalidated_skills':    [],
            'validation_percentage': 0.0,
            'skill_project_mapping': {},
            'validation_score':      0.0,
        }

    projects_summary = "\n".join([f"- Project: {p.get('title', '')}. Description: {p.get('description', '')}" for p in projects])
    experience_summary = "\n".join([f"- Experience at {e.get('company', '')} as {e.get('job_title', '')}: {e.get('description', '')}" for e in experience_entries])

    prompt = SKILLS_VALIDATION_USER_PROMPT.format(
        skills=", ".join(skills),
        projects=projects_summary,
        experience=experience_summary
    )

    try:
        raw_response = _call_gemini(SKILLS_VALIDATION_SYSTEM_PROMPT, prompt)
        result = _try_parse_json(raw_response)
        if result is not None and isinstance(result, dict):
            validated_skills = result.get('validated_skills', [])
            unvalidated_skills = result.get('unvalidated_skills', [])
            
            # Format and calculate scores
            total = len(skills)
            val_count = len(validated_skills)
            val_pct = val_count / total if total > 0 else 0.0
            val_score = val_pct * 15.0
            
            skill_project_mapping = {}
            for item in validated_skills:
                skill_project_mapping[item.get('skill', '')] = item.get('projects', [])
            for skill in unvalidated_skills:
                skill_project_mapping[skill] = []

            return {
                'validated_skills':      validated_skills,
                'unvalidated_skills':    unvalidated_skills,
                'validation_percentage': val_pct,
                'skill_project_mapping': skill_project_mapping,
                'validation_score':      val_score,
            }
    except Exception as exc:
        logger.warning(f"Failed to validate skills with Gemini: {exc}")

    # Fallback to local keyword search
    validated_skills = []
    unvalidated_skills = []
    skill_project_mapping = {}
    
    experience_text = ' '.join(
        f"{e.get('job_title', '')} {e.get('company', '')} {e.get('description', '')}"
        for e in experience_entries if isinstance(e, dict)
    ).lower()

    for skill in skills:
        matched_projects = []
        skill_lower = skill.lower()
        
        for project in projects:
            project_text = f"{project.get('title', '')} {project.get('description', '')}".lower()
            if skill_lower in project_text:
                matched_projects.append(project.get('title', 'Untitled Project'))
                
        if skill_lower in experience_text:
            matched_projects.append('Experience Section')
            
        if matched_projects:
            validated_skills.append({'skill': skill, 'projects': matched_projects, 'similarity': 1.0})
            skill_project_mapping[skill] = matched_projects
        else:
            unvalidated_skills.append(skill)
            skill_project_mapping[skill] = []
            
    total = len(skills)
    val_pct = len(validated_skills) / total if total > 0 else 0.0
    val_score = val_pct * 15.0

    return {
        'validated_skills':      validated_skills,
        'unvalidated_skills':    unvalidated_skills,
        'validation_percentage': val_pct,
        'skill_project_mapping': skill_project_mapping,
        'validation_score':      val_score,
    }
