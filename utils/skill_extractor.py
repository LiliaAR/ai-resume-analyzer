"""
Skill extraction and categorization module
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json
import os

class SkillExtractor:
    """Extract and categorize skills from resume text"""
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def extract_skills(self, resume_text):
        """Extract skills and categorize them"""
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert at extracting and categorizing skills from resumes.

Analyze this resume and extract ALL skills, then categorize them.

Resume:
{resume_text}

Return a JSON object with this exact structure:
{{
  "technical_skills": {{
    "programming_languages": ["list of languages"],
    "frameworks_libraries": ["list of frameworks"],
    "databases": ["list of databases"],
    "cloud_platforms": ["AWS, Azure, etc"],
    "ai_ml": ["AI/ML specific tools"],
    "devops": ["Docker, Kubernetes, etc"],
    "other_technical": ["any other technical skills"]
  }},
  "soft_skills": [
    "list of soft skills like leadership, communication, etc"
  ],
  "domain_knowledge": [
    "industry-specific knowledge"
  ],
  "certifications": [
    "any certifications mentioned"
  ]
}}

Return ONLY the JSON object, no other text.
""")
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(resume_text=resume_text)
        
        try:
            skills_data = json.loads(result)
            return skills_data
        except json.JSONDecodeError:
            return {
                "technical_skills": {},
                "soft_skills": [],
                "domain_knowledge": [],
                "certifications": []
            }
    
    def suggest_missing_skills(self, current_skills, target_role="AI/ML Engineer"):
        """Suggest skills to add based on target role"""
        
        prompt = ChatPromptTemplate.from_template("""
You are a career coach helping someone target a {target_role} role.

They currently have these skills:
{current_skills}

Suggest 10 high-value skills they should add to be competitive for {target_role} roles in 2025.

Return a JSON array of objects with this structure:
[
  {{
    "skill": "Skill name",
    "priority": "High/Medium/Low",
    "reason": "Why this skill matters"
  }}
]

Return ONLY the JSON array, no other text.
""")
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(
            current_skills=json.dumps(current_skills, indent=2),
            target_role=target_role
        )
        
        try:
            suggestions = json.loads(result)
            return suggestions
        except json.JSONDecodeError:
            return []
    
    def match_to_job_description(self, resume_skills, job_description):
        """Match resume skills to job description"""
        
        prompt = ChatPromptTemplate.from_template("""
Compare the candidate's skills to the job requirements.

Candidate Skills:
{resume_skills}

Job Description:
{job_description}

Analyze the match and return JSON:
{{
  "match_percentage": 85,
  "matching_skills": ["skills that match"],
  "missing_critical_skills": ["required skills they lack"],
  "nice_to_have_missing": ["preferred skills they lack"],
  "recommendations": [
    "specific advice on how to bridge gaps"
  ]
}}

Return ONLY the JSON object, no other text.
""")
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(
            resume_skills=json.dumps(resume_skills, indent=2),
            job_description=job_description
        )
        
        try:
            match_data = json.loads(result)
            return match_data
        except json.JSONDecodeError:
            return {
                "match_percentage": 0,
                "matching_skills": [],
                "missing_critical_skills": [],
                "nice_to_have_missing": [],
                "recommendations": []
            }
