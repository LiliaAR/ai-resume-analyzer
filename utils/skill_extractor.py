"""
Skill extraction and categorization module
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Validate input
        if not resume_text or not resume_text.strip():
            logger.warning("Empty resume text provided")
            return self._empty_skills_response()
        
        if len(resume_text.strip()) < 50:
            logger.warning("Resume text too short (less than 50 characters)")
            return self._empty_skills_response()
        
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
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(resume_text=resume_text)
            
            # Try to parse JSON
            skills_data = json.loads(result)
            logger.info("Successfully extracted skills")
            return skills_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Raw LLM response: {result[:200]}...")
            return self._empty_skills_response()
            
        except Exception as e:
            logger.error(f"Skill extraction failed: {type(e).__name__} - {e}")
            return self._empty_skills_response()
    
    def suggest_missing_skills(self, current_skills, target_role="AI/ML Engineer"):
        """Suggest skills to add based on target role"""
        
        # Validate input
        if not current_skills:
            logger.warning("No current skills provided for suggestions")
            return []
        
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
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(
                current_skills=json.dumps(current_skills, indent=2),
                target_role=target_role
            )
            
            suggestions = json.loads(result)
            logger.info(f"Generated {len(suggestions)} skill suggestions for {target_role}")
            return suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for skill suggestions: {e}")
            return []
            
        except Exception as e:
            logger.error(f"Skill suggestion failed: {type(e).__name__} - {e}")
            return []
    
    def match_to_job_description(self, resume_skills, job_description):
        """Match resume skills to job description"""
        
        # Validate inputs
        if not resume_skills:
            logger.warning("No resume skills provided for matching")
            return self._empty_match_response()
        
        if not job_description or not job_description.strip():
            logger.warning("Empty job description provided")
            return self._empty_match_response()
        
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
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(
                resume_skills=json.dumps(resume_skills, indent=2),
                job_description=job_description
            )
            
            match_data = json.loads(result)
            logger.info(f"Job match calculated: {match_data.get('match_percentage', 0)}%")
            return match_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for job match: {e}")
            return self._empty_match_response()
            
        except Exception as e:
            logger.error(f"Job matching failed: {type(e).__name__} - {e}")
            return self._empty_match_response()
    
    def _empty_skills_response(self):
        """Return empty skills structure"""
        return {
            "technical_skills": {},
            "soft_skills": [],
            "domain_knowledge": [],
            "certifications": []
        }
    
    def _empty_match_response(self):
        """Return empty match structure"""
        return {
            "match_percentage": 0,
            "matching_skills": [],
            "missing_critical_skills": [],
            "nice_to_have_missing": [],
            "recommendations": []
        }
