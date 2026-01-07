import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import PyPDF2
import os
from dotenv import load_dotenv
import json

# Import skill extractor
from utils.skill_extractor import SkillExtractor

load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

@st.cache_resource
def get_skill_extractor():
    return SkillExtractor(llm=get_llm())

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_resume(resume_text, llm):
    """Analyze resume using GPT-4o"""
    
    prompt_template = ChatPromptTemplate.from_template("""
You are an expert resume analyst and career coach. Analyze the following resume and provide detailed feedback.

Resume:
{resume_text}

Provide a comprehensive analysis in the following format:

## Overall Assessment
[Brief overview of the resume's strengths and weaknesses]

## Key Strengths
- [List 3-5 strengths]

## Areas for Improvement
- [List 3-5 specific improvements]

## ATS Optimization Tips
- [List 3-5 specific tips for passing ATS systems]

## Interview Preparation
Based on this resume, you might be asked:
1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]

## Action Items
[List 5 specific, actionable next steps to improve this resume]
""")
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(resume_text=resume_text)
    return result

# Main UI
st.title("📄 AI Resume Analyzer")
st.markdown("**Upload your resume and get AI-powered insights to land your dream job**")

st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses **GPT-4o** to analyze your resume and provide:
    - Comprehensive resume analysis
    - Skills extraction & categorization
    - ATS optimization tips
    - Skill gap analysis
    - Interview prep questions
    """)
    
    st.divider()
    
    st.markdown("**Built by Lilia Allen Rowland**")
    st.markdown("[GitHub](https://github.com/LiliaAR) | [LinkedIn](https://www.linkedin.com/in/liliaallenrowland/)")

# Main content
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)
    
    st.success(f"✅ Extracted {len(resume_text)} characters from your resume")
    
    # Show preview
    with st.expander("📄 View Extracted Text"):
        st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Full Analysis", "🎯 Skills Analysis", "💼 Job Match"])
    
    with tab1:
        if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
            llm = get_llm()
            
            with st.spinner("AI is analyzing your resume... This may take 30-60 seconds..."):
                analysis = analyze_resume(resume_text, llm)
            
            st.markdown("## 📊 Analysis Results")
            st.markdown(analysis)
            
            # Download button
            st.download_button(
                label="📥 Download Analysis",
                data=analysis,
                file_name="resume_analysis.txt",
                mime="text/plain"
            )
    
    with tab2:
        st.markdown("### 🎯 Skill Extraction & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Extract Skills", use_container_width=True):
                skill_extractor = get_skill_extractor()
                
                with st.spinner("Extracting and categorizing skills..."):
                    skills = skill_extractor.extract_skills(resume_text)
                
                # Store in session state
                st.session_state['skills'] = skills
        
        with col2:
            target_role = st.selectbox(
                "Target Role",
                ["AI/ML Engineer", "Data Scientist", "Software Engineer", 
                 "DevOps Engineer", "Product Manager"]
            )
        
        # Display skills if extracted
        if 'skills' in st.session_state:
            skills = st.session_state['skills']
            
            st.markdown("---")
            st.markdown("#### 🛠️ Technical Skills")
            
            tech_skills = skills.get('technical_skills', {})
            
            if tech_skills.get('programming_languages'):
                st.markdown("**Programming Languages:**")
                st.write(", ".join(tech_skills['programming_languages']))
            
            if tech_skills.get('frameworks_libraries'):
                st.markdown("**Frameworks & Libraries:**")
                st.write(", ".join(tech_skills['frameworks_libraries']))
            
            if tech_skills.get('ai_ml'):
                st.markdown("**AI/ML Tools:**")
                st.write(", ".join(tech_skills['ai_ml']))
            
            if tech_skills.get('cloud_platforms'):
                st.markdown("**Cloud Platforms:**")
                st.write(", ".join(tech_skills['cloud_platforms']))
            
            if tech_skills.get('databases'):
                st.markdown("**Databases:**")
                st.write(", ".join(tech_skills['databases']))
            
            st.markdown("---")
            st.markdown("#### 💡 Soft Skills")
            if skills.get('soft_skills'):
                st.write(", ".join(skills['soft_skills']))
            
            st.markdown("---")
            st.markdown("#### 📚 Domain Knowledge")
            if skills.get('domain_knowledge'):
                st.write(", ".join(skills['domain_knowledge']))
            
            # Suggestions
            st.markdown("---")
            if st.button("Get Skill Recommendations", use_container_width=True):
                with st.spinner(f"Analyzing gaps for {target_role} role..."):
                    suggestions = skill_extractor.suggest_missing_skills(skills, target_role)
                
                st.markdown(f"#### 🎯 Recommended Skills for {target_role}")
                
                for suggestion in suggestions:
                    priority_color = {
                        "High": "🔴",
                        "Medium": "🟡",
                        "Low": "🟢"
                    }.get(suggestion.get('priority', 'Medium'), "⚪")
                    
                    st.markdown(f"{priority_color} **{suggestion.get('skill')}** ({suggestion.get('priority')} Priority)")
                    st.markdown(f"_{suggestion.get('reason')}_")
                    st.markdown("")
    
    with tab3:
        st.markdown("### 💼 Job Description Match")
        
        job_description = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Paste the job description you want to match against..."
        )
        
        if job_description and 'skills' in st.session_state:
            if st.button("Analyze Match", type="primary", use_container_width=True):
                skill_extractor = get_skill_extractor()
                
                with st.spinner("Analyzing job match..."):
                    match_data = skill_extractor.match_to_job_description(
                        st.session_state['skills'],
                        job_description
                    )
                
                # Display match percentage
                match_pct = match_data.get('match_percentage', 0)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Match Score", f"{match_pct}%")
                
                with col2:
                    matching_count = len(match_data.get('matching_skills', []))
                    st.metric("Matching Skills", matching_count)
                
                with col3:
                    missing_count = len(match_data.get('missing_critical_skills', []))
                    st.metric("Missing Critical", missing_count)
                
                st.markdown("---")
                
                # Matching skills
                if match_data.get('matching_skills'):
                    st.markdown("#### ✅ Your Matching Skills")
                    st.write(", ".join(match_data['matching_skills']))
                
                # Missing critical
                if match_data.get('missing_critical_skills'):
                    st.markdown("#### ❌ Missing Critical Skills")
                    st.write(", ".join(match_data['missing_critical_skills']))
                
                # Nice to have
                if match_data.get('nice_to_have_missing'):
                    st.markdown("#### ⚠️ Nice-to-Have Skills You're Missing")
                    st.write(", ".join(match_data['nice_to_have_missing']))
                
                # Recommendations
                if match_data.get('recommendations'):
                    st.markdown("#### 💡 Recommendations")
                    for rec in match_data['recommendations']:
                        st.markdown(f"- {rec}")
        
        elif not job_description:
            st.info("👆 Paste a job description above and extract skills first")
        else:
            st.warning("⚠️ Extract skills from your resume first (go to Skills Analysis tab)")

else:
    st.info("👆 Upload your resume PDF to get started")
    
    # Example use case
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ Upload**")
        st.markdown("Upload your resume in PDF format")
    
    with col2:
        st.markdown("**2️⃣ Analyze**")
        st.markdown("AI extracts skills and analyzes content")
    
    with col3:
        st.markdown("**3️⃣ Improve**")
        st.markdown("Get actionable feedback and match to jobs")
