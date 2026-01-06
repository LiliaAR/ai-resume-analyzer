import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import PyPDF2
import os
from dotenv import load_dotenv

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

## Skills Analysis
**Technical Skills Found:** [List them]
**Missing Skills for Modern Tech Roles:** [Suggest 3-5]

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
    - Strengths & weaknesses
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
    
    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        llm = get_llm()
        
        with st.spinner("AI is analyzing your resume... This may take 30-60 seconds..."):
            analysis = analyze_resume(resume_text, llm)
        
        st.divider()
        st.markdown("## 📊 Analysis Results")
        st.markdown(analysis)
        
        # Download button
        st.download_button(
            label="📥 Download Analysis",
            data=analysis,
            file_name="resume_analysis.txt",
            mime="text/plain"
        )

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
        st.markdown("AI analyzes your resume content")
    
    with col3:
        st.markdown("**3️⃣ Improve**")
        st.markdown("Get actionable feedback to improve")
