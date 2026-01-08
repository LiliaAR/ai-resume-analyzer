# Usage Examples

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up your OpenAI API key:**
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

3. **Run the app:**
```bash
streamlit run app.py
```

4. **Upload a resume PDF and analyze!**

## Example Workflow

1. Upload your resume PDF
2. Go to "Skills Analysis" tab → Click "Extract Skills"
3. Review categorized skills
4. Go to "Job Match" tab → Paste a job description
5. Get match percentage and recommendations

## Sample Job Description for Testing
```
Senior AI/ML Engineer

Requirements:
- 5+ years Python experience
- Experience with LangChain, LlamaIndex, or similar frameworks
- Production ML deployment experience
- Strong understanding of RAG architectures
- Experience with cloud platforms (AWS/Azure/GCP)
- Excellent communication skills

Nice to have:
- Experience with vector databases
- Knowledge of prompt engineering
- Open source contributions
```

## Tips

- For best results, use well-formatted PDF resumes
- More detailed resumes = better skill extraction
- Try different target roles in the dropdown
