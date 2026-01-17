"""
Unit tests for skill extraction module
"""
import pytest
from utils.skill_extractor import SkillExtractor

class TestSkillExtractor:
    
    def test_initialization(self):
        """Test SkillExtractor initializes correctly"""
        extractor = SkillExtractor()
        assert extractor is not None
        assert extractor.llm is not None
    
    def test_empty_skills_response(self):
        """Test empty skills response structure"""
        extractor = SkillExtractor()
        response = extractor._empty_skills_response()
        
        assert isinstance(response, dict)
        assert 'technical_skills' in response
        assert 'soft_skills' in response
        assert 'domain_knowledge' in response
        assert 'certifications' in response
    
    def test_empty_match_response(self):
        """Test empty match response structure"""
        extractor = SkillExtractor()
        response = extractor._empty_match_response()
        
        assert isinstance(response, dict)
        assert response['match_percentage'] == 0
        assert isinstance(response['matching_skills'], list)
        assert isinstance(response['missing_critical_skills'], list)
    
    def test_extract_skills_empty_input(self):
        """Test that empty input returns empty response"""
        extractor = SkillExtractor()
        result = extractor.extract_skills("")
        
        assert isinstance(result, dict)
        assert 'technical_skills' in result
    
    def test_extract_skills_short_input(self):
        """Test that very short input returns empty response"""
        extractor = SkillExtractor()
        result = extractor.extract_skills("Short")
        
        assert isinstance(result, dict)
        assert 'technical_skills' in result

# Add more tests as the module grows
