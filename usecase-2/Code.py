import os
import logging
from typing import Dict, List, Any
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from playwright.sync_api import sync_playwright

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TestGenerator:
    """AI-Powered Test Case Generator and Automation Tool"""

    def __init__(self):
        self.sample_stories = [
            {"id": "US001", "title": "User Login", "description": "As a user, I want to log in to the system",
             "acceptance_criteria": [
                 "Given the user is on the login page",
                 "When the user enters valid credentials",
                 "Then the user should be redirected to the dashboard"]},
            {"id": "US002", "title": "Password Reset", "description": "As a user, I want to reset my password",
             "acceptance_criteria": [
                 "Given the user is on the login page",
                 "When the user clicks on 'Forgot Password'",
                 "And the user enters their email",
                 "Then a reset link should be sent to their email"]}    
        ]

        # AI Models
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Initialize NLTK for NLP tests
        try:
          nltk.data.find('tokenizers/punkt')
          nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
          nltk.download('punkt')
    
          try:
            nltk.data.find('tokenizers/punkt_tab')
          except LookupError:
       
            nltk.download('all')

    def preprocess_story(self, user_story: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess user story for NLP analysis"""
        processed_story = {
            "id": user_story["id"],
            "title": user_story["title"],
            "tokens": word_tokenize(user_story["description"]),
            "criteria_tokens": [word_tokenize(criteria) for criteria in user_story["acceptance_criteria"]]
        }
        return processed_story

    def generate_bdd_test(self, user_story: Dict[str, Any]) -> str:
        """Generate BDD-style test case using AI"""
        input_text = f"Generate BDD test case: {user_story['description']}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output_ids = self.model.generate(
           input_ids,
           max_length=200,
           repetition_penalty=2.0,
           do_sample=True,
           top_k=50,
           top_p=0.95,
           temperature=0.7,
           num_beams=5,
           early_stopping=True,
           length_penalty=2.0
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Format as Gherkin syntax
        feature_name = user_story['title']
        scenario_name = f"Test {user_story['title']}"
        
        gherkin_output = f"Feature: {feature_name}\n\n"
        gherkin_output += f"Scenario: {scenario_name}\n"
        
        # Add acceptance criteria directly
        for criteria in user_story["acceptance_criteria"]:
            gherkin_output += f"  {criteria}\n"
            
        return gherkin_output

    def generate_nlp_test(self, user_story: Dict[str, Any]) -> str:
        """Generate NLP-based test analysis"""
        processed = self.preprocess_story(user_story)
        
        # Create a simple NLP analysis report
        nlp_output = f"# NLP Test Analysis for {user_story['title']}\n\n"
        nlp_output += "## Story Description Analysis\n"
        nlp_output += f"- Story ID: {user_story['id']}\n"
        nlp_output += f"- Word Count: {len(processed['tokens'])}\n"
        nlp_output += f"- Key Terms: {', '.join(processed['tokens'][:5])}\n\n"
        
        nlp_output += "## Acceptance Criteria Analysis\n"
        for i, criteria in enumerate(user_story["acceptance_criteria"]):
            tokens = processed['criteria_tokens'][i]
            nlp_output += f"### Criteria {i+1}\n"
            nlp_output += f"- Raw Text: {criteria}\n"
            nlp_output += f"- Word Count: {len(tokens)}\n"
            nlp_output += f"- Action Words: {[word for word in tokens if word.lower() in ['when', 'then', 'given', 'and']]}\n"
            nlp_output += "\n"
            
        nlp_output += "## Test Coverage Estimation\n"
        coverage = len(user_story["acceptance_criteria"]) * 20  # Simple heuristic
        nlp_output += f"- Estimated Coverage: {min(coverage, 100)}%\n"
        nlp_output += f"- Criteria Count: {len(user_story['acceptance_criteria'])}\n"
        
        return nlp_output

    def generate_playwright_test(self, user_story: Dict[str, Any]) -> str:
        """Generate Playwright test script using Page Object Model (POM)"""
        test_code = "import { test, expect } from '@playwright/test';\n"
        test_code += f"import {{ LoginPage }} from './pages/LoginPage';\n"

        test_code += f"test('{user_story['title']}', async ({{ page }}) => {{\n"
        test_code += "  const loginPage = new LoginPage(page);\n"

        step_mapping = {
            "Given the user is on the login page": "  await loginPage.goto();\n",
            "When the user enters valid credentials": "  await loginPage.login('testuser', 'password123');\n",
            "Then the user should be redirected to the dashboard": "  await expect(page).toHaveURL('https://example.com/dashboard');\n",
            "When the user clicks on 'Forgot Password'": "  await loginPage.clickForgotPassword();\n",
            "And the user enters their email": "  await loginPage.enterEmail('user@example.com');\n  await loginPage.submitReset();\n",
            "Then a reset link should be sent to their email": "  await expect(page.locator('text=Reset link sent')).toBeVisible();\n"
        }

        for criteria in user_story["acceptance_criteria"]:
            test_code += step_mapping.get(criteria, f"  // TODO: Implement step - {criteria}\n")

        test_code += "});\n"
        return test_code

    def save_to_file(self, content: str, file_path: str):
        """Save generated test cases or scripts to a file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")

    def run(self):
        """Run the test generation pipeline"""
        logger.info("Starting AI-powered test generator...")

        output_dirs = ["output/bdd_tests", "output/nlp_tests", "output/playwright_tests"]
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)

        for story in self.sample_stories:
            logger.info(f"Processing '{story['title']}'...")

            try:
                # Generate BDD tests
                bdd_test = self.generate_bdd_test(story)
                self.save_to_file(bdd_test, f"output/bdd_tests/{story['id']}.feature")
                
                # Generate NLP analysis
                nlp_test = self.generate_nlp_test(story)
                self.save_to_file(nlp_test, f"output/nlp_tests/{story['id']}_analysis.md")

                # Generate Playwright tests
                playwright_test = self.generate_playwright_test(story)
                self.save_to_file(playwright_test, f"output/playwright_tests/{story['id']}.spec.js")

                logger.info(f"Generated tests for '{story['title']}'")
            except Exception as e:
                logger.error(f"Error generating tests for '{story['title']}': {e}")

        logger.info(f"- BDD tests saved to: output/bdd_tests")
        logger.info(f"- NLP tests saved to: output/nlp_tests")
        logger.info(f"- Playwright tests saved to: output/playwright_tests")
        logger.info("\nTest generation complete!")

if __name__ == "__main__":
    generator = TestGenerator()
    generator.run()
