import os
import logging
from typing import Dict, List, Any
import json
import pandas as pd
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

        #AI Model
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')

    def generate_bdd_test(self, user_story: Dict[str, Any]) -> str:
        """Generate BDD-style test case using AI"""
        input_text = f"Generate BDD test case: {user_story['description']}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output_ids = self.model.generate(input_ids, max_length=200)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

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
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")

    def run(self):
        """Run the test generation pipeline"""
        logger.info("Starting AI-powered test generator...")

        output_dirs = ["output/bdd_tests", "output/playwright_tests"]
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)

        for story in self.sample_stories:
            logger.info(f"Processing '{story['title']}'...")

            try:
                bdd_test = self.generate_bdd_test(story)
                self.save_to_file(bdd_test, f"output/bdd_tests/{story['id']}.feature")

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