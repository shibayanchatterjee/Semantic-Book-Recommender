# huggingface_vscode.py

import os
import json
import argparse
import requests
from pathlib import Path
import subprocess
import sys
from typing import List, Dict, Any, Optional, Union

class HuggingFaceVSCode:
    """A utility for integrating Hugging Face models with VS Code"""
    
    def __init__(self):
        self.config_file = Path.home() / ".hf_vscode_config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Error: Invalid configuration file. Using default configuration.")
                return {"api_token": None, "recent_models": []}
        return {"api_token": None, "recent_models": []}
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)
    
    def set_token(self, token: str) -> None:
        """Set the Hugging Face API token"""
        self.config["api_token"] = token
        self._save_config()
        print("API token saved successfully.")
    
    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for models in the Hugging Face Hub"""
        headers = {}
        if self.config["api_token"]:
            headers["Authorization"] = f"Bearer {self.config['api_token']}"
            
        response = requests.get(
            f"https://huggingface.co/api/models?search={query}&limit={limit}",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error searching models: {response.status_code}")
            return []
            
        models = response.json()
        
        # Print results in a formatted way
        print(f"\nFound {len(models)} models matching '{query}':\n")
        print("{:<5} {:<30} {:<20} {:<30}".format("#", "Model ID", "Task", "Downloads"))
        print("-" * 85)
        
        for i, model in enumerate(models):
            model_id = model.get("modelId", "Unknown")
            task = model.get("pipeline_tag", "Unknown")
            downloads = model.get("downloads", 0)
            print("{:<5} {:<30} {:<20} {:<30}".format(
                i+1, 
                model_id[:30], 
                task[:20], 
                f"{downloads:,}"
            ))
            
        return models
    
    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        headers = {}
        if self.config["api_token"]:
            headers["Authorization"] = f"Bearer {self.config['api_token']}"
            
        try:
            response = requests.get(
                f"https://huggingface.co/api/models/{model_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Error getting model details: {response.status_code}")
                return {}
                
            return response.json()
        except Exception as e:
            print(f"Error: {str(e)}")
            return {}
    
    def generate_code(self, model_id: str, lang: str = "python", task: str = None) -> str:
        """Generate code for using the specified model"""
        # Add model to recent list
        if model_id not in self.config["recent_models"]:
            self.config["recent_models"] = [model_id] + self.config["recent_models"]
            # Keep only 10 most recent
            self.config["recent_models"] = self.config["recent_models"][:10]
            self._save_config()
        
        # Get model details to determine the task if not specified
        if not task:
            details = self.get_model_details(model_id)
            task = details.get("pipeline_tag", "unknown")
        
        if lang.lower() == "python":
            return self._generate_python_code(model_id, task)
        elif lang.lower() in ["javascript", "js"]:
            return self._generate_js_code(model_id, task)
        else:
            print(f"Language {lang} not supported. Generating Python code instead.")
            return self._generate_python_code(model_id, task)
    
    def _generate_python_code(self, model_id: str, task: str) -> str:
        """Generate Python code for the model"""
        # Different code templates based on the task
        if task in ["text-classification", "sentiment-analysis"]:
            code = f"""
from transformers import pipeline

# Load the classification model
classifier = pipeline("text-classification", model="{model_id}")

# Example input
text = "I love using Hugging Face models in VS Code!"

# Get classification results
result = classifier(text)
print(f"Classification result: {{result}}")
"""
        
        elif task in ["text-generation", "text2text-generation"]:
            code = f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained("{model_id}")

# Example input
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {{generated_text}}")
"""
        
        elif task in ["question-answering"]:
            code = f"""
from transformers import pipeline

# Load question answering pipeline
qa_pipeline = pipeline("question-answering", model="{model_id}")

# Example input
context = "Hugging Face is an AI company that develops tools for building applications using machine learning."
question = "What does Hugging Face develop?"

# Get answer
result = qa_pipeline(question=question, context=context)
print(f"Answer: {{result['answer']}}")
print(f"Score: {{result['score']}}")
"""
        
        elif task in ["image-classification"]:
            code = f"""
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Load image classification pipeline
classifier = pipeline("image-classification", model="{model_id}")

# Example: Load an image from URL
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
image = Image.open(BytesIO(requests.get(image_url).content))

# Classify image
result = classifier(image)
print(f"Classification results: {{result}}")
"""
        
        else:
            # Generic code for any model
            code = f"""
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModel.from_pretrained("{model_id}")

# Example input
text = "Hello, I'm using Hugging Face models in VS Code!"
inputs = tokenizer(text, return_tensors="pt")

# Get model output
outputs = model(**inputs)

# Process the outputs (will vary depending on the model type)
print(f"Model output shape: {{outputs.last_hidden_state.shape}}")
"""
            
        return code
        
    def _generate_js_code(self, model_id: str, task: str) -> str:
        """Generate JavaScript code for the model"""
        # Generic JavaScript code
        code = f"""
// Using Hugging Face Inference API with JavaScript
// First install: npm install @huggingface/inference

import {{ HfInference }} from '@huggingface/inference';

// Initialize with your API token
const hf = new HfInference(process.env.HF_API_TOKEN);

async function runInference() {{
  try {{
    // Replace task with: 'textClassification', 'textGeneration', etc.
    const result = await hf.{self._js_task_name(task)}({{
      model: '{model_id}',
      inputs: 'Your input text here',
      // Add additional parameters based on the model's requirements
    }});
    
    console.log('Result:', result);
    return result;
  }} catch (error) {{
    console.error('Error running inference:', error);
  }}
}}

runInference();
"""
        return code
            
    def _js_task_name(self, task: str) -> str:
        """Convert HF task name to JS API method name"""
        task_mapping = {
            "text-classification": "textClassification",
            "sentiment-analysis": "textClassification",
            "text-generation": "textGeneration",
            "text2text-generation": "textGeneration",
            "question-answering": "questionAnswering",
            "image-classification": "imageClassification",
            "fill-mask": "fillMask"
        }
        return task_mapping.get(task, "pipeline")
    
    def insert_to_vscode(self, code: str) -> None:
        """Insert the generated code into the active VS Code editor"""
        try:
            # Create a temporary file with the code
            temp_file = Path("temp_hf_code.txt")
            with open(temp_file, 'w') as f:
                f.write(code)
                
            # Use VS Code CLI to insert the content
            if os.name == 'nt':  # Windows
                subprocess.run(["code", "--goto", str(temp_file)])
            else:  # macOS/Linux
                subprocess.run(["code", "--goto", str(temp_file)])
                
            print(f"\nCode generated for {model_id}. Opening in VS Code...")
        except Exception as e:
            print(f"Error inserting to VS Code: {str(e)}")
            print("\nGenerated code:\n")
            print(code)
    
    def install_vscode_extension(self) -> None:
        """Create a VS Code extension for Hugging Face integration"""
        print("Creating VS Code extension for Hugging Face integration...")
        
        # Create extension folder structure
        extension_dir = Path("huggingface-vscode-extension")
        extension_dir.mkdir(exist_ok=True)
        
        # Create package.json
        package_json = {
            "name": "huggingface-vscode",
            "displayName": "Hugging Face Integration",
            "description": "Integrate Hugging Face models with VS Code",
            "version": "0.1.0",
            "engines": {"vscode": "^1.60.0"},
            "categories": ["Machine Learning", "Other"],
            "activationEvents": ["onCommand:huggingface-vscode.searchModels"],
            "main": "./extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "huggingface-vscode.searchModels",
                        "title": "Hugging Face: Search Models"
                    }
                ]
            },
            "scripts": {"vscode:prepublish": "npm run compile"},
            "dependencies": {
                "child_process": "^1.0.2"
            }
        }
        
        with open(extension_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Create extension.js
        extension_js = """
const vscode = require('vscode');
const { exec } = require('child_process');

function activate(context) {
    console.log('Hugging Face Extension is now active');

    let disposable = vscode.commands.registerCommand('huggingface-vscode.searchModels', async function () {
        const query = await vscode.window.showInputBox({
            placeHolder: 'Search for Hugging Face models',
            prompt: 'Enter a search term'
        });
        
        if (query) {
            const terminal = vscode.window.createTerminal('Hugging Face');
            terminal.sendText(`python ${__dirname}/huggingface_vscode.py search "${query}"`);
            terminal.show();
        }
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}
"""
        
        with open(extension_dir / "extension.js", 'w') as f:
            f.write(extension_js)
        
        # Copy the Python script to the extension directory
        with open(extension_dir / "huggingface_vscode.py", 'w') as f:
            with open(__file__, 'r') as source:
                f.write(source.read())
        
        print(f"Extension created in {extension_dir.absolute()}")
        print("To install this extension in VS Code:")
        print("1. Open VS Code")
        print("2. Press Ctrl+Shift+P (or Cmd+Shift+P on macOS)")
        print("3. Type 'Extensions: Install from VSIX...'")
        print("4. Navigate to the extension folder and install it")

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Integration for VS Code")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Set token command
    token_parser = subparsers.add_parser("token", help="Set Hugging Face API token")
    token_parser.add_argument("api_token", help="Your Hugging Face API token")
    
    # Search models command
    search_parser = subparsers.add_parser("search", help="Search for models")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    
    # Generate code command
    code_parser = subparsers.add_parser("code", help="Generate code for a model")
    code_parser.add_argument("model_id", help="Model ID (e.g., 'gpt2', 'bert-base-uncased')")
    code_parser.add_argument("--lang", choices=["python", "javascript", "js"], default="python", 
                            help="Programming language")
    code_parser.add_argument("--task", help="Specific task for the model")
    
    # Install extension command
    subparsers.add_parser("install-extension", help="Create a VS Code extension")
    
    args = parser.parse_args()
    
    hf_vscode = HuggingFaceVSCode()
    
    if args.command == "token":
        hf_vscode.set_token(args.api_token)
    elif args.command == "search":
        hf_vscode.search_models(args.query, args.limit)
    elif args.command == "code":
        code = hf_vscode.generate_code(args.model_id, args.lang, args.task)
        print(code)
    elif args.command == "install-extension":
        hf_vscode.install_vscode_extension()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()