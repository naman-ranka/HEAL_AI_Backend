#!/usr/bin/env python3
"""
Check available Gemini models
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    
    print("🔍 Checking available Gemini models...")
    
    try:
        models = list(genai.list_models())
        print(f"\n📋 Found {len(models)} models:")
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
            else:
                print(f"❌ {model.name} (no generateContent support)")
                
        # Test a simple generation
        print("\n🧪 Testing model generation...")
        
        # Try different models
        test_models = ['gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro-vision']
        
        for model_name in test_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello, this is a test.")
                print(f"✅ {model_name} - Working")
                break
            except Exception as e:
                print(f"❌ {model_name} - Error: {str(e)[:100]}...")
                
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        
else:
    print("❌ No GEMINI_API_KEY found")
