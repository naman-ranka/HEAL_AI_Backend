#!/usr/bin/env python3
"""
Setup script for HEAL backend
This script helps set up the backend environment and dependencies.
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_environment():
    """Set up the backend environment."""
    print("🚀 Setting up HEAL backend environment...\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Change to backend directory
    os.chdir("backend")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("🔧 Creating .env file...")
        with open(".env", "w") as f:
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
        print("✅ .env file created")
        print("⚠️  Please edit the .env file and add your actual Gemini API key")
    else:
        print("✅ .env file already exists")
    
    print("\n🎉 HEAL Backend with Genkit setup completed!")
    print("\nNext steps:")
    print("1. Get a Gemini API key from: https://aistudio.google.com/app/apikey")
    print("2. Edit backend/.env and replace 'your_gemini_api_key_here' with your actual API key")
    print("3. Test the setup: cd backend && python test_backend.py")
    print("4. Run the server: cd backend && python main.py")
    print("5. For enhanced debugging: cd backend && genkit start -- python main.py")
    print("\n🚀 Genkit Features Available:")
    print("   - Structured AI responses with Pydantic schemas")
    print("   - Type-safe document analysis flows")
    print("   - Built-in observability and debugging")
    print("   - Ready-to-use chatbot functionality")
    print("\n📚 Available endpoints after setup:")
    print("   - http://localhost:8000/docs (Interactive API documentation)")
    print("   - http://localhost:8000/health (System health check)")
    print("   - http://localhost:8000/upload (Insurance document analysis)")
    print("   - http://localhost:8000/chat (AI chatbot for policy questions)")
    print("   - http://localhost:8000/summarize (Document summarization)")
    print("   - http://localhost:8000/generate-questions (Generate policy questions)")
    print("\n🛠️ Genkit Developer UI:")
    print("   - Run: genkit start -- python main.py")
    print("   - Access enhanced debugging and flow visualization")
    
    return True

if __name__ == "__main__":
    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
