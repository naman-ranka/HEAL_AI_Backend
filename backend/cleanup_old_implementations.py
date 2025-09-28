#!/usr/bin/env python3
"""
Cleanup Script - Remove Old Implementations
Removes custom RAG and Genkit implementations, keeping only LangChain
"""

import os
import shutil
from pathlib import Path

def print_action(action, details=""):
    print(f"🔧 {action}")
    if details:
        print(f"   📋 {details}")

def cleanup_old_files():
    """Remove old implementation files"""
    
    print("🧹 CLEANING UP OLD IMPLEMENTATIONS")
    print("=" * 50)
    
    # Files to remove
    files_to_remove = [
        # Old main implementation
        "main.py",
        
        # Genkit implementations
        "genkit_rag_chat.py",
        "genkit_api.py", 
        "test_genkit_system.py",
        "requirements_genkit.txt",
        
        # Test files for old implementations
        "test_conversational_flow.py",
        "test_real_embeddings.py",
        "debug_upload.py",
        "test_upload.py",
        "check_models.py",
        "test_model.py",
        "test_imports.py",
        
        # Old requirements
        "requirements.txt"
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "rag",
        "ai", 
        "database",
        "tests"
    ]
    
    removed_files = []
    removed_dirs = []
    
    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print_action(f"Removed file: {file_path}")
            except Exception as e:
                print_action(f"Failed to remove {file_path}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_path)
                print_action(f"Removed directory: {dir_path}/")
            except Exception as e:
                print_action(f"Failed to remove {dir_path}/: {e}")
    
    # Database files
    db_files = [
        "heal.db",
        "genkit_heal.db",
        "genkit_rag.db"
    ]
    
    removed_dbs = []
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                removed_dbs.append(db_file)
                print_action(f"Removed database: {db_file}")
            except Exception as e:
                print_action(f"Failed to remove {db_file}: {e}")
    
    print("\n" + "=" * 50)
    print("📊 CLEANUP SUMMARY")
    print("=" * 50)
    
    print(f"✅ Files removed: {len(removed_files)}")
    for file in removed_files:
        print(f"   📄 {file}")
    
    print(f"\n✅ Directories removed: {len(removed_dirs)}")
    for directory in removed_dirs:
        print(f"   📁 {directory}/")
    
    print(f"\n✅ Databases removed: {len(removed_dbs)}")
    for db in removed_dbs:
        print(f"   🗄️ {db}")
    
    print("\n🎯 REMAINING FILES (LangChain Implementation):")
    remaining_files = [
        "langchain_main.py",
        "requirements_langchain.txt", 
        "test_langchain_complete.py",
        "cleanup_old_implementations.py"
    ]
    
    for file in remaining_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")

def create_new_structure():
    """Create clean structure for LangChain implementation"""
    
    print("\n🏗️  CREATING NEW STRUCTURE")
    print("=" * 50)
    
    # Rename files to standard names
    renames = [
        ("langchain_main.py", "main.py"),
        ("requirements_langchain.txt", "requirements.txt"),
        ("test_langchain_complete.py", "test_complete.py")
    ]
    
    for old_name, new_name in renames:
        if os.path.exists(old_name) and not os.path.exists(new_name):
            try:
                os.rename(old_name, new_name)
                print_action(f"Renamed: {old_name} → {new_name}")
            except Exception as e:
                print_action(f"Failed to rename {old_name}: {e}")
    
    # Create uploads directory
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print_action(f"Created directory: {uploads_dir}/")
    
    print("\n✨ NEW CLEAN STRUCTURE:")
    print("   📄 main.py (LangChain implementation)")
    print("   📄 requirements.txt (LangChain dependencies)")
    print("   📄 test_complete.py (comprehensive tests)")
    print("   📁 uploads/ (document storage)")
    print("   🗄️ langchain_heal.db (will be created on first run)")

def main():
    """Main cleanup function"""
    
    print("🚀 HEAL - IMPLEMENTATION CLEANUP")
    print("Moving to single LangChain implementation")
    print("=" * 60)
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will remove all old implementations!")
    print("   • Custom RAG implementation")
    print("   • Genkit-style implementation") 
    print("   • All associated files and databases")
    print("   • Test files for old systems")
    
    confirm = input("\n❓ Continue with cleanup? (yes/no): ").lower().strip()
    
    if confirm in ['yes', 'y']:
        cleanup_old_files()
        create_new_structure()
        
        print("\n🎉 CLEANUP COMPLETE!")
        print("=" * 50)
        print("✅ Successfully migrated to single LangChain implementation")
        print("\n🚀 Next steps:")
        print("   1. pip install -r requirements.txt")
        print("   2. python main.py")
        print("   3. python test_complete.py")
        print("\n📚 API Documentation: http://localhost:8000/docs")
        
    else:
        print("\n❌ Cleanup cancelled. No files were removed.")
        print("💡 All implementations remain available.")

if __name__ == "__main__":
    main()
