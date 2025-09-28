#!/usr/bin/env python3
"""
Debug script to test the complete upload and RAG process
Run this to trace every step of document processing
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print('='*50)

def test_health():
    print_section("HEALTH CHECK")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_gemini():
    print_section("GEMINI API TEST")
    try:
        response = requests.post(f"{BASE_URL}/debug/test-gemini", json={
            "prompt": "Hello Gemini, please respond with 'Test successful' if you can hear this."
        })
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

def test_documents_list():
    print_section("DOCUMENTS LIST")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Total documents: {result.get('total', 0)}")
        for doc in result.get('documents', []):
            print(f"  📄 {doc['original_name']} - {doc['processing_status']} - {doc['chunk_count']} chunks")
        return True
    except Exception as e:
        print(f"❌ Documents list failed: {e}")
        return False

def test_latest_upload():
    print_section("LATEST UPLOAD DEBUG")
    try:
        response = requests.get(f"{BASE_URL}/debug/latest-upload")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Document info
            doc_info = result.get('document_info', {})
            print(f"📄 Document: {doc_info.get('original_name', 'Unknown')}")
            print(f"📁 File size: {doc_info.get('file_size', 0)} bytes")
            print(f"📋 Status: {doc_info.get('processing_status', 'Unknown')}")
            print(f"🕒 Upload time: {doc_info.get('upload_timestamp', 'Unknown')}")
            
            # Chunk statistics
            stats = result.get('chunk_statistics', {})
            print(f"\n📊 CHUNK STATISTICS:")
            print(f"  Total chunks: {stats.get('total_chunks', 0)}")
            print(f"  Chunks with embeddings: {stats.get('chunks_with_embeddings', 0)}")
            print(f"  Average chunk length: {stats.get('average_chunk_length', 0):.1f} chars")
            print(f"  Total text length: {stats.get('total_text_length', 0)} chars")
            
            # Policy analysis
            policy = result.get('policy_analysis', {})
            if policy:
                print(f"\n📋 POLICY ANALYSIS:")
                print(f"  Deductible: {policy.get('deductible', 'Not found')}")
                print(f"  Out-of-pocket max: {policy.get('out_of_pocket_max', 'Not found')}")
                print(f"  Copay: {policy.get('copay', 'Not found')}")
                print(f"  Confidence: {policy.get('confidence_score', 0)}")
            
            # Show first few chunks
            chunks = result.get('chunks', [])
            if chunks:
                print(f"\n📝 FIRST FEW CHUNKS:")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"  Chunk {i+1}: {chunk['chunk_text'][:100]}...")
                    print(f"    Length: {chunk['text_length']} chars, Has embedding: {chunk['has_embedding']}")
            
            return True
        else:
            print(f"❌ Failed to get latest upload: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Latest upload debug failed: {e}")
        return False

def test_rag_search():
    print_section("RAG SEARCH TEST")
    try:
        response = requests.post(f"{BASE_URL}/rag/search", json={
            "query": "What is the deductible?",
            "top_k": 3
        })
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Query: {result.get('query', '')}")
            print(f"Total found: {result.get('total_found', 0)}")
            print(f"Execution time: {result.get('execution_time_ms', 0)}ms")
            
            chunks = result.get('chunks', [])
            if chunks:
                print(f"\n🔍 RETRIEVED CHUNKS:")
                for i, chunk in enumerate(chunks):
                    print(f"  {i+1}. Similarity: {chunk['similarity_score']:.3f}")
                    print(f"     Source: {chunk['source_document']}")
                    print(f"     Text: {chunk['text'][:150]}...")
            else:
                print("❌ No chunks retrieved")
                
            return len(chunks) > 0
        else:
            print(f"❌ RAG search failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ RAG search test failed: {e}")
        return False

def test_chat_session():
    print_section("CHAT SESSION TEST")
    try:
        # Create session
        response = requests.post(f"{BASE_URL}/chat/sessions", json={})
        print(f"Create session status: {response.status_code}")
        
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data.get('session_id')
            print(f"✅ Created session: {session_id}")
            
            # Send test message
            response = requests.post(f"{BASE_URL}/chat/sessions/{session_id}/messages", json={
                "message": "What is my deductible?"
            })
            print(f"Send message status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"💬 Bot response: {result.get('message', '')[:200]}...")
                print(f"🎯 Confidence: {result.get('confidence', 0)}")
                print(f"📚 Sources: {len(result.get('sources', []))}")
                return True
            else:
                print(f"❌ Send message failed: {response.text}")
                return False
        else:
            print(f"❌ Create session failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat session test failed: {e}")
        return False

def main():
    print("🚀 HEAL DEBUG - Comprehensive Upload Process Test")
    print("=" * 60)
    
    # Test sequence
    tests = [
        ("Health Check", test_health),
        ("Gemini API", test_gemini),
        ("Documents List", test_documents_list),
        ("Latest Upload Analysis", test_latest_upload),
        ("RAG Search", test_rag_search),
        ("Chat Session", test_chat_session)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n⏳ Running {test_name}...")
        try:
            results[test_name] = test_func()
            status = "✅ PASSED" if results[test_name] else "❌ FAILED"
            print(f"Result: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ FAILED: {e}")
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some issues detected. Check logs above.")

if __name__ == "__main__":
    main()
