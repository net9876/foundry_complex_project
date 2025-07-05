#!/usr/bin/env python3
"""
Test Azure OpenAI Embeddings Deployment
Verifies that your text-embedding-ada-002 model is working
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def test_embeddings():
    """Test the embeddings deployment"""
    print("🧪 Testing Azure OpenAI Embeddings...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    key = os.getenv('AZURE_OPENAI_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
    
    print(f"📡 Endpoint: {endpoint}")
    print(f"🚀 Deployment: {deployment_name}")
    print(f"🔑 Key: ***{key[-4:] if key else 'NOT_FOUND'}")
    
    if not endpoint or not key:
        print("❌ Missing endpoint or key in .env file")
        return False
    
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version="2024-02-15-preview"
        )
        
        # Test embedding
        print("\n🔄 Testing embeddings...")
        test_text = "This is a test for AI-102 lab embeddings functionality."
        
        response = client.embeddings.create(
            model=deployment_name,
            input=test_text
        )
        
        # Get the embedding
        embedding = response.data[0].embedding
        embedding_length = len(embedding)
        
        print(f"✅ Embeddings successful!")
        print(f"📊 Text: '{test_text}'")
        print(f"📏 Embedding dimensions: {embedding_length}")
        print(f"🔢 First 5 values: {embedding[:5]}")
        print(f"💰 Tokens used: {response.usage.total_tokens}")
        
        # Test with multiple texts
        print(f"\n🔄 Testing batch embeddings...")
        test_texts = [
            "Azure AI services for document processing",
            "Computer vision and image analysis",
            "Natural language processing capabilities"
        ]
        
        batch_response = client.embeddings.create(
            model=deployment_name,
            input=test_texts
        )
        
        print(f"✅ Batch embeddings successful!")
        print(f"📊 Processed {len(test_texts)} texts")
        print(f"💰 Total tokens used: {batch_response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return False

def test_other_services():
    """Test other AI services from .env"""
    print(f"\n🧠 Testing other AI services configuration...")
    
    services = {
        'Document Intelligence': 'DOCUMENT_INTELLIGENCE_ENDPOINT',
        'Computer Vision': 'COMPUTER_VISION_ENDPOINT',
        'Text Analytics': 'TEXT_ANALYTICS_ENDPOINT',
        'Speech Service': 'SPEECH_ENDPOINT',
        'AI Search': 'SEARCH_SERVICE_ENDPOINT',
        'Storage': 'AZURE_STORAGE_CONNECTION_STRING'
    }
    
    for service_name, env_var in services.items():
        value = os.getenv(env_var)
        status = "✅" if value and value != "NOT_FOUND" else "❌"
        print(f"   {status} {service_name}: {'Configured' if value and value != 'NOT_FOUND' else 'Missing'}")

def main():
    print("🚀 AI-102 Lab Services Test")
    print("="*50)
    
    # Test embeddings
    embeddings_work = test_embeddings()
    
    # Test other services
    test_other_services()
    
    print(f"\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    if embeddings_work:
        print("✅ Azure OpenAI Embeddings: WORKING")
        print("🎯 Ready for:")
        print("   • RAG (Retrieval Augmented Generation)")
        print("   • Semantic search")
        print("   • Document similarity")
        print("   • Knowledge mining tasks")
    else:
        print("❌ Azure OpenAI Embeddings: FAILED")
    
    print(f"\n🚀 Next Steps for AI-102 Lab:")
    print("1. ✅ Embeddings are working - you can build RAG systems")
    print("2. 🔄 Request quota for GPT models in Azure Portal")
    print("3. 🧠 Use other AI services (Document Intelligence, Computer Vision)")
    print("4. 🎯 Start with knowledge mining and document processing exercises")

if __name__ == "__main__":
    main()