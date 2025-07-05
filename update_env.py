#!/usr/bin/env python3
"""
Auto-update .env file with Azure AI Foundry credentials
Updates your .env file with working OpenAI resource credentials
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

def run_az_command(command: str) -> str:
    """Run Azure CLI command and return output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def get_resource_credentials():
    """Get credentials from both Azure OpenAI resources"""
    print("🔍 Discovering Azure OpenAI resources...")
    
    # Your working Foundry resource (from the screenshot)
    foundry_resource = "poc76-mcqdm2q-eastus2"
    foundry_rg = "rg-ai102-lab"  # Assuming same resource group
    
    # Your original lab resource
    lab_resource = "openai-vge1s0"
    lab_rg = "rg-ai102-lab"
    
    credentials = {}
    
    # Try to get Foundry resource credentials (working one)
    print(f"📡 Getting credentials for {foundry_resource}...")
    try:
        # Get endpoint
        endpoint_cmd = f'az cognitiveservices account show --name {foundry_resource} --resource-group {foundry_rg} --query "properties.endpoint" -o tsv'
        foundry_endpoint = run_az_command(endpoint_cmd)
        
        # Get key
        key_cmd = f'az cognitiveservices account keys list --name {foundry_resource} --resource-group {foundry_rg} --query "key1" -o tsv'
        foundry_key = run_az_command(key_cmd)
        
        if foundry_endpoint and foundry_key:
            credentials['foundry'] = {
                'endpoint': foundry_endpoint,
                'key': foundry_key,
                'resource_name': foundry_resource,
                'resource_group': foundry_rg
            }
            print(f"✅ Found working Foundry resource: {foundry_resource}")
        
    except Exception as e:
        print(f"⚠️  Could not get Foundry credentials: {e}")
    
    # Try to get lab resource credentials (backup)
    print(f"📡 Getting credentials for {lab_resource}...")
    try:
        # Get endpoint
        endpoint_cmd = f'az cognitiveservices account show --name {lab_resource} --resource-group {lab_rg} --query "properties.endpoint" -o tsv'
        lab_endpoint = run_az_command(endpoint_cmd)
        
        # Get key
        key_cmd = f'az cognitiveservices account keys list --name {lab_resource} --resource-group {lab_rg} --query "key1" -o tsv'
        lab_key = run_az_command(key_cmd)
        
        if lab_endpoint and lab_key:
            credentials['lab'] = {
                'endpoint': lab_endpoint,
                'key': lab_key,
                'resource_name': lab_resource,
                'resource_group': lab_rg
            }
            print(f"✅ Found lab resource: {lab_resource}")
        
    except Exception as e:
        print(f"⚠️  Could not get lab credentials: {e}")
    
    return credentials

def get_other_services():
    """Get credentials for other AI services"""
    print("🧠 Getting other AI services credentials...")
    
    services = {}
    resource_group = "rg-ai102-lab"
    
    # AI Services to check
    ai_services = {
        'ai_foundry': 'ai-foundry-vge1s0',
        'document_intelligence': 'docint-vge1s0', 
        'computer_vision': 'cv-vge1s0',
        'text_analytics': 'text-vge1s0',
        'speech': 'speech-vge1s0'
    }
    
    for service_name, resource_name in ai_services.items():
        try:
            # Get endpoint
            endpoint_cmd = f'az cognitiveservices account show --name {resource_name} --resource-group {resource_group} --query "properties.endpoint" -o tsv'
            endpoint = run_az_command(endpoint_cmd)
            
            # Get key
            key_cmd = f'az cognitiveservices account keys list --name {resource_name} --resource-group {resource_group} --query "key1" -o tsv'
            key = run_az_command(key_cmd)
            
            if endpoint and key:
                services[service_name] = {
                    'endpoint': endpoint,
                    'key': key
                }
                print(f"   ✅ {service_name}: {resource_name}")
            else:
                print(f"   ⚠️  {service_name}: Not found or no access")
                
        except Exception as e:
            print(f"   ❌ {service_name}: Error - {e}")
    
    # Get Azure Search
    try:
        search_name = "search-vge1s0"
        search_endpoint = f"https://{search_name}.search.windows.net"
        
        key_cmd = f'az search admin-key show --service-name {search_name} --resource-group {resource_group} --query "primaryKey" -o tsv'
        search_key = run_az_command(key_cmd)
        
        if search_key:
            services['search'] = {
                'endpoint': search_endpoint,
                'key': search_key
            }
            print(f"   ✅ search: {search_name}")
    except:
        print(f"   ⚠️  search: Not found")
    
    # Get Storage
    try:
        storage_name = "storagevge1s0"
        conn_cmd = f'az storage account show-connection-string --name {storage_name} --resource-group {resource_group} --query "connectionString" -o tsv'
        storage_conn = run_az_command(conn_cmd)
        
        if storage_conn:
            services['storage'] = {
                'connection_string': storage_conn
            }
            print(f"   ✅ storage: {storage_name}")
    except:
        print(f"   ⚠️  storage: Not found")
    
    return services

def create_env_content(openai_creds, other_services):
    """Create .env file content"""
    
    # Use Foundry credentials if available, otherwise lab credentials
    if 'foundry' in openai_creds:
        openai_endpoint = openai_creds['foundry']['endpoint']
        openai_key = openai_creds['foundry']['key']
        resource_name = openai_creds['foundry']['resource_name']
        print(f"🎯 Using Foundry resource: {resource_name}")
    elif 'lab' in openai_creds:
        openai_endpoint = openai_creds['lab']['endpoint']
        openai_key = openai_creds['lab']['key']
        resource_name = openai_creds['lab']['resource_name']
        print(f"🎯 Using lab resource: {resource_name}")
    else:
        print("❌ No OpenAI credentials found!")
        return None
    
    # Get subscription ID
    try:
        subscription_id = run_az_command('az account show --query "id" -o tsv')
    except:
        subscription_id = "YOUR_SUBSCRIPTION_ID"
    
    env_content = f"""# AI-102 Lab Environment Variables
# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Updated with working Azure AI Foundry credentials

# Azure Subscription
AZURE_SUBSCRIPTION_ID={subscription_id}
AZURE_RESOURCE_GROUP=rg-ai102-lab
AZURE_LOCATION=eastus

# Azure OpenAI (Working Foundry Resource)
AZURE_OPENAI_ENDPOINT={openai_endpoint}
AZURE_OPENAI_KEY={openai_key}
AZURE_OPENAI_RESOURCE_NAME={resource_name}

# Model Deployment Names (from your Foundry deployments)
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-2
AZURE_OPENAI_GPT4_DEPLOYMENT=gpt-4
AZURE_OPENAI_GPT35_DEPLOYMENT=gpt-35-turbo
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure AI Foundry
AI_FOUNDRY_ENDPOINT={other_services.get('ai_foundry', {}).get('endpoint', 'NOT_FOUND')}
AI_FOUNDRY_KEY={other_services.get('ai_foundry', {}).get('key', 'NOT_FOUND')}

# Document Intelligence
DOCUMENT_INTELLIGENCE_ENDPOINT={other_services.get('document_intelligence', {}).get('endpoint', 'NOT_FOUND')}
DOCUMENT_INTELLIGENCE_KEY={other_services.get('document_intelligence', {}).get('key', 'NOT_FOUND')}

# Computer Vision
COMPUTER_VISION_ENDPOINT={other_services.get('computer_vision', {}).get('endpoint', 'NOT_FOUND')}
COMPUTER_VISION_KEY={other_services.get('computer_vision', {}).get('key', 'NOT_FOUND')}

# Text Analytics (Language Service)
TEXT_ANALYTICS_ENDPOINT={other_services.get('text_analytics', {}).get('endpoint', 'NOT_FOUND')}
TEXT_ANALYTICS_KEY={other_services.get('text_analytics', {}).get('key', 'NOT_FOUND')}

# Speech Service
SPEECH_ENDPOINT={other_services.get('speech', {}).get('endpoint', 'NOT_FOUND')}
SPEECH_KEY={other_services.get('speech', {}).get('key', 'NOT_FOUND')}
SPEECH_REGION=eastus

# Azure AI Search
SEARCH_SERVICE_ENDPOINT={other_services.get('search', {}).get('endpoint', 'NOT_FOUND')}
SEARCH_SERVICE_KEY={other_services.get('search', {}).get('key', 'NOT_FOUND')}
SEARCH_INDEX_NAME=loan-documents

# Storage Account
AZURE_STORAGE_CONNECTION_STRING={other_services.get('storage', {}).get('connection_string', 'NOT_FOUND')}

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Working Model Endpoints (for reference)
# GPT-4: {openai_endpoint}openai/deployments/gpt-4-2/chat/completions?api-version=2024-02-15-preview
# GPT-3.5: {openai_endpoint}openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-15-preview
# Embeddings: {openai_endpoint}openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-02-15-preview
"""
    
    return env_content

def backup_existing_env():
    """Backup existing .env file"""
    if os.path.exists('.env'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'.env.backup_{timestamp}'
        os.rename('.env', backup_name)
        print(f"📋 Backed up existing .env to {backup_name}")
        return backup_name
    return None

def main():
    print("🚀 AI-102 Lab .env Auto-Updater")
    print("="*50)
    
    # Check if Azure CLI is available
    try:
        run_az_command('az account show')
        print("✅ Azure CLI authenticated")
    except:
        print("❌ Azure CLI not authenticated. Please run: az login")
        return
    
    # Get credentials
    openai_creds = get_resource_credentials()
    other_services = get_other_services()
    
    if not openai_creds:
        print("❌ Could not find any OpenAI resources!")
        return
    
    # Create .env content
    env_content = create_env_content(openai_creds, other_services)
    if not env_content:
        return
    
    # Backup existing .env
    backup_file = backup_existing_env()
    
    # Write new .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print(f"✅ Created new .env file with working credentials!")
        
        # Show summary
        print("\n" + "="*50)
        print("📋 CONFIGURATION SUMMARY")
        print("="*50)
        
        if 'foundry' in openai_creds:
            print(f"🎯 OpenAI Resource: {openai_creds['foundry']['resource_name']} (Foundry)")
            print(f"🔗 Endpoint: {openai_creds['foundry']['endpoint']}")
        elif 'lab' in openai_creds:
            print(f"🎯 OpenAI Resource: {openai_creds['lab']['resource_name']} (Lab)")
            print(f"🔗 Endpoint: {openai_creds['lab']['endpoint']}")
        
        print(f"📦 Model Deployments Available:")
        print(f"   • gpt-4-2 (Primary GPT-4)")
        print(f"   • gpt-4 (Alternative GPT-4)")
        print(f"   • gpt-35-turbo (Fast model)")
        print(f"   • text-embedding-ada-002 (Embeddings)")
        
        print(f"\n🧠 Other AI Services:")
        for service, details in other_services.items():
            status = "✅" if details.get('endpoint') or details.get('connection_string') else "❌"
            print(f"   {status} {service}")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Test your setup: python -c \"from dotenv import load_dotenv; load_dotenv(); print('✅ .env loaded')\"")
        print(f"2. Install dependencies: pip install -r requirements.txt")
        print(f"3. Start building your AI-102 lab!")
        
        if backup_file:
            print(f"\n💾 Backup: Original .env saved as {backup_file}")
        
    except Exception as e:
        print(f"❌ Error writing .env file: {e}")
        return

if __name__ == "__main__":
    main()