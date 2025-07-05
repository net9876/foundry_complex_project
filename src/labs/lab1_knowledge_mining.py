#!/usr/bin/env python3
"""
AI-102 Lab 1: Knowledge Mining & RAG System
===========================================

This lab demonstrates:
- Document processing and chunking
- Embedding generation with Azure OpenAI
- Vector storage and similarity search
- Knowledge extraction and retrieval
- RAG (Retrieval Augmented Generation) foundations

AI-102 Exam Coverage:
- Implement knowledge mining and information extraction solutions (15-20%)
- Implement generative AI solutions (15-20%)
- Plan and manage an Azure AI solution (20-25%)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with its metadata"""
    id: str
    content: str
    source_file: str
    chunk_index: int
    embedding: List[float] = None
    metadata: Dict[str, Any] = None

class KnowledgeMiningSystem:
    """
    AI-102 Knowledge Mining System
    
    Demonstrates core AI-102 concepts:
    - Document ingestion and processing
    - Embedding generation for semantic search
    - Vector similarity operations
    - Knowledge extraction workflows
    """
    
    def __init__(self):
        """Initialize the knowledge mining system"""
        self.setup_azure_openai()
        self.document_chunks: List[DocumentChunk] = []
        self.knowledge_base_path = Path("data/knowledge_base")
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        print("🧠 AI-102 Knowledge Mining System initialized")
        print(f"📊 Using model: {self.embedding_deployment}")
        print(f"💾 Knowledge base: {self.knowledge_base_path}")
    
    def setup_azure_openai(self):
        """Setup Azure OpenAI client for embeddings"""
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = os.getenv('AZURE_OPENAI_KEY')
        self.embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Missing Azure OpenAI credentials in .env file")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-02-15-preview"
        )
        print(f"✅ Connected to Azure OpenAI: {self.endpoint}")
    
    def load_documents(self, documents_path: str = "data/sample_documents") -> List[str]:
        """
        Load documents from the specified directory
        
        AI-102 Skill: Document ingestion for knowledge mining
        """
        documents = []
        docs_path = Path(documents_path)
        
        if not docs_path.exists():
            print(f"❌ Documents path not found: {documents_path}")
            return documents
        
        print(f"📁 Loading documents from: {documents_path}")
        
        # Support various file types
        supported_extensions = ['.txt', '.md', '.json']
        
        for file_path in docs_path.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents.append({
                                'content': content,
                                'source': file_path.name,
                                'path': str(file_path)
                            })
                            print(f"   ✅ Loaded: {file_path.name} ({len(content)} chars)")
                except Exception as e:
                    print(f"   ❌ Error loading {file_path.name}: {e}")
        
        print(f"📚 Loaded {len(documents)} documents")
        return documents
    
    def chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split document into overlapping chunks for better context preservation
        
        AI-102 Skill: Document segmentation for knowledge extraction
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if content[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(content):
                break
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, str]]) -> List[DocumentChunk]:
        """
        Process documents into chunks with embeddings
        
        AI-102 Skills:
        - Document processing workflows
        - Embedding generation for semantic search
        """
        print("\n🔄 Processing documents into chunks...")
        
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            source = doc['source']
            
            # Chunk the document
            chunks = self.chunk_document(content)
            print(f"   📄 {source}: {len(chunks)} chunks")
            
            # Create DocumentChunk objects
            for i, chunk_content in enumerate(chunks):
                chunk_id = hashlib.md5(f"{source}_{i}_{chunk_content}".encode()).hexdigest()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    source_file=source,
                    chunk_index=i,
                    metadata={
                        'processed_at': datetime.now().isoformat(),
                        'chunk_size': len(chunk_content),
                        'original_doc_size': len(content)
                    }
                )
                all_chunks.append(chunk)
        
        print(f"✅ Created {len(all_chunks)} document chunks")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 10) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks using Azure OpenAI
        
        AI-102 Skill: Embedding generation for semantic search capabilities
        """
        print(f"\n🔮 Generating embeddings for {len(chunks)} chunks...")
        
        total_tokens = 0
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            print(f"   🔄 Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            try:
                # Generate embeddings for the batch
                response = self.client.embeddings.create(
                    model=self.embedding_deployment,
                    input=batch_texts
                )
                
                # Store embeddings in chunks
                for j, embedding_data in enumerate(response.data):
                    chunks[i + j].embedding = embedding_data.embedding
                
                total_tokens += response.usage.total_tokens
                
            except Exception as e:
                print(f"   ❌ Error generating embeddings for batch: {e}")
                return chunks
        
        print(f"✅ Generated embeddings using {total_tokens} tokens")
        print(f"📏 Embedding dimensions: {len(chunks[0].embedding) if chunks and chunks[0].embedding else 'Unknown'}")
        
        return chunks
    
    def save_knowledge_base(self, chunks: List[DocumentChunk]):
        """
        Save processed chunks to knowledge base
        
        AI-102 Skill: Knowledge base management and persistence
        """
        print(f"\n💾 Saving knowledge base...")
        
        knowledge_base = {
            'created_at': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'model_used': self.embedding_deployment,
            'chunks': []
        }
        
        for chunk in chunks:
            chunk_data = {
                'id': chunk.id,
                'content': chunk.content,
                'source_file': chunk.source_file,
                'chunk_index': chunk.chunk_index,
                'embedding': chunk.embedding,
                'metadata': chunk.metadata
            }
            knowledge_base['chunks'].append(chunk_data)
        
        # Save to JSON file
        kb_file = self.knowledge_base_path / 'knowledge_base.json'
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Knowledge base saved: {kb_file}")
        print(f"📊 Contains {len(chunks)} chunks from {len(set(c.source_file for c in chunks))} documents")
    
    def load_knowledge_base(self) -> List[DocumentChunk]:
        """
        Load existing knowledge base
        
        AI-102 Skill: Knowledge base retrieval and management
        """
        kb_file = self.knowledge_base_path / 'knowledge_base.json'
        
        if not kb_file.exists():
            print("📝 No existing knowledge base found")
            return []
        
        print(f"📖 Loading knowledge base from: {kb_file}")
        
        try:
            with open(kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            for chunk_data in data['chunks']:
                chunk = DocumentChunk(
                    id=chunk_data['id'],
                    content=chunk_data['content'],
                    source_file=chunk_data['source_file'],
                    chunk_index=chunk_data['chunk_index'],
                    embedding=chunk_data['embedding'],
                    metadata=chunk_data['metadata']
                )
                chunks.append(chunk)
            
            print(f"✅ Loaded {len(chunks)} chunks from knowledge base")
            print(f"📅 Created: {data['created_at']}")
            print(f"🤖 Model: {data['model_used']}")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return []
    
    def semantic_search(self, query: str, chunks: List[DocumentChunk], top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform semantic search using embeddings
        
        AI-102 Skills:
        - Semantic search implementation
        - Vector similarity calculations
        - Relevance ranking
        """
        print(f"\n🔍 Performing semantic search for: '{query}'")
        
        # Generate embedding for the query
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
                input=[query]
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            return []
        
        # Calculate similarities
        similarities = []
        query_vector = np.array(query_embedding).reshape(1, -1)
        
        for chunk in chunks:
            if chunk.embedding:
                chunk_vector = np.array(chunk.embedding).reshape(1, -1)
                similarity = cosine_similarity(query_vector, chunk_vector)[0][0]
                similarities.append((chunk, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        top_results = similarities[:top_k]
        
        print(f"🎯 Found {len(top_results)} relevant chunks:")
        for i, (chunk, score) in enumerate(top_results, 1):
            print(f"   {i}. {chunk.source_file} (similarity: {score:.4f})")
            print(f"      Preview: {chunk.content[:100]}...")
        
        return top_results
    
    def analyze_knowledge_base(self, chunks: List[DocumentChunk]):
        """
        Analyze the knowledge base for insights
        
        AI-102 Skill: Knowledge analysis and reporting
        """
        print(f"\n📊 Knowledge Base Analysis")
        print("=" * 50)
        
        if not chunks:
            print("❌ No chunks to analyze")
            return
        
        # Basic statistics
        total_chunks = len(chunks)
        total_content_size = sum(len(chunk.content) for chunk in chunks)
        unique_sources = set(chunk.source_file for chunk in chunks)
        
        print(f"📚 Total chunks: {total_chunks}")
        print(f"📄 Source documents: {len(unique_sources)}")
        print(f"📏 Total content size: {total_content_size:,} characters")
        print(f"⚖️  Average chunk size: {total_content_size // total_chunks:.0f} characters")
        
        # Source breakdown
        print(f"\n📋 Source breakdown:")
        source_counts = {}
        for chunk in chunks:
            source_counts[chunk.source_file] = source_counts.get(chunk.source_file, 0) + 1
        
        for source, count in sorted(source_counts.items()):
            percentage = (count / total_chunks) * 100
            print(f"   • {source}: {count} chunks ({percentage:.1f}%)")
        
        # Embedding statistics
        if chunks[0].embedding:
            embedding_dims = len(chunks[0].embedding)
            print(f"\n🔮 Embedding statistics:")
            print(f"   • Dimensions: {embedding_dims}")
            print(f"   • Model: {self.embedding_deployment}")
            print(f"   • Vector storage size: ~{(embedding_dims * total_chunks * 4) / 1024:.1f} KB")

def run_lab1_demo():
    """
    Run the complete AI-102 Lab 1 demonstration
    
    This demonstrates all key AI-102 knowledge mining concepts
    """
    print("🚀 AI-102 Lab 1: Knowledge Mining & RAG System")
    print("=" * 60)
    print("Demonstrating key exam skills:")
    print("• Document processing and chunking")
    print("• Embedding generation with Azure OpenAI") 
    print("• Vector similarity search")
    print("• Knowledge base management")
    print("• RAG system foundations")
    print("=" * 60)
    
    # Initialize the system
    km = KnowledgeMiningSystem()
    
    # Try to load existing knowledge base
    chunks = km.load_knowledge_base()
    
    # If no knowledge base exists, create one
    if not chunks:
        print("\n🔄 Creating new knowledge base...")
        
        # Load documents
        documents = km.load_documents()
        if not documents:
            print("❌ No documents found. Please add documents to data/sample_documents/")
            return
        
        # Process documents into chunks
        chunks = km.process_documents(documents)
        
        # Generate embeddings
        chunks = km.generate_embeddings(chunks)
        
        # Save knowledge base
        km.save_knowledge_base(chunks)
        km.document_chunks = chunks
    else:
        km.document_chunks = chunks
    
    # Analyze the knowledge base
    km.analyze_knowledge_base(chunks)
    
    # Demonstrate semantic search
    print(f"\n🔍 Semantic Search Demonstration")
    print("-" * 40)
    
    # Example queries for AI-102 domains
    test_queries = [
        "document processing and analysis",
        "computer vision and image recognition", 
        "artificial intelligence services",
        "machine learning capabilities"
    ]
    
    for query in test_queries:
        results = km.semantic_search(query, chunks, top_k=3)
        
        if results:
            print(f"\n🎯 Top result for '{query}':")
            best_chunk, best_score = results[0]
            print(f"   📄 Source: {best_chunk.source_file}")
            print(f"   🎯 Similarity: {best_score:.4f}")
            print(f"   📝 Content: {best_chunk.content[:200]}...")
    
    # Interactive search
    print(f"\n💬 Interactive Semantic Search")
    print("-" * 40)
    print("Enter your search queries (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\n🔍 Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            results = km.semantic_search(query, chunks, top_k=3)
            
            if results:
                print(f"\n📊 Search Results for: '{query}'")
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"\n{i}. {chunk.source_file} (Score: {score:.4f})")
                    print(f"   {chunk.content}")
            else:
                print("❌ No results found")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 AI-102 Lab 1 Complete!")
    print("✅ You've successfully demonstrated:")
    print("   • Document ingestion and processing")
    print("   • Embedding generation with Azure OpenAI")
    print("   • Vector similarity search")
    print("   • Knowledge base persistence")
    print("   • Semantic search functionality")
    print("\n🎯 Ready for AI-102 Exam Domain:")
    print("   • Knowledge mining and information extraction (15-20%)")


class AdvancedKnowledgeMining:
    """
    Advanced features for AI-102 exam scenarios
    
    Additional features that demonstrate deeper AI-102 concepts
    """
    
    def __init__(self, base_system: KnowledgeMiningSystem):
        self.base_system = base_system
        
    def document_similarity_matrix(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Create similarity matrix between all documents
        
        AI-102 Skill: Advanced document analysis and comparison
        """
        print("\n📊 Creating document similarity matrix...")
        
        if not chunks or not chunks[0].embedding:
            print("❌ No embeddings available for similarity calculation")
            return np.array([])
        
        # Group chunks by source document
        doc_groups = {}
        for chunk in chunks:
            if chunk.source_file not in doc_groups:
                doc_groups[chunk.source_file] = []
            doc_groups[chunk.source_file].append(chunk)
        
        # Calculate average embedding per document
        doc_embeddings = {}
        for doc_name, doc_chunks in doc_groups.items():
            embeddings = np.array([chunk.embedding for chunk in doc_chunks])
            avg_embedding = np.mean(embeddings, axis=0)
            doc_embeddings[doc_name] = avg_embedding
        
        # Create similarity matrix
        doc_names = list(doc_embeddings.keys())
        n_docs = len(doc_names)
        similarity_matrix = np.zeros((n_docs, n_docs))
        
        for i, doc1 in enumerate(doc_names):
            for j, doc2 in enumerate(doc_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    emb1 = doc_embeddings[doc1].reshape(1, -1)
                    emb2 = doc_embeddings[doc2].reshape(1, -1)
                    similarity = cosine_similarity(emb1, emb2)[0][0]
                    similarity_matrix[i][j] = similarity
        
        # Display results
        print("📋 Document Similarity Matrix:")
        print("   " + "".join(f"{i+1:>8}" for i in range(n_docs)))
        for i, doc_name in enumerate(doc_names):
            row = f"{i+1}. {doc_name[:15]:15}"
            for j in range(n_docs):
                row += f"{similarity_matrix[i][j]:8.3f}"
            print(row)
        
        return similarity_matrix
    
    def extract_key_topics(self, chunks: List[DocumentChunk], top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Extract key topics from the knowledge base using simple frequency analysis
        
        AI-102 Skill: Knowledge extraction and topic identification
        """
        print(f"\n🎯 Extracting top {top_k} topics...")
        
        # Simple word frequency analysis
        word_counts = {}
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 
                     'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        for chunk in chunks:
            words = chunk.content.lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 3 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top topics
        top_topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        print("📋 Top Topics:")
        for i, (topic, count) in enumerate(top_topics, 1):
            print(f"   {i:2}. {topic:20} ({count} occurrences)")
        
        return top_topics
    
    def knowledge_gaps_analysis(self, chunks: List[DocumentChunk], test_queries: List[str]) -> Dict[str, float]:
        """
        Analyze knowledge gaps by testing various queries
        
        AI-102 Skill: Knowledge base quality assessment
        """
        print(f"\n🔍 Analyzing knowledge gaps with {len(test_queries)} test queries...")
        
        gap_analysis = {}
        
        for query in test_queries:
            results = self.base_system.semantic_search(query, chunks, top_k=1)
            
            if results:
                best_score = results[0][1]
                gap_analysis[query] = best_score
            else:
                gap_analysis[query] = 0.0
        
        # Sort by score (lowest scores indicate potential gaps)
        sorted_gaps = sorted(gap_analysis.items(), key=lambda x: x[1])
        
        print("📊 Knowledge Gap Analysis (lower scores indicate potential gaps):")
        for query, score in sorted_gaps:
            status = "✅ Good" if score > 0.7 else "⚠️ Fair" if score > 0.5 else "❌ Gap"
            print(f"   {status} {query:30} (score: {score:.3f})")
        
        return gap_analysis
    
    def export_knowledge_summary(self, chunks: List[DocumentChunk], output_file: str = "knowledge_summary.md"):
        """
        Export a markdown summary of the knowledge base
        
        AI-102 Skill: Knowledge documentation and reporting
        """
        print(f"\n📝 Exporting knowledge summary to {output_file}...")
        
        summary_path = Path("data/knowledge_base") / output_file
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# AI-102 Knowledge Base Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic statistics
            f.write("## Statistics\n\n")
            f.write(f"- **Total chunks:** {len(chunks)}\n")
            f.write(f"- **Source documents:** {len(set(c.source_file for c in chunks))}\n")
            f.write(f"- **Embedding model:** {self.base_system.embedding_deployment}\n")
            f.write(f"- **Total content size:** {sum(len(c.content) for c in chunks):,} characters\n\n")
            
            # Source breakdown
            f.write("## Source Documents\n\n")
            source_counts = {}
            for chunk in chunks:
                source_counts[chunk.source_file] = source_counts.get(chunk.source_file, 0) + 1
            
            for source, count in sorted(source_counts.items()):
                f.write(f"- **{source}:** {count} chunks\n")
            
            f.write("\n## Document Contents\n\n")
            
            # Group by source and show content
            doc_groups = {}
            for chunk in chunks:
                if chunk.source_file not in doc_groups:
                    doc_groups[chunk.source_file] = []
                doc_groups[chunk.source_file].append(chunk)
            
            for source, source_chunks in sorted(doc_groups.items()):
                f.write(f"### {source}\n\n")
                for i, chunk in enumerate(source_chunks):
                    f.write(f"**Chunk {i+1}:**\n")
                    f.write(f"{chunk.content}\n\n")
        
        print(f"✅ Knowledge summary exported to: {summary_path}")


def run_advanced_demo():
    """
    Run advanced AI-102 knowledge mining demonstrations
    """
    print("\n🚀 Running Advanced AI-102 Knowledge Mining Demo")
    print("=" * 60)
    
    # Initialize systems
    km = KnowledgeMiningSystem()
    chunks = km.load_knowledge_base()
    
    if not chunks:
        print("❌ No knowledge base found. Please run the basic demo first.")
        return
    
    advanced = AdvancedKnowledgeMining(km)
    
    # Document similarity analysis
    advanced.document_similarity_matrix(chunks)
    
    # Topic extraction
    advanced.extract_key_topics(chunks)
    
    # Knowledge gaps analysis
    test_queries = [
        "machine learning algorithms",
        "data processing pipelines", 
        "computer vision models",
        "natural language processing",
        "cloud computing services",
        "artificial intelligence applications",
        "document analysis workflows",
        "image recognition systems"
    ]
    
    advanced.knowledge_gaps_analysis(chunks, test_queries)
    
    # Export summary
    advanced.export_knowledge_summary(chunks)
    
    print("\n🎉 Advanced demo complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        run_advanced_demo()
    else:
        run_lab1_demo()