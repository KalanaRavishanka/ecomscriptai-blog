---
title: "Data Ingestion for E-commerce RAG: Vectorizing Product Catalogs with Pinecone"
date: 2026-09-16T12:00:00+01:00
description: "Learn how to process E-commerce catalogs, generate OpenAI embeddings, and safely batch-upload them into Pinecone for Hybrid Search RAG pipelines."
draft: false
tags: ["RAG", "Data Engineering", "Pinecone", "Python", "Vector Database"]
categories: ["Artificial Intelligence", "Data Science"]
ShowReadingTime: true
ShowPostNavLinks: true
---

In our previous post, we architected a production-ready RAG pipeline that uses Hybrid Search to fetch accurate e-commerce data and answer customer queries. However, a retrieval engine is completely useless without the actual data. 

How do we take a massive retail catalog—containing thousands of products, JSON structures, descriptions, and dynamic stock levels—and convert it into a format that our AI can understand? 

This process is called **Data Ingestion and Vectorization**. In this guide, we will build a robust Python script to process product data, generate OpenAI embeddings, and safely batch-upload them into a Pinecone vector database.

---

## 1. The Ingestion Architecture

Ingesting data for a production e-commerce system is not as simple as uploading a single PDF file. Retail catalogs are structured data (usually JSON or CSV from a database like MongoDB or PostgreSQL). We need to extract the semantic text for the AI to "read" while preserving the exact product details (SKUs, price, category) as Metadata.

![Vector Ingestion Flow](/images/vector-ingestion-flow.jpg)
*Figure 1: The Data Ingestion Flow: Extracting JSON catalogs, converting descriptions to dense vectors via OpenAI, and uploading to Pinecone with attached metadata.*

---

## 2. The Bottleneck: Why Batch Processing is Critical

If your store has 10,000 products, sending all 10,000 to the OpenAI Embeddings API at the exact same time will result in an immediate `429 Too Many Requests` rate-limit error. Furthermore, pushing a payload that massive into Pinecone in a single request will crash your pipeline.

> **The Solution: Chunking and Batch Upserts.**
> We must slice our dataset into smaller, manageable batches (e.g., 100 products at a time). We process a batch, generate the vectors, push them to Pinecone, and move to the next. This ensures system stability and fault tolerance.

---

## 3. Step-by-Step Python Implementation

Let's write the ingestion script. We will assume you have exported your product catalog into a simple JSON file format.

### Prerequisites

Ensure you have your `.env` file configured with your API keys as discussed in the previous article.

```env
OPENAI_API_KEY="sk-your-openai-api-key-here"
PINECONE_API_KEY="your-pinecone-api-key-here"
```

### The Ingestion Script (`ingest_catalog.py`)

Below is the complete, production-grade script to process a JSON catalog and upload it to Pinecone using LangChain.

```python
import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# 1. Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ecommerce-catalog"

# 2. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index if it doesn't exist (Dimension 1536 for OpenAI small embeddings)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
# 3. Load E-commerce Data (Mock JSON Example)
# In production, this data would be fetched via an API or SQL query.
mock_catalog = [
    {
        "sku": "SKU-A109",
        "name": "Sony WH-1000XM5 Wireless Headphones",
        "description": "Industry-leading noise canceling Bluetooth headphones with multipoint connection and 30-hour battery life.",
        "category": "electronics",
        "price": 348.00,
        "status": "active"
    },
    {
        "sku": "SKU-B202",
        "name": "Ergonomic Office Chair",
        "description": "Mesh back office chair with adjustable lumbar support and 3D armrests. Perfect for long coding sessions.",
        "category": "furniture",
        "price": 199.99,
        "status": "out_of_stock"
    }
]

# 4. Transform Data into LangChain Documents
documents = []
for item in mock_catalog:
    # The 'page_content' is what the AI reads to understand the product
    semantic_content = f"Product Name: {item['name']}\nDescription: {item['description']}"
    
    # The 'metadata' is crucial for Hybrid Search and Exact Matching
    metadata = {
        "sku": item["sku"],
        "category": item["category"],
        "price": item["price"],
        "status": item["status"]
    }
    
    doc = Document(page_content=semantic_content, metadata=metadata)
    documents.append(doc)

# 5. Generate Embeddings and Batch Upload
print(f"Starting vectorization for {len(documents)} products...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# We use PineconeVectorStore.from_documents which automatically handles 
# the chunking, embedding generation, and Pinecone upserts under the hood.
PineconeVectorStore.from_documents(
    documents, 
    embeddings, 
    index_name=INDEX_NAME
)

print("Data Ingestion Complete! Your catalog is now ready for Hybrid Search.")
```

---

## 4. Understanding the Metadata Strategy

In the code above, notice how we separate `page_content` from `metadata`. 

We do **not** put the price or stock status in the `page_content` text block. Why? Because prices and stock change constantly. If you embed "Price: $348" into the vector, you have to completely delete and re-embed the vector if the price drops to $299. 

By storing dynamic attributes in the `metadata` dictionary, you can easily update those specific fields in Pinecone via a lightweight API call without recalculating the expensive OpenAI text embeddings!

## Conclusion

You now have a fully functioning RAG architecture. You can dynamically ingest your product catalog, attach filterable metadata, and retrieve contextual answers to power an intelligent customer support agent. 

By handling batch processing and understanding the separation of semantic text vs metadata, your pipeline is now ready to scale to thousands of products without crashing.