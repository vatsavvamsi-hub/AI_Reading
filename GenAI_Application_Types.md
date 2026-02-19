# Generative AI Application Types & Architecture Patterns

## 1. RAG Chatbot
**Pattern**: Retrieval + Generation + Conversation  
**Flow**: User query → Vector search → Retrieved context + query → LLM → Response  
**Use Cases**: Customer support, documentation Q&A, knowledge bases

## 2. Agentic Systems / AI Agents
**Pattern**: Reasoning Loop + Tool Use  
**Flow**: Task → Tool selection → Execution → Evaluation → Repeat until complete  
**Use Cases**: Autonomous task execution, complex workflows, research automation

## 3. Content Generation
**Pattern**: Prompt Engineering + Output Formatting  
**Flow**: Structured prompts → LLM → Post-processing → Formatted output  
**Use Cases**: Code generators, article writers, marketing copy

## 4. Classification & Information Extraction
**Pattern**: Structured Input → LLM → Structured Output  
**Flow**: Documents/text → LLM with schema → Parsed results  
**Use Cases**: Sentiment analysis, entity extraction, document classification

## 5. Search & Ranking Systems
**Pattern**: Query Understanding → Retrieval → Re-ranking  
**Flow**: NL query → Multiple retrieval methods → LLM re-ranking → Results  
**Use Cases**: Advanced search, recommendation systems, relevance ranking

## 6. Summarization Applications
**Pattern**: Input Processing → Chunking → Summarization → Aggregation  
**Flow**: Large documents → Split chunks → Summarize each → Combine  
**Use Cases**: Document summarization, meeting notes, report generation

## 7. Code Generation & Developer Tools
**Pattern**: Context Aware Generation + Validation  
**Flow**: Code context → LLM → Generated code → Linting/testing → Feedback loop  
**Use Cases**: IDE plugins, code completion, test generation

## 8. Question Answering Systems
**Pattern**: Multi-step Reasoning  
**Flow**: Question → Decompose → Retrieve per sub-question → Synthesize → Answer  
**Use Cases**: Complex Q&A, fact verification, research tools

## 9. Recommendation Systems
**Pattern**: User Context + Item Matching + Personalization  
**Flow**: User profile → Candidate retrieval → LLM ranking → Recommendations  
**Use Cases**: Product recommendations, content discovery

## 10. Multi-Modal Applications
**Pattern**: Input Integration + Processing + Generation  
**Flow**: Image/audio/text → Respective models → LLM coordination → Output  
**Use Cases**: Image analysis, voice assistants, document analysis

---

## Quick Comparison

| Type | Primary Goal | Complexity | Key Component |
|------|--------------|------------|---------------|
| RAG Chatbot | Answer from knowledge base | Medium | Vector DB + LLM |
| Agent | Execute tasks autonomously | High | Tool orchestration |
| Content Gen | Create new content | Low-Medium | Prompt engineering |
| Classification | Categorize data | Low-Medium | Output schema |
| Code Gen | Generate/fix code | High | Context management |
