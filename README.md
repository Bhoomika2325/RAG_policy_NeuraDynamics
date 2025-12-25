## Policy Question Answering System (RAG-based)
### Project Overview
This project is a policy-aware question answering system built using Retrieval-Augmented Generation (RAG).
The system answers user questions strictly based on company policy documents and is designed to avoid hallucinations, make clear decisions, and allow easy evaluation.

The goal of this project is not to build a chatbot UI, but to demonstrate prompt engineering, retrieval accuracy, reasoning, and evaluation—which are the core skills required for this internship.

## Policy Data Used
**Amazon Return & Refund Policy (PDF)**

This policy was chosen because it is a real-world, publicly available document that
includes general return rules, brand-specific exceptions (Apple), defective item handling,
and refund timelines.

## System Architecture
The system follows a simple and robust pipeline:

1. **Document Loading & Chunking using Langchain**
   - PDFs are loaded and split into chunks of 400 characters with overlap.
   - This improves retrieval precision while preserving context.

2. **Vector Storage & Retrieval**
   - Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.
   - FAISS is used for vector storage.
   - For each query, the top 4 most relevant chunks (`top_k = 4`) are retrieved.

3. **RAG-Based Answer Generation**
   - Retrieved policy text is injected into a carefully designed prompt.
   - The LLM is instructed to answer only using the provided context.

4. **Controlled Answer Generation (LangChain + LangGraph)**
   - Retrieved policy text is injected into a carefully designed prompt.
   - The LLM is instructed to answer only using the retrieved context.
   - Structured outputs (answer, action, action_input) are produced for reliable evaluation.
---

## Prompt & Decision Design
Instead of free-form answers, the system produces structured outputs:

- **answer** – grounded response from the policy text
- **action** – whether a policy applies (`APPLY_POLICY` or `NO_INFO`)
- **action_input** – the policy category (e.g., return window, defective item)

This makes the system easier to evaluate and reduces hallucinations.

---
## Promptv1 vs Promotv2
I Intentionally compared to types of prompts, here are the excerpts of the two prompts:

**Prompt v1 (Excerpt)**:

`Answer the user’s question using the provided policy context.
If the information is not present, say so clearly`

### Limitations Observed
- Answers were inconsistent across runs
- No explicit signal when a policy did not apply
- Difficult to evaluate programmatically
- Occasional over-generalization when context was weak

**Prompt v2 (Excerpt)**:

`Decide whether the policy explicitly applies.
If it applies, return a grounded answer and classify the policy.
If not, clearly return NO_INFO.
Output structured JSON only.`

#### Why Prompt v2 Is Better

Prompt v2 transforms the system from a chat-style assistant into a policy decision engine.
This makes the system safer, easier to evaluate, and more suitable for real-world policy and compliance use cases.

---
## Why LangChain and LangGraph?
- LangChain simplifies document ingestion, embedding, retrieval, and prompt management.
- LangGraph provides explicit control over the RAG workflow and state transitions,
making the system easier to debug, evaluate, and extend.
---
## Hallucination Control & Confidence
The system does not expose numeric confidence scores. Instead, confidence is expressed
through decisions:

- `APPLY_POLICY` → the policy explicitly supports the answer
- `NO_INFO` → the policy does not cover the question

This conservative approach is suitable for policy-based systems.

---
## Evaluation Strategy
Evaluation is performed using a fixed set of diverse questions, including:
- Clearly answerable questions
- Brand-specific cases
- Defective item scenarios
- Cancellation-related queries
- Out-of-scope questions

This allows consistent and repeatable evaluation of retrieval and grounding quality.

---
## Challenges & Solutions
- **Structured output limitations** → The groq based models did not support structured outputs, so I enforced JSON via prompt + manual parsing
- **Label inconsistency** → constrained outputs to a fixed policy names
- **Hallucination risk** → strict grounding and explicit refusal rules

---

## Why This Approach
This project prioritizes:
- Correctness over creativity
- Grounded answers over speculation
- Evaluation clarity over UI complexity

These choices reflect how real-world policy and compliance systems are built.

---

## Conclusion
This project demonstrates a practical approach to building a reliable RAG-based
policy assistant with strong hallucination control and clear evaluation design.

# RAG_policy_NeuraDynamics
