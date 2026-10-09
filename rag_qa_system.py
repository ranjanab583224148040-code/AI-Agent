import os
import sys
import re
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PDF_PATH = r"C:\Users\Admin\Downloads\Dummy_Student_Database_for_RAG.pdf"

# Global parsed database for instantaneous offline fallback
PARSED_STUDENTS = []

def parse_student_pdf(file_path: str):
    global PARSED_STUDENTS
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    tokens = [line.strip() for line in full_text.splitlines() if line.strip()]
    
    docs = []
    docs.append(Document(
        page_content="Full Student Database:\n" + full_text,
        metadata={"type": "full_database", "source": file_path}
    ))
    
    PARSED_STUDENTS = []
    i = 0
    while i < len(tokens):
        if re.match(r"^STU\d+$", tokens[i]):
            if i + 12 <= len(tokens):
                rec = tokens[i:i+12]
                student_dict = {
                    "student_id": rec[0],
                    "name": rec[1],
                    "dept": rec[2],
                    "year": rec[3],
                    "email": rec[4],
                    "phone": rec[5],
                    "attendance": rec[6],
                    "python": int(rec[7]) if rec[7].isdigit() else rec[7],
                    "dsa": int(rec[8]) if rec[8].isdigit() else rec[8],
                    "ai": int(rec[9]) if rec[9].isdigit() else rec[9],
                    "overall": int(rec[10]) if rec[10].isdigit() else rec[10],
                    "placement": rec[11]
                }
                PARSED_STUDENTS.append(student_dict)
                student_info = (
                    f"Student ID: {rec[0]} | Name: {rec[1]} | Department: {rec[2]} | "
                    f"Year: {rec[3]} | Email: {rec[4]} | Phone: {rec[5]} | "
                    f"Attendance: {rec[6]} | Python: {rec[7]} | DSA: {rec[8]} | "
                    f"AI: {rec[9]} | Overall: {rec[10]} | Placement: {rec[11]}"
                )
                docs.append(Document(
                    page_content=student_info,
                    metadata={"student_id": rec[0], "name": rec[1], "dept": rec[2], "source": file_path}
                ))
                i += 12
                continue
        i += 1
        
    return docs

def offline_fallback_query(query: str) -> str:
    """Answers queries directly from the parsed records without consuming API quota."""
    q_lower = query.lower()
    
    # 1. Search by student name or student ID
    for s in PARSED_STUDENTS:
        if s["name"].lower() in q_lower or s["student_id"].lower() in q_lower:
            return (
                f"### Student Record: {s['name']} ({s['student_id']})\n"
                f"- **Department**: {s['dept']}\n"
                f"- **Year**: {s['year']}\n"
                f"- **Email**: {s['email']}\n"
                f"- **Phone**: {s['phone']}\n"
                f"- **Attendance**: {s['attendance']}\n"
                f"- **Python**: {s['python']}\n"
                f"- **DSA**: {s['dsa']}\n"
                f"- **AI**: {s['ai']}\n"
                f"- **Overall Mark**: {s['overall']}\n"
                f"- **Placement Status**: {s['placement']}"
            )
            
    # 2. Highest overall
    if "highest" in q_lower and ("overall" in q_lower or "mark" in q_lower):
        top = max(PARSED_STUDENTS, key=lambda x: int(x["overall"]) if str(x["overall"]).isdigit() else 0)
        return f"**{top['name']}** ({top['student_id']}, {top['dept']}) has the highest overall mark of **{top['overall']}**."

    # 3. Highest attendance
    if "highest" in q_lower and "attendance" in q_lower:
        top_att = max(PARSED_STUDENTS, key=lambda x: int(x["attendance"].replace("%", "")) if "%" in str(x["attendance"]) else 0)
        return f"**{top_att['name']}** ({top_att['student_id']}) has the highest attendance at **{top_att['attendance']}**."

    # 4. Placement eligible count
    if "placement" in q_lower and ("eligible" in q_lower or "how many" in q_lower):
        eligible = [s for s in PARSED_STUDENTS if s["placement"].lower() == "eligible"]
        return f"There are **{len(eligible)}** placement eligible students (out of {len(PARSED_STUDENTS)} total)."

    # 5. Department filter
    for dept in ["cse", "it", "ece", "eee", "aids"]:
        if dept in q_lower:
            matched = [s for s in PARSED_STUDENTS if s["dept"].lower() == dept]
            names = [f"{s['name']} ({s['student_id']})" for s in matched]
            return f"Students in **{dept.upper()}** ({len(matched)} total):\n" + "\n".join(f"- {n}" for n in names)

    # Generic search: return top matching records
    matches = []
    for s in PARSED_STUDENTS:
        record_str = f"{s['student_id']} {s['name']} {s['dept']} {s['email']} {s['placement']}"
        if any(word in record_str.lower() for word in q_lower.split() if len(word) > 2):
            matches.append(s)
            
    if matches:
        res = [f"- **{m['name']}** ({m['student_id']}, {m['dept']}): Overall {m['overall']}, Placement: {m['placement']}" for m in matches[:5]]
        return "Matching records found:\n" + "\n".join(res)

    return "No matching records found in the student database."

def build_rag_system(file_path: str):
    docs = parse_student_pdf(file_path)
    
    # Try setting up online vector store and LLM
    rag_chain = None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 25})

        # Use gemini-3.5-flash-lite to minimize quota consumption
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)

        prompt = ChatPromptTemplate.from_template("""You are an expert Question-Answering RAG system for the Student Database.
Use the following retrieved context to answer the user question accurately, completely, and concisely.

Context:
{context}

Question:
{question}

Provide a direct, factual answer:""")

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        print(f"[Notice] Online RAG setup encountered an issue: {e}. Defaulting to offline engine.")

    def query(user_question: str):
        if rag_chain is not None:
            try:
                return rag_chain.invoke(user_question)
            except Exception as e:
                # Quota exceeded or network error -> use offline engine automatically!
                print(f"[Notice] Online API limit reached (429/Quota). Using built-in local retrieval...")
                return offline_fallback_query(user_question)
        else:
            return offline_fallback_query(user_question)

    return query

if __name__ == "__main__":
    query_fn = build_rag_system(PDF_PATH)
    
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        print(f"\n[Query]: {user_query}")
        print(f"[Answer]:\n{query_fn(user_query).strip()}\n")
    else:
        print("=" * 65)
        print("  Student Database RAG System (Online & Offline Resilient)")
        print("=" * 65)
        print("Ready! Type your question below (or 'exit' to quit):\n")
        while True:
            try:
                user_q = input("Ask a question: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                print(f"\n[Answer]:\n{query_fn(user_q).strip()}\n")
            except (KeyboardInterrupt, EOFError):
                break
