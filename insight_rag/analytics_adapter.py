"""Present document evidence in the shared question UI without inventing SQL results."""
def analytics_answer(result, question):
    answer = result.get("answer") or "No supported answer was returned."
    supported = result.get("evidence_status") == "supported"
    sources = result.get("sources") or []
    return {**result, "status": "success", "result_status": "success",
            "user_goal": question, "answer": answer, "message": answer, "story": answer,
            "analysis_source": "document_rag", "data": [], "columns": [], "row_count": 0,
            "sql_used": None, "result_valid": supported, "confidence": "unverified",
            "provenance": {"source_type": "document", "sources": sources,
                           "methodology": "Document RAG with page citations; no SQL execution."},
            "evidence": sources, "findings": [], "suggested_questions": [],
            "limitations": [] if supported else ["Document evidence was not fully verified."],
            "verdict": {"verified": False, "checks": [], "warnings": []}}
