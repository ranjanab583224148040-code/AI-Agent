"""
Document Ingestion CLI Script
=============================================================================
Scans documents/ directory (PDF, TXT, DOCX, Markdown), chunks documents,
generates embeddings using Gemini, and stores them in a persistent Chroma DB.

Usage:
    python ingest.py
    python ingest.py --docs-dir my_docs --chunk-size 800 --chunk-overlap 150
=============================================================================
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding for Windows command line / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from rag_pipeline import ingest_documents


def main():
    parser = argparse.ArgumentParser(
        description="Ingest local documents (PDF, TXT, DOCX, MD) into Chroma Vector Store for RAG."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(config.DOCUMENTS_DIR),
        help=f"Directory containing documents to ingest (default: {config.DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear existing vector database before indexing (default: clears to prevent duplicates)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Override chunk size in characters (default from config: {config.CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"Override chunk overlap in characters (default from config: {config.CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📚 DOCUMENT INGESTION PIPELINE (RAG)")
    print("=" * 70)
    print(f"  • Documents Folder : {args.docs_dir}")
    print(f"  • Vector Store Dir : {config.CHROMA_PERSIST_DIR}")
    print(f"  • Embedding Model  : {config.EMBEDDING_MODEL}")
    print(f"  • Chunk Size       : {args.chunk_size or config.CHUNK_SIZE}")
    print(f"  • Chunk Overlap    : {args.chunk_overlap or config.CHUNK_OVERLAP}")
    print(f"  • Reset On Ingest  : {not args.keep_existing}")
    print("=" * 70 + "\n")

    docs_path = Path(args.docs_dir).resolve()
    if not docs_path.exists():
        print(f"Creating documents folder at: {docs_path}")
        docs_path.mkdir(parents=True, exist_ok=True)

    try:
        total_chunks = ingest_documents(
            docs_dir=docs_path,
            clear_existing=not args.keep_existing,
        )
        if total_chunks > 0:
            print("\n" + "=" * 70)
            print("🎉 Ingestion complete! The documents are now ready for RAG search.")
            print(f"   Indexed chunks : {total_chunks}")
            print(f"   Vector DB path : {config.CHROMA_PERSIST_DIR}")
            print("   You can now run 'python langchain_agent.py' and ask questions about your docs!")
            print("=" * 70)
        else:
            print("\n⚠️ No documents were indexed. Please place .pdf, .txt, .docx, or .md files in the folder and retry.")
    except Exception as exc:
        print(f"\n❌ Ingestion failed with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
