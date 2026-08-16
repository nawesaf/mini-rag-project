from time import perf_counter

import multi_modal_rag as rag_v1
import multi_modal_rag_v2 as rag_v2


PDF_PATH = "docs/test1.pdf"
DOCUMENT_ID = "comparison-test1"

QUESTIONS = [
    "Quel est le sujet principal du document ?",
    "Quels résultats chiffrés importants sont présentés ?",
    "Que montre le principal tableau du document ?",
    "Décris la figure ou le diagramme le plus important.",
]

QUESTIONS2 = [
    "Quels sont les trois composants principaux du système PIER-QA ?",
    "quel modèle obtient la meilleure Similarity et quelle est sa valeur ?",
    "à quoi servent Elasticsearch et Docstore ?",
    "quelles sont les valeurs de lora_r, lora_alpha et lora_dropout ?",
]


def reset_document(rag) -> None:
    """Remove chunks left by an earlier comparison run."""
    try:
        rag.delete_document(DOCUMENT_ID)
    except Exception as exc:
        print(f"Aucun index précédent à supprimer : {exc}")


def test_rag(name, rag, questions: list[str] | None = None) -> None:
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    questions = QUESTIONS2 if questions is None else questions

    reset_document(rag)
    rag.chat_history.clear()

    start = perf_counter()
    documents = rag.run_complete_ingestion_pipeline(
        PDF_PATH,
        DOCUMENT_ID,
    )
    ingestion_duration = perf_counter() - start

    chunk_count = len(documents) if documents is not None else "non retourné"
    print(f"Chunks indexés : {chunk_count}")
    print(f"Durée d'ingestion : {ingestion_duration:.2f} s")

    for question in questions:
        # Each question is independent, so history cannot favor one RAG.
        rag.chat_history.clear()

        start = perf_counter()
        answer, rewritten_query = rag.get_answer(
            question,
            DOCUMENT_ID,
        )
        answer_duration = perf_counter() - start

        print(f"\nQuestion : {question}")
        print(f"Requête utilisée : {rewritten_query}")
        print(f"Réponse : {answer}")
        print(f"Durée : {answer_duration:.2f} s")


def test_specific_query(query: str) -> None:
    """Index the test PDF and compare both RAGs on one query."""
    if not query.strip():
        raise ValueError("La requête ne peut pas être vide.")

    questions = [query]
    # test_rag("RAG v1", rag_v1, questions)
    test_rag("RAG v2", rag_v2, questions)


def main() -> None:
    # test_rag("RAG v1", rag_v1)
    test_rag("RAG v2", rag_v2)
    # test_specific_query("quelles sont les valeurs de lora_r, lora_alpha et lora_dropout ?")

if __name__ == "__main__":
    main()
