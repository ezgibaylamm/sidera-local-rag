from pathlib import Path
import json

import fitz

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from src.config import (
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from src.utils import chunk_text

from src.database import (
    initialize_database,
    get_connection,
    count_chunks,
)

from src.embeddings import (
    get_embedding_model,
    generate_embedding,
)


# =========================================================
# PDF READING
# =========================================================

def read_pdf(path: Path) -> str:
    """
    Disk üzerindeki PDF'i okur.
    CLI / manuel ingestion uyumluluğu için korunur.
    """

    document = fitz.open(path)

    try:
        return "".join(
            page.get_text()
            for page in document
        )

    finally:
        document.close()


def read_pdf_bytes(
    pdf_bytes: bytes,
) -> str:
    """
    Streamlit gibi web arayüzlerinden gelen
    PDF byte verisini diske kaydetmeden okur.
    """

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        return "".join(
            page.get_text()
            for page in document
        )

    finally:
        document.close()


# =========================================================
# DATABASE HELPERS
# =========================================================

def delete_document_chunks(
    source_name: str,
) -> None:
    """
    Aynı PDF yeniden indexlenirse eski chunk'ları siler.
    Böylece önceki ingestion'dan kalan fazla chunk'lar
    veritabanında tutulmaz.
    """

    safe_name = Path(source_name).name

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM document_chunks
            WHERE source_name = ?
            """,
            (safe_name,),
        )

        connection.commit()


def clear_knowledge_base() -> None:
    """
    Tüm document chunk'larını temizler.
    Tek PDF ile çalışma modunda yeni upload öncesi kullanılır.
    """

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM document_chunks
            """
        )

        connection.commit()


def save_chunk(
    source_name: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
) -> None:

    safe_name = Path(source_name).name

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO document_chunks
            (
                source_name,
                chunk_index,
                content,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                safe_name,
                chunk_index,
                content,
                json.dumps(embedding),
            ),
        )

        connection.commit()


# =========================================================
# WEB / IN-MEMORY INGESTION
# =========================================================

def ingest_pdf_bytes(
    pdf_bytes: bytes,
    source_name: str,
    embedding_client,
    replace_knowledge_base: bool = True,
    progress_callback=None,
) -> dict:
    """
    Web arayüzünden yüklenen PDF'i:

    1. RAM'den okur
    2. Metni chunk'lara böler
    3. Her chunk için embedding üretir
    4. SQLite'a kaydeder

    PDF'in documents klasörüne manuel olarak
    kopyalanmasına gerek yoktur.
    """

    initialize_database()

    safe_name = Path(source_name).name

    text = read_pdf_bytes(
        pdf_bytes
    )

    if not text.strip():
        raise ValueError(
            "No extractable text was found in this PDF. "
            "The file may be scanned or image-based."
        )

    chunks = chunk_text(
        text,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError(
            "The PDF was read successfully, "
            "but no text chunks could be created."
        )

    # Tek belge ile sohbet varsayılan davranışıdır.
    if replace_knowledge_base:
        clear_knowledge_base()

    else:
        # Aynı isimli dosyanın eski sürümünü temizle.
        delete_document_chunks(
            safe_name
        )

    total_chunks = len(chunks)

    # Her chunk için ayrı connection açmak yerine
    # aynı transaction içinde daha verimli kaydediyoruz.
    with get_connection() as connection:

        for index, chunk in enumerate(
            chunks
        ):

            embedding = generate_embedding(
                embedding_client,
                chunk,
            )

            connection.execute(
                """
                INSERT OR REPLACE INTO document_chunks
                (
                    source_name,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    safe_name,
                    index,
                    chunk,
                    json.dumps(embedding),
                ),
            )

            if progress_callback is not None:
                progress_callback(
                    index + 1,
                    total_chunks,
                )

        connection.commit()

    return {
        "source_name": safe_name,
        "chunks": total_chunks,
        "characters": len(text),
        "stored_chunks": count_chunks(),
    }


# =========================================================
# CLI / MANUAL INGESTION
# =========================================================

def main() -> None:
    """
    Eski documents klasörü workflow'unu da korur.
    İstenirse terminalden ingest.py çalıştırılabilir.
    """

    initialize_database()

    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(
        config
    )

    manager = (
        FoundryLocalManager.instance
    )

    pdfs = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdfs:
        print(
            "No PDF files found."
        )
        return

    model = get_embedding_model(
        manager
    )

    client = (
        model.get_embedding_client()
    )

    try:

        for pdf in pdfs:

            print(
                f"\nReading: {pdf.name}"
            )

            text = read_pdf(
                pdf
            )

            chunks = chunk_text(
                text,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

            print(
                f"Chunks: {len(chunks)}"
            )

            # Aynı PDF yeniden işlenirse
            # eski chunk'ları önce temizle.
            delete_document_chunks(
                pdf.name
            )

            for index, chunk in enumerate(
                chunks
            ):

                print(
                    f"\rEmbedding chunk "
                    f"{index + 1}/{len(chunks)}",
                    end="",
                    flush=True,
                )

                embedding = generate_embedding(
                    client,
                    chunk,
                )

                save_chunk(
                    source_name=pdf.name,
                    chunk_index=index,
                    content=chunk,
                    embedding=embedding,
                )

            print()

    finally:

        model.unload()

        print(
            "Embedding model unloaded."
        )

    print(
        f"Stored chunks: {count_chunks()}"
    )


if __name__ == "__main__":
    main()