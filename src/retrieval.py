import json
import math
import re

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.database import get_connection
from src.embeddings import get_embedding_model, generate_embedding


# =========================================================
# TEXT HELPERS
# =========================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "which",
    "with",
}


def normalize_text(text: str) -> str:
    """
    Metni lexical karşılaştırma için normalize eder.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set[str]:
    """
    Anlamlı kelimeleri çıkarır.
    """

    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if token not in STOP_WORDS
        and len(token) > 1
    }


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            a * a
            for a in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            b * b
            for b in vector_b
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


# =========================================================
# LEXICAL SCORE
# =========================================================

def lexical_similarity(
    query: str,
    content: str,
) -> float:
    """
    Query'deki önemli kelimelerin chunk içinde
    ne kadar bulunduğunu hesaplar.
    """

    query_tokens = tokenize(query)
    content_tokens = tokenize(content)

    if not query_tokens:
        return 0.0

    matches = (
        query_tokens
        & content_tokens
    )

    return (
        len(matches)
        / len(query_tokens)
    )


# =========================================================
# STRUCTURAL / ANCHOR BONUS
# =========================================================

def get_anchor_adjustment(
    query: str,
    content: str,
) -> float:
    """
    Week 3, Week 4, Phase 2 gibi yapısal ifadeleri
    semantic similarity'den daha güçlü hale getirir.

    Örneğin:
        Query -> Week 3
        Chunk -> Week 3      bonus
        Chunk -> Week 4      penalty
    """

    query_lower = query.lower()
    content_lower = content.lower()

    adjustment = 0.0

    # -------------------------
    # WEEK
    # -------------------------

    query_weeks = set(
        re.findall(
            r"\bweek\s*(\d+)\b",
            query_lower,
        )
    )

    content_weeks = set(
        re.findall(
            r"\bweek\s*(\d+)\b",
            content_lower,
        )
    )

    if query_weeks:

        if query_weeks & content_weeks:
            adjustment += 0.25

        elif content_weeks:
            adjustment -= 0.18

    # -------------------------
    # PHASE
    # -------------------------

    query_phases = set(
        re.findall(
            r"\bphase\s*(\d+)\b",
            query_lower,
        )
    )

    content_phases = set(
        re.findall(
            r"\bphase\s*(\d+)\b",
            content_lower,
        )
    )

    if query_phases:

        if query_phases & content_phases:
            adjustment += 0.20

        elif content_phases:
            adjustment -= 0.12

    # -------------------------
    # EXACT IMPORTANT PHRASES
    # -------------------------

    normalized_query = normalize_text(
        query
    )

    normalized_content = normalize_text(
        content
    )

    important_patterns = [
        r"end of week \d+",
        r"week \d+",
        r"phase \d+",
    ]

    for pattern in important_patterns:

        matches = re.findall(
            pattern,
            normalized_query,
        )

        for phrase in matches:

            if phrase in normalized_content:
                adjustment += 0.08

    return adjustment




# =========================================================
# SUMMARY MODE
# =========================================================

def is_summary_query(
    query: str,
) -> bool:
    """
    Belgenin tamamını veya geniş bir bölümünü özetlemeyi
    amaçlayan sorguları tespit eder.

    Bu sorgularda yalnızca en benzer 3 chunk'ı almak yerine
    belgenin farklı bölgelerinden daha geniş context seçilir.
    """

    normalized = normalize_text(
        query
    )

    summary_phrases = [
        "summarize",
        "summary",
        "overview",
        "main topics",
        "main topic",
        "important concepts",
        "key concepts",
        "key points",
        "main points",
        "document mainly about",
        "document about",
        "what is this document about",
        "what is the document about",
        "concise overview",
        "general overview",
        "main process described",
        "main processes described",
    ]

    return any(
        phrase in normalized
        for phrase in summary_phrases
    )


def select_summary_chunks(
    results: list[dict],
    summary_k: int = 8,
) -> list[dict]:
    """
    Summary sorgularında dokümanın yalnızca tek bir bölümüne
    yoğunlaşmamak için chunk'ları source bazında farklı
    konumlardan seçer.

    Her doküman eşit aralıklı bölümlere ayrılır ve her bölümden
    hybrid rank score'u en yüksek chunk alınır.
    """

    if not results:
        return []

    grouped = {}

    for item in results:
        grouped.setdefault(
            item["source_name"],
            [],
        ).append(item)

    source_names = list(
        grouped.keys()
    )

    source_count = len(
        source_names
    )

    # Her source en az bir temsilci alsın.
    base_budget = max(
        1,
        summary_k // source_count,
    )

    remainder = max(
        0,
        summary_k - (
            base_budget * source_count
        ),
    )

    selected = []

    for source_position, source_name in enumerate(
        source_names
    ):

        source_items = sorted(
            grouped[source_name],
            key=lambda item: (
                item["chunk_index"]
            ),
        )

        budget = base_budget

        if source_position < remainder:
            budget += 1

        budget = min(
            budget,
            len(source_items),
        )

        if budget <= 0:
            continue

        # Dokümanı eşit aralıklı bucket'lara böl.
        total = len(
            source_items
        )

        for bucket_index in range(
            budget
        ):

            start = (
                bucket_index
                * total
                // budget
            )

            end = (
                (bucket_index + 1)
                * total
                // budget
            )

            bucket = source_items[
                start:end
            ]

            if not bucket:
                continue

            best_item = max(
                bucket,
                key=lambda item: (
                    item["rank_score"]
                ),
            )

            selected.append(
                best_item
            )

    # Aynı chunk yanlışlıkla iki kez seçilmesin.
    unique = []
    seen = set()

    for item in selected:

        key = (
            item["source_name"],
            item["chunk_index"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    # Yeterli temsilci çıkmadıysa en iyi kalan chunk'larla tamamla.
    if len(unique) < summary_k:

        ranked_remaining = sorted(
            results,
            key=lambda item: (
                item["rank_score"]
            ),
            reverse=True,
        )

        for item in ranked_remaining:

            key = (
                item["source_name"],
                item["chunk_index"],
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(item)

            if len(unique) >= summary_k:
                break

    # Context modeline doküman sırasına yakın bir akış ver.
    unique.sort(
        key=lambda item: (
            item["source_name"],
            item["chunk_index"],
        )
    )

    return unique[:summary_k]


# =========================================================
# QUERY EXPANSION
# =========================================================

def build_query_variants(
    query: str,
) -> list[str]:
    """
    Kullanıcının orijinal sorgusuna ek olarak
    retrieval için birkaç deterministik arama varyantı üretir.

    Amaç:
    - dolaylı soruları daha iyi yakalamak
    - özellikle "too many chunks" <-> "retrieve fewer chunks"
      gibi ters ifade edilen ilişkileri bulabilmek
    - cevap modeline dış bilgi eklememek

    Not:
    Bu varyantlar yalnızca retrieval için kullanılır.
    Chat modeline yine gerçek doküman context'i gönderilir.
    """

    normalized = normalize_text(query)

    variants = [query.strip()]

    # Genel, kısa keyword varyantı
    keyword_tokens = [
        token
        for token in normalized.split()
        if token not in STOP_WORDS
        and len(token) > 2
    ]

    if keyword_tokens:
        keyword_variant = " ".join(keyword_tokens)

        if keyword_variant not in variants:
            variants.append(keyword_variant)

    # -----------------------------------------------------
    # Chunk miktarı / performans ilişkisi
    # -----------------------------------------------------

    chunk_terms = {
        "chunk",
        "chunks",
        "retrieving",
        "retrieve",
        "retrieval",
    }

    quantity_terms = {
        "many",
        "more",
        "too",
        "large",
        "numerous",
    }

    if (
        any(term in normalized.split() for term in chunk_terms)
        and any(term in normalized.split() for term in quantity_terms)
    ):
        variants.extend(
            [
                "retrieve fewer chunks performance response time",
                "retrieving fewer chunks optimization",
                "response time retrieval chunks performance",
            ]
        )

    # -----------------------------------------------------
    # Hallucination / grounding soruları
    # -----------------------------------------------------

    if any(
        term in normalized
        for term in [
            "hallucination",
            "hallucinations",
            "fabricate",
            "fabrication",
            "grounded",
            "grounding",
            "accuracy",
            "accurate",
        ]
    ):
        variants.extend(
            [
                "rag reduced hallucinations accurate source grounded answers",
                "retrieved context reduce hallucinations",
                "document grounded answers context insufficient",
                "use retrieved document context avoid hallucinations",
            ]
        )

    # -----------------------------------------------------
    # Slow / performance soruları
    # -----------------------------------------------------

    if any(
        phrase in normalized
        for phrase in [
            "too slow",
            "slow response",
            "slow responses",
            "performance",
            "response time",
        ]
    ):
        variants.extend(
            [
                "performance debugging retrieve fewer chunks",
                "response time smaller model caching embeddings",
                "local rag performance optimization",
            ]
        )

    # -----------------------------------------------------
    # SQLite / embedding / ingestion gibi süreç soruları
    # -----------------------------------------------------

    if "ingestion" in normalized:
        variants.append(
            "document chunk embedding sqlite ingestion pipeline"
        )

    if (
        "embedding" in normalized
        and "store" in normalized
    ):
        variants.append(
            "store embeddings sqlite document chunks"
        )

    # -----------------------------------------------------
    # Duplicate temizleme
    # -----------------------------------------------------

    unique_variants = []

    for item in variants:
        cleaned = item.strip()

        if (
            cleaned
            and cleaned not in unique_variants
        ):
            unique_variants.append(cleaned)

    # Çok fazla embedding hesaplamamak için limit.
    return unique_variants[:5]


def generate_query_embeddings(
    embedding_client,
    query: str,
) -> list[tuple[str, list[float]]]:
    """
    Orijinal sorgu + query expansion varyantları için
    embedding üretir.
    """

    variants = build_query_variants(
        query
    )

    embeddings = []

    for variant in variants:

        vector = generate_embedding(
            embedding_client,
            variant,
        )

        embeddings.append(
            (
                variant,
                vector,
            )
        )

    return embeddings


def best_semantic_match(
    query_embeddings: list[
        tuple[str, list[float]]
    ],
    chunk_embedding: list[float],
) -> tuple[float, str]:
    """
    Bir chunk için tüm query varyantları arasındaki
    en yüksek cosine similarity skorunu döndürür.
    """

    best_score = -1.0
    best_query = ""

    for (
        variant,
        query_embedding,
    ) in query_embeddings:

        score = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        if score > best_score:
            best_score = score
            best_query = variant

    return (
        best_score,
        best_query,
    )


# =========================================================
# HYBRID SCORE
# =========================================================

def calculate_rank_score(
    query: str,
    content: str,
    semantic_score: float,
) -> tuple[float, float, float]:

    lexical_score = lexical_similarity(
        query,
        content,
    )

    anchor_adjustment = (
        get_anchor_adjustment(
            query,
            content,
        )
    )

    # Semantic retrieval hâlâ ana sinyal.
    # Lexical match özellikle bölüm/hafta isimlerinde
    # sıralamayı düzeltmeye yardım eder.

    rank_score = (
        semantic_score * 0.78
        + lexical_score * 0.22
        + anchor_adjustment
    )

    return (
        rank_score,
        lexical_score,
        anchor_adjustment,
    )


# =========================================================
# NEIGHBOR CONTEXT
# =========================================================

def build_context_with_neighbors(
    selected_row: dict,
    row_lookup: dict,
    window: int = 1,
) -> str:
    """
    Chunk sınırlarında bilgi kaybını azaltmak için
    seçilen chunk'ın önceki ve sonraki chunk'ını
    context'e ekler.
    """

    source_name = (
        selected_row["source_name"]
    )

    chunk_index = int(
        selected_row["chunk_index"]
    )

    parts = []

    for offset in range(
        -window,
        window + 1,
    ):

        neighbor_index = (
            chunk_index + offset
        )

        key = (
            source_name,
            neighbor_index,
        )

        neighbor = row_lookup.get(
            key
        )

        if neighbor is None:
            continue

        parts.append(
            (
                f"[Chunk {neighbor_index}]\n"
                f"{neighbor['content']}"
            )
        )

    return "\n\n".join(parts)


# =========================================================
# RETRIEVAL
# =========================================================

def get_top_chunks_with_client(
    query: str,
    embedding_client,
    top_k: int = 3,
):
    """
    Hybrid retrieval:

    1. Orijinal query + retrieval varyantları üretir
    2. Tüm query varyantlarını embed eder
    3. Her chunk için en iyi cosine similarity skorunu kullanır
    4. Lexical keyword overlap ekler
    5. Week / Phase gibi anchor ifadelerini önemser
    6. Summary sorgularında belgenin farklı bölgelerinden
       geniş context seçer
    7. Normal sorgularda en iyi chunk'ların komşularını
       context'e ekler
    """

    query_embeddings = generate_query_embeddings(
        embedding_client,
        query,
    )

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                source_name,
                chunk_index,
                content,
                embedding
            FROM document_chunks
            """
        ).fetchall()

    # ---------------------------------------------
    # Chunk lookup
    # ---------------------------------------------

    row_lookup = {}

    for row in rows:

        row_lookup[
            (
                row["source_name"],
                int(row["chunk_index"]),
            )
        ] = {
            "source_name": row["source_name"],
            "chunk_index": int(
                row["chunk_index"]
            ),
            "content": row["content"],
        }

    # ---------------------------------------------
    # Rank all chunks
    # ---------------------------------------------

    results = []

    for row in rows:

        chunk_embedding = json.loads(
            row["embedding"]
        )

        (
            semantic_score,
            matched_query,
        ) = best_semantic_match(
            query_embeddings,
            chunk_embedding,
        )

        (
            rank_score,
            lexical_score,
            anchor_adjustment,
        ) = calculate_rank_score(
            query,
            row["content"],
            semantic_score,
        )

        results.append(
            {
                "source_name": (
                    row["source_name"]
                ),
                "chunk_index": int(
                    row["chunk_index"]
                ),
                "raw_content": (
                    row["content"]
                ),

                # UI'da similarity olarak
                # göstermeye devam ediyoruz.
                "score": semantic_score,

                # Debug için:
                "rank_score": rank_score,
                "lexical_score": lexical_score,
                "anchor_adjustment": (
                    anchor_adjustment
                ),
                "matched_query": matched_query,
            }
        )

    # Semantic değil hybrid score'a göre sırala
    results.sort(
        key=lambda item: (
            item["rank_score"]
        ),
        reverse=True,
    )

    summary_mode = is_summary_query(
        query
    )

    if summary_mode:

        # Genel özet sorularında belgenin farklı bölgelerinden
        # daha geniş context kullan.
        selected = select_summary_chunks(
            results,
            summary_k=max(
                top_k,
                8,
            ),
        )

    else:

        selected = results[:top_k]

    # ---------------------------------------------
    # Context oluştur
    # ---------------------------------------------

    final_results = []

    for item in selected:

        if summary_mode:

            # Summary modunda 8 ayrı chunk zaten geniş kapsam sağlar.
            # Neighbor eklemek tekrar ve gereksiz context üretir.
            context = item[
                "raw_content"
            ]

        else:

            context = (
                build_context_with_neighbors(
                    item,
                    row_lookup,
                    window=1,
                )
            )

        final_results.append(
            {
                "source_name": (
                    item["source_name"]
                ),
                "chunk_index": (
                    item["chunk_index"]
                ),
                "content": context,

                # Mevcut chat.py / app.py uyumu
                "score": item["score"],

                # Debug / UI için
                "rank_score": (
                    item["rank_score"]
                ),
                "lexical_score": (
                    item["lexical_score"]
                ),
                "anchor_adjustment": (
                    item[
                        "anchor_adjustment"
                    ]
                ),
                "matched_query": (
                    item["matched_query"]
                ),
                "summary_mode": (
                    summary_mode
                ),
            }
        )

    return final_results


# =========================================================
# STANDALONE RETRIEVAL
# =========================================================

def get_top_chunks(
    query: str,
    manager: FoundryLocalManager,
    top_k: int = 3,
):
    """
    Standalone kullanım için embedding modelini
    kendi yükler ve işlem sonunda kapatır.
    """

    model = get_embedding_model(
        manager
    )

    client = (
        model.get_embedding_client()
    )

    try:

        return (
            get_top_chunks_with_client(
                query,
                client,
                top_k=top_k,
            )
        )

    finally:

        model.unload()

        print(
            "Embedding model unloaded."
        )


# =========================================================
# TEST
# =========================================================

def main() -> None:

    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(
        config
    )

    manager = (
        FoundryLocalManager.instance
    )

    query = (
        "Summarize this document in 5 key points."
    )

    results = get_top_chunks(
        query,
        manager,
        top_k=3,
    )

    print(
        f"\nQuery: {query}\n"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"Result {index}"
        )

        print(
            f"Source: "
            f"{result['source_name']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

        print(
            f"Semantic similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Lexical score: "
            f"{result['lexical_score']:.4f}"
        )

        print(
            f"Anchor adjustment: "
            f"{result['anchor_adjustment']:.4f}"
        )

        print(
            f"Final rank score: "
            f"{result['rank_score']:.4f}"
        )

        print(
            f"Matched retrieval query: "
            f"{result['matched_query']}"
        )

        print(
            result["content"][:1000]
        )

        print(
            "-" * 60
        )


if __name__ == "__main__":
    main()