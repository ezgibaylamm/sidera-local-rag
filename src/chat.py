import re
import time

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from src.embeddings import get_embedding_model
from src.retrieval import get_top_chunks_with_client


CHAT_MODEL_ALIAS = "phi-3.5-mini"
SIMILARITY_THRESHOLD = 0.40


def print_performance(
    retrieval_time: float,
    generation_time: float,
    total_time: float,
) -> None:
    """
    Retrieval, generation ve toplam cevap süresini
    terminalde gösterir.
    """

    print("\n[Sidera Performance]")
    print(
        f"Retrieval: {retrieval_time:.2f}s"
    )
    print(
        f"Generation: {generation_time:.2f}s"
    )
    print(
        f"Total: {total_time:.2f}s"
    )
    print()


def get_chat_model(
    manager: FoundryLocalManager,
):
    """
    Chat modelini indirir ve yükler.
    """

    print(
        f"Preparing chat model: "
        f"{CHAT_MODEL_ALIAS}"
    )

    model = manager.catalog.get_model(
        CHAT_MODEL_ALIAS
    )

    model.download(
        lambda progress: print(
            f"\rDownloading chat model: "
            f"{progress:.1f}%",
            end="",
            flush=True,
        )
    )

    print()

    model.load()

    print(
        "Chat model loaded successfully."
    )

    return model


def collect_chat_response(
    chat_client,
    messages: list[dict],
    stream_to_terminal: bool = False,
) -> str:
    """
    Streaming cevabı tek string olarak toplar.
    """

    answer_parts = []

    if stream_to_terminal:
        print(
            "\nAssistant: ",
            end="",
            flush=True,
        )

    for chunk in chat_client.complete_streaming_chat(
        messages
    ):
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if not delta:
            continue

        content = delta.content

        if not content:
            continue

        answer_parts.append(
            content
        )

        if stream_to_terminal:
            print(
                content,
                end="",
                flush=True,
            )

    if stream_to_terminal:
        print()

    return "".join(
        answer_parts
    ).strip()


# =========================================================
# SUMMARY MODE
# =========================================================

def get_requested_point_count(
    question: str,
    default: int = 5,
) -> int:
    """
    '5 key points', '4 points' gibi
    kullanıcı isteğini yakalar.
    """

    text = question.lower()

    patterns = [
        r"\b(\d+)\s+key\s+points?\b",
        r"\b(\d+)\s+main\s+points?\b",
        r"\b(\d+)\s+points?\b",
        r"\bin\s+(\d+)\s+points?\b",
        r"\bin\s+(\d+)\s+steps?\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
        )

        if match:
            count = int(
                match.group(1)
            )

            return max(
                1,
                min(
                    count,
                    10,
                ),
            )

    return default


def normalize_sentence(
    sentence: str,
) -> str:
    """
    Duplicate kontrolü için normalize eder.
    """

    return re.sub(
        r"\s+",
        " ",
        sentence.strip().lower(),
    )


def sentences_are_similar(
    sentence_a: str,
    sentence_b: str,
    threshold: float = 0.80,
) -> bool:
    """
    Chunk overlap nedeniyle gelen
    çok benzer cümleleri tespit eder.
    """

    words_a = set(
        normalize_sentence(
            sentence_a
        ).split()
    )

    words_b = set(
        normalize_sentence(
            sentence_b
        ).split()
    )

    if not words_a or not words_b:
        return False

    intersection = len(
        words_a & words_b
    )

    union = len(
        words_a | words_b
    )

    if union == 0:
        return False

    similarity = (
        intersection / union
    )

    return similarity >= threshold


def split_text_into_sentences(
    text: str,
) -> list[str]:
    """
    Chunk metnini cümlelere ayırır.

    Yarım kalan chunk parçalarını,
    bağlamsız cümleleri ve eksik
    sentence fragment'larını eler.
    """

    text = text.replace(
        "\r",
        "\n",
    )

    raw_parts = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        cleaned_line = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            line,
        ).strip()

        if not cleaned_line:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+",
            cleaned_line,
        )

        raw_parts.extend(
            parts
        )

    sentences = []

    for part in raw_parts:
        sentence = re.sub(
            r"\s+",
            " ",
            part,
        ).strip()

        # Çok kısa parçaları / başlıkları at.
        if len(sentence) < 30:
            continue

        # Chunk ortasından başlamış parçaları at.
        # Örn:
        # "mers at 07:00..."
        # "butter, and 2 kg..."
        first_character = sentence[0]

        if (
            first_character.isalpha()
            and first_character.islower()
        ):
            continue

        # Saat ifadesiyle başlayan kopmuş
        # chunk parçalarını at.
        # Örn:
        # "12:00 to prepare doughs..."
        if re.match(
            r"^\d{1,2}:\d{2}\s+to\b",
            sentence,
            re.IGNORECASE,
        ):
            continue

        # Tek başına anlamı eksik kalan
        # context-dependent cümleleri at.
        context_dependent_starts = (
            "the remaining ",
            "this ",
            "these ",
            "those ",
            "it ",
            "they ",
            "such ",
        )

        if sentence.lower().startswith(
            context_dependent_starts
        ):
            continue

        # Tamamlanmamış cümleleri at.
        if sentence[-1] not in ".!?":
            continue

        sentences.append(
            sentence
        )

    return sentences


def build_sentence_candidates(
    results: list[dict],
    max_candidates: int = 60,
) -> list[dict]:
    """
    Retrieved chunk'lardan extractive
    sentence candidate listesi oluşturur.

    Burada LLM kullanılmaz.
    """

    candidates = []
    seen = set()

    for result in results:
        sentences = split_text_into_sentences(
            result["content"]
        )

        for sentence in sentences:
            normalized = normalize_sentence(
                sentence
            )

            if normalized in seen:
                continue

            is_near_duplicate = any(
                sentences_are_similar(
                    sentence,
                    candidate["text"],
                )
                for candidate in candidates
            )

            if is_near_duplicate:
                continue

            seen.add(
                normalized
            )

            candidates.append(
                {
                    "text": sentence,
                    "source_name": result[
                        "source_name"
                    ],
                    "chunk_index": result[
                        "chunk_index"
                    ],
                }
            )

            if (
                len(candidates)
                >= max_candidates
            ):
                return candidates

    return candidates


def select_sentence_ids(
    question: str,
    candidates: list[dict],
    requested_count: int,
    chat_client,
) -> list[int]:
    """
    Phi yalnızca hangi ORİJİNAL
    cümlelerin seçileceğine karar verir.

    Cümle yazmaz veya değiştirmez.
    """

    numbered_candidates = "\n".join(
        (
            f"{index}. "
            f"{candidate['text']}"
        )
        for index, candidate in enumerate(
            candidates,
            start=1,
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are selecting sentences for an "
                "extractive document summary. "

                "Every numbered candidate is an original "
                "sentence taken from the document. "

                "Select the sentences that best represent "
                "the document overall. "

                "Prefer important recurring topics, policies, "
                "procedures, schedules, and core facts. "

                "Select at most ONE sentence from the same "
                "document chunk. "

                "Choose sentences from different chunks "
                "whenever possible to maximize topic coverage. "

                "Prefer sentences describing a complete "
                "policy, workflow, schedule, or major topic "
                "over narrow details such as a single "
                "ingredient, temperature, or isolated "
                "cleaning action. "

                "Prefer topic diversity across the selected "
                "sentences and avoid redundancy. "

                "Avoid sentence fragments, isolated minor "
                "details, metadata, filenames, page references, "
                "test descriptions, and statements about the "
                "document itself unless they are central to "
                "the document's subject. "

                "You MUST NOT rewrite, combine, paraphrase, "
                "correct, summarize, explain, or modify "
                "any candidate sentence. "

                "Return ONLY the selected sentence numbers "
                "separated by commas. "

                "Return no other words or explanations. "

                f"Select exactly {requested_count} sentences "
                "if enough useful candidates exist."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User summary request:\n"
                f"{question}\n\n"

                f"Document sentences:\n"
                f"{numbered_candidates}\n\n"

                "Return only the selected numbers."
            ),
        },
    ]

    raw_answer = collect_chat_response(
        chat_client,
        messages,
        stream_to_terminal=False,
    )

    numbers = [
        int(value)
        for value in re.findall(
            r"\d+",
            raw_answer,
        )
    ]

    selected = []
    used_chunks = set()

    for number in numbers:
        index = number - 1

        if not (
            0 <= index < len(candidates)
        ):
            continue

        if index in selected:
            continue

        candidate_chunk = candidates[
            index
        ]["chunk_index"]

        # Aynı chunk'tan mümkün olduğunca
        # sadece bir summary maddesi seç.
        if candidate_chunk in used_chunks:
            continue

        selected.append(
            index
        )

        used_chunks.add(
            candidate_chunk
        )

        if (
            len(selected)
            >= requested_count
        ):
            break

    # Yeterli farklı chunk yoksa,
    # kalan geçerli candidate'lardan tamamla.
    if (
        len(selected)
        < requested_count
    ):
        for index in range(
            len(candidates)
        ):
            if index in selected:
                continue

            selected.append(
                index
            )

            if (
                len(selected)
                >= requested_count
            ):
                break

    return selected


def build_extractive_summary(
    question: str,
    results: list[dict],
    chat_client,
) -> str:
    """
    Extractive summary:

    retrieved chunks
        ->
    original complete sentences
        ->
    fragment cleanup
        ->
    duplicate cleanup
        ->
    model chooses sentence IDs
        ->
    original sentences displayed unchanged
    """

    candidates = build_sentence_candidates(
        results
    )

    if not candidates:
        return (
            "I don't know based on the "
            "provided documents."
        )

    requested_count = get_requested_point_count(
        question,
        default=5,
    )

    requested_count = min(
        requested_count,
        len(candidates),
    )

    selected_ids = select_sentence_ids(
        question,
        candidates,
        requested_count,
        chat_client,
    )

    selected_candidates = [
        candidates[index]
        for index in selected_ids
    ]

    # Burada LLM yeni metin üretmez.
    # PDF'den gelen orijinal cümleler
    # değiştirilmeden gösterilir.
    answer = "\n".join(
        (
            f"{index}. "
            f"{candidate['text']}"
        )
        for index, candidate in enumerate(
            selected_candidates,
            start=1,
        )
    )

    print(
        "\nAssistant:\n"
        f"{answer}"
    )

    return answer


# =========================================================
# NORMAL RAG ANSWER
# =========================================================

def answer_query(
    question: str,
    embedding_client,
    chat_client,
) -> tuple[str, list[dict]]:
    """
    Normal soru:
    retrieval -> LLM answer

    Summary:
    retrieval
        -> original sentences
        -> fragment cleanup
        -> duplicate cleanup
        -> sentence selection
        -> extractive summary

    Her sorguda retrieval, generation ve toplam süre
    terminale yazdırılır.
    """

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()

    results = get_top_chunks_with_client(
        question,
        embedding_client,
        top_k=3,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if not results:
        total_time = (
            time.perf_counter()
            - total_start
        )

        print_performance(
            retrieval_time,
            0.0,
            total_time,
        )

        return (
            "I don't know based on the "
            "provided documents.",
            [],
        )

    summary_mode = any(
        result.get(
            "summary_mode",
            False,
        )
        for result in results
    )

    best_score = results[0]["score"]

    # Normal sorularda similarity threshold.
    if (
        not summary_mode
        and best_score
        < SIMILARITY_THRESHOLD
    ):
        total_time = (
            time.perf_counter()
            - total_start
        )

        print_performance(
            retrieval_time,
            0.0,
            total_time,
        )

        return (
            "I don't know based on the "
            "provided documents.",
            [],
        )

    # =====================================================
    # EXTRACTIVE SUMMARY
    # =====================================================

    if summary_mode:
        generation_start = (
            time.perf_counter()
        )

        answer = build_extractive_summary(
            question,
            results,
            chat_client,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        print_performance(
            retrieval_time,
            generation_time,
            total_time,
        )

        return (
            answer,
            results,
        )

    # =====================================================
    # NORMAL DOCUMENT Q&A
    # =====================================================

    context = "\n\n".join(
        (
            f"[Source: "
            f"{result['source_name']}, "
            f"Chunk: "
            f"{result['chunk_index']}]\n"
            f"{result['content']}"
        )
        for result in results
    )

    main_steps_rule = ""

    if (
        "main steps"
        in question.lower()
    ):
        main_steps_rule = (
            "\nIMPORTANT FORMAT RULE: "
            "The user asked for the main steps. "
            "Return AT MOST 5 numbered steps. "
            "Never output more than 5 numbered items. "
            "Include only essential steps. "
            "Do not include optional steps, testing, "
            "setup scripts, UI alternatives, or "
            "implementation suggestions."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Sidera, a local "
                "document-grounded assistant. "

                "Answer using only the provided context. "

                "Do not introduce outside facts, "
                "assumptions, or invented details. "

                "You may make a simple inference only "
                "when it follows directly from an "
                "explicit statement in the context. "

                "If only part of the question is "
                "supported, answer only that part. "

                "If the context does not contain "
                "enough information, respond exactly with: "

                "\"I don't know based on the "
                "provided documents.\" "

                "Keep the answer concise and "
                "directly relevant. "

                f"{main_steps_rule}\n\n"

                f"Context:\n{context}"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    generation_start = (
        time.perf_counter()
    )

    answer = collect_chat_response(
        chat_client,
        messages,
        stream_to_terminal=True,
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    print_performance(
        retrieval_time,
        generation_time,
        total_time,
    )

    return (
        answer,
        results,
    )

def print_sources(
    results: list[dict],
) -> None:
    """
    Kullanılan retrieval kaynaklarını gösterir.
    """

    if not results:
        return

    print("\nSources:")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{result['source_name']} "
            f"(Chunk "
            f"{result['chunk_index']}, "
            f"Similarity "
            f"{result['score']:.4f})"
        )

    print()


def main() -> None:
    """
    Terminal Local RAG chatbot.
    """

    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(
        config
    )

    manager = (
        FoundryLocalManager.instance
    )

    print(
        "\nPreparing Local RAG Assistant...\n"
    )

    embedding_model = get_embedding_model(
        manager
    )

    embedding_client = (
        embedding_model.get_embedding_client()
    )

    chat_model = get_chat_model(
        manager
    )

    chat_client = (
        chat_model.get_chat_client()
    )

    print(
        "\n=============================="
    )
    print(
        "     Local RAG Assistant"
    )
    print(
        "=============================="
    )
    print(
        "Ask questions about your documents."
    )
    print(
        "Type 'exit' to quit.\n"
    )

    try:
        while True:
            question = input(
                "You: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
            }:
                print(
                    "\nGoodbye!"
                )
                break

            if not question:
                print(
                    "Please enter a question.\n"
                )
                continue

            try:
                answer, sources = answer_query(
                    question,
                    embedding_client,
                    chat_client,
                )

                if not sources:
                    print(
                        f"\nAssistant: "
                        f"{answer}\n"
                    )
                    continue

                print_sources(
                    sources
                )

            except Exception as error:
                print(
                    f"\nError: "
                    f"{error}\n"
                )

    finally:
        chat_model.unload()
        embedding_model.unload()

        print(
            "\nModels unloaded."
        )


if __name__ == "__main__":
    main()