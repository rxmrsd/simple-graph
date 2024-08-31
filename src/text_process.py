"""text_process.py"""
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """句読点も句切り文字に含めるようにするためのスプリッタ"""

    def __init__(self, **kwargs: Any) -> None:
        """Constract"""
        separators = [
            "\n\n",
            "\n",
            "。",
            # "、",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)


def doc_splitter(doc_path: Path) -> list:
    """Document splitter

    Args:
        doc_path (Path): document path

    Returns:
        list: texts
    """
    with doc_path.open() as f:
        state_of_the_union = f.read()
    text_splitter = JapaneseCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    texts = text_splitter.create_documents([state_of_the_union])
    for i, text in enumerate(texts):
        text.id = str(i)

    return texts
