# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List


def render_documents(documents: List[dict]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        text = doc.get("text", "")
        doc_id = doc.get("id", str(i))
        parts.append(f"[Document {i} | id={doc_id}]\n{text}")
    return "\n\n".join(parts) if parts else "(no documents provided)"
