# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
import re

# Granite Guardian model family identifier
GUARDIAN_MODEL_FAMILY = "granite-guardian"

# Output parsing: 3.3 uses <score>yes</score> / <score>no</score>; 3.2 uses plain Yes / No
V33_SCORE_RE = re.compile(r"<score>\s*(yes|no)\s*</score>", re.IGNORECASE)
V32_LABEL_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

# Required inference settings enforced regardless of caller config
REQUIRED_TEMPERATURE = 0.0
REQUIRED_MAX_TOKENS = 20  # sufficient for <score>yes</score> plus optional whitespace
REQUIRED_MAX_TOKENS_WITH_THINK = (
    1024  # fallback when think=True and user does not specify max_tokens
)
REQUIRED_LOGPROBS = 4
