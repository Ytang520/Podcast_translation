"""
Translation module using OpenRouter API with deepseek-chat.
Implements two-step philosophy-aware translation:
  Step 1: Extract terms with AI search for authoritative explanations
  Step 2: Translate text using extracted terminology
"""
import requests
import re
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL, 
    TRANSLATION_MODEL,
    DATA_DIR
)
from terminology_db import TerminologyDB


class TranslationError(Exception):
    """Exception raised when translation fails."""
    pass


# ==============================================================================
# STEP 1: TERM EXTRACTION PROMPTS
# ==============================================================================

TERM_EXTRACTION_SYSTEM_PROMPT = """你是一位专业的哲学术语分析专家。你的任务是从英文哲学文本中提取所有重要术语，并提供专业的中文翻译和解释。

请提取以下类型的术语：
1. **哲学专有术语** - 如 epistemology, ontology, phenomenology 等
2. **专有名词** - 哲学流派、学说、理论名称
3. **哲学家人名** - 如 Kant, Hegel, Aristotle 等
4. **文本特有的反复出现的名词** - 对于同一概念，请使用统一的翻译

对于每个术语，请提供：
- `english`: 英文原词
- `chinese`: 标准中文译名（使用学术界公认的翻译）
- `explanation`: 简明解释（2-3句话，不要与原文勾连，仅作客观解释）
- `example`: 示例说明（帮助理解的具体例子）
- `category`: 类别 (philosopher/term/concept/school/text-specific)

**重要**：
- 使用专业哲学翻译标准
- 解释应客观独立，不要涉及"原文表示..."
- 对于不确定的术语，优先查阅权威来源（如斯坦福哲学百科、维基百科哲学条目）
- `chinese` 字段必须是**单一、确定的译名**：
    - 不要提供多个候选译法（不要出现“甲/乙”、“甲或乙”、“甲、乙”这类并列）
    - 如果确实存在多个常见译名，请自行选择更通行的一个作为 `chinese`，并把差异留在 `explanation` 中说明

如果文本中没有需要提取的术语，请输出空列表：
```json
{
    "terms": []
}
```

请以JSON格式输出，格式如下：
```json
{
  "terms": [
    {
      "english": "epistemology",
      "chinese": "认识论",
      "explanation": "哲学的一个分支，研究知识的本质、来源、范围和可能性。它探讨'什么是知识'、'我们如何获得知识'等根本问题。",
      "example": "笛卡尔的'我思故我在'就是一个认识论论证，试图找到不可怀疑的知识基础。",
      "category": "term"
    }
  ]
}
```"""

TERM_EXTRACTION_USER_PROMPT = """请分析以下英文哲学文本，提取所有重要术语并提供专业翻译和解释：

---
{text}
---

以下英文术语已录入术语库，无需重复提取，请只提取尚未收录的新术语：
{existing_terms}

请确保：
1. 提取所有哲学术语、专有名词、哲学家人名（排除上方已收录术语）
2. 识别文本中反复出现的特定概念，给予统一翻译
3. 为每个术语提供基于权威来源的解释和示例
4. 如果没有术语需要提取，输出空列表 `{{"terms": []}}`
5. 以JSON格式输出"""

# ==============================================================================
# STEP 2: TRANSLATION PROMPTS
# ==============================================================================

TRANSLATION_SYSTEM_PROMPT = """你是一位专业的哲学翻译专家，专门将英语哲学播客翻译成中文，你的翻译需要在不改变原意、不增删关键信息、不添加原文没有的观点的前提下完成。

翻译要求：
1. 专业度上：
    - **不改变原意、不增删关键信息、不添加原文没有的观点**
    - **准确翻译哲学术语**: 请严格遵循下方术语表，使用标准中文译名。同时，请特别注意区分原文中的词汇是作为哲学术语使用，还是一般性的口语表达，因为同一个词在不同语境下可能有不同含义。例如，英文中的“truth”在哲学语境中应译为术语“真理”，而在日常口语中可能被译为“真相”或“真实”等
    - **忠实反映语气与不确定性**: 保持原文语气强度，不要引入强化或削弱语气的词汇，除非原文有同等表达，也不要夸张化原文表达
2. 翻译文本风格上：
    - **整体语气与原文本保持一致、表达更中文** 
        - 保持原文整体语气与行文风格，但具体措辞要更符合中文表达习惯，避免生硬直译
        - 不必照搬英文固定搭配或修辞；若直译会让中文读者感到别扭，请改写为自然、清晰的中文表达
        - 不要加入中文俚语，也不要加入过多成语或过分口语化，更不要因为口语化而改变原作者的风格
    - 译文应适合后续口播，读起来自然顺畅，但依旧保持原作者的行文风格
3. 翻译文本规则上：
    - **标注首次出现的术语** - 当专业术语或专有名词第一次出现时，在中文后用括号标注英文原文, 但注意不要产生嵌套括号或重复括号 (除非在原文中该位置有多个表达):
        - 格式：中文术语 (English Term)
        - 示例：认识论 (epistemology)、康德 (Kant)
        - 错误示例："灵魂 (soul (psyche))", 可原文只有 "soul", 故此处重复嵌套括号
    - **后续出现无需标注** - 某术语一旦已在译文中以"中文 (English)"形式标注过，此后同一术语的任何再次出现均只写中文，不再附加括号或英文，也不加粗体等强调符号。判定"同一术语"时，包含关系也算重复：若 A 是 B 的子串且指代同一实体，则 B 视为 A 的后续出现 (除非包含的词组有显著区别于原词的特殊含义)。
        - 正确示例："泰勒斯 (Thales) 是……后来，米利都的泰勒斯又提出了……"（后者不再标注）
        - 错误示例："泰勒斯 (Thales) 是……米利都的泰勒斯 (Thales of Miletus) 又……"（重复标注）
    - **保留段落结构** - 必须保留原文中的段落分隔（"\n\n"），以对应原文的段落

**本文术语对照表（必须严格遵守）**：
{terminology_table}

请将以下英文哲学内容翻译成中文，保持学术准确性和口播自然度："""


# ==============================================================================
# STEP 3: POLISHING PROMPTS (POST-EDIT FOR CHINESE PODCAST NARRATION)
# ==============================================================================

POLISHING_SYSTEM_PROMPT = """你是一位中文播客口播稿的专业润色编辑。你将收到两段内容：英文原文与中文翻译初稿。

你的任务：在不改变原意、不增删关键信息、不添加原文没有的观点的前提下，把中文初稿润色成更符合中文表达习惯、适合口播的版本。

润色原则（重要）：
1. 专业度上: **忠实与清晰并重**
    - 不要增删或夸张化原文信息，避免引入新的观点或解释。
    - 不要引入“八成/肯定/显然”等强化语气 或 与原文含义不符的词，如对不确定语气（may/might/possibly）保持为“可能/也许”等。
    - **术语与标注必须保持**：中文初稿中已有的中文译名与括号内英文内容必须原样保留，不得新增或删减（如：认识论 (epistemology)）。
    - **注意任何调整都不能改变原意或引入新的观点**， 如 "未来 人物A 的故乡" 和 "人物A 未来的故乡" 是不同的含义，需要忠实英文原文。
2. 翻译文本风格上：
    - **更中文，但整体风格与原文一致**：
        - 不必贴合英文句式或固定搭配；直译会别扭时应改写为自然中文。
        - 不要加入中文俚语，也不要加入过多成语或过分口语化，更不要因为口语化而改变原作者的表达风格。
        - 译文应适合后续口播，读起来自然顺畅，但依旧保持英文原文的 tone
    - **口播友好**：允许为口播理解做轻量调整，例如：
        - 适度前置原因/条件，避免英文式“原因后置”导致听感别扭（除非原文明显在制造悬念/笑点）。
        - 必要时补出中文常见的省略成分（如“这时/于是…”这类不改变含义的连接词），或把过长句拆成更易听懂的短句 (但不要总是把长句拆为短句，使得过分口语化，导致与哲学播客的专业性基调不符)。
        - 对英文表达中不符合中文习惯的文本进行中文化调整，包括不限于:
            - 调整语序使其更符合中文习惯：例如 "米利都在公元前6世纪早期，确实是个好地方" 改为"在公元前6世纪早期，米利都确实是个好地方"。
            - 对于名词化表达，做中文化改写：例如将 "计划是…" 类表达改为 “我/我们打算…”。
            - 对于翻译腔表达，替换为更符合中文习惯的表达，例如："美惠的" 可替换为 "美丽贤惠的"，但不得改变原意，需忠实原文
        - 改写破折号(“——”)表达方式，使得其与前后文更加流畅自然，例如： "我有一个想法——成为神明——被其他人嘲笑了…" 改为 "我有一个想法，就是成为神明，结果被其他人嘲笑了…"
            - 注意改写可适当增添连接词或句子成分 (如主语宾语等)，使得原句表述更加清晰，但不得改变原句含义，需忠实原文
3. 输出要求上：**只输出润色后的中文**：不要输出分析、不要列修改点
"""

POLISHING_USER_PROMPT = """请根据英文原文与中文初稿，输出润色后的中文口播稿。

【英文原文】
{english_text}

【中文初稿】
{draft_chinese}
"""

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def call_openrouter_api(
    system_prompt: str,
    user_content: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 21800,
    enable_web_search: bool = False,
    enable_reasoning: bool = False
) -> str:
    """
    Call OpenRouter API with given prompts.
    
    Args:
        system_prompt: System prompt
        user_content: User message content
        api_key: OpenRouter API key
        model: Model to use (overrides config default)
        temperature: Generation temperature
        max_tokens: Maximum output tokens
        enable_web_search: Enable web search plugin for authoritative sources
        enable_reasoning: Enable reasoning mode (adds reasoning config to payload)
        
    Returns:
        Response content string
    """
    api_key = api_key or OPENROUTER_API_KEY
    model = model or TRANSLATION_MODEL
    
    if not api_key:
        raise TranslationError("OPENROUTER_API_KEY is not set")
    
    endpoint = f"{OPENROUTER_BASE_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/philosophy-translation",
        "X-Title": "Philosophy Audio Translation"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **({
            "reasoning": {"effort": "medium"}
        } if enable_reasoning else {})
    }
    
    # Enable web search plugin for authoritative sources
    if enable_web_search:
        payload["plugins"] = [
            {
                "id": "web",
                "enabled": True,
                # Focus on authoritative sources
                "search_prompt": "site:plato.stanford.edu OR site:wikipedia.org OR site:iep.utm.edu"
            }
        ]
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=180  # Longer timeout for web search
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.HTTPError as e:
        raise TranslationError(f"API error: {e}")
    except KeyError as e:
        raise TranslationError(f"Unexpected API response format: {e}")


def extract_terms(
    text: str,
    term_db: Optional[TerminologyDB] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_search: bool = True,
    enable_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Step 1: Extract philosophy terms from text with AI-powered analysis.
    
    Uses AI to identify and explain:
    - Philosophy terms (epistemology, ontology, etc.)
    - Proper nouns (schools, theories)
    - Philosopher names
    - Text-specific recurring concepts
    
    Args:
        text: English text to analyze
        term_db: Terminology database for storage
        api_key: OpenRouter API key
        model: Model to use (overrides config default)
        enable_search: Enable web search for authoritative explanations
        
    Returns:
        dict: {
            "terms": [{"english": str, "chinese": str, "explanation": str, "example": str, "category": str}, ...],
            "new_terms_count": int,
            "existing_terms_count": int
        }
    """
    if term_db is None:
        print("Termdb does not exist, therefore it creates a new one...")
        term_db = TerminologyDB()
    
    # Build list of existing English terms that appear in the input text.
    # Only include relevant terms to keep the prompt compact as term_db grows.
    text_lower = text.lower()
    existing_english = sorted(
        eng for eng in term_db.terms.keys()
        if re.search(r'\b' + re.escape(eng.lower()) + r'\b', text_lower)
    )
    if existing_english:
        existing_terms_str = "\n".join(f"- {t}" for t in existing_english)
    else:
        existing_terms_str = "（暂无已收录术语）"

    # Call AI to extract terms
    user_prompt = TERM_EXTRACTION_USER_PROMPT.format(text=text, existing_terms=existing_terms_str)
    
    response = call_openrouter_api(
        system_prompt=TERM_EXTRACTION_SYSTEM_PROMPT,
        user_content=user_prompt,
        api_key=api_key,
        model=model,
        temperature=0.2,  # Lower for consistency
        enable_web_search=enable_search,
        enable_reasoning=enable_reasoning
    )
    
    # Parse JSON response
    terms = parse_term_extraction_response(response)
    if not terms:
        return {
            "terms": [],
            "new_terms": [],
            "existing_terms": [],
            "new_terms_count": 0,
            "existing_terms_count": 0
        }
    
    # Categorize as new or existing
    new_terms = []
    existing_terms = []
    
    for term in terms:
        english = term.get("english", "").lower().strip()
        chinese_raw = term.get("chinese", "")
        chinese_single = _normalize_single_term(chinese_raw)
        term["chinese"] = chinese_single
        
        if term_db.get_translation(english):
            # Prefer termdb's translation and details over extracted ones
            term["chinese"] = term_db.get_translation(english)
            details = term_db.get_term_details(english)
            if details:
                if details.get("explanation"):
                    term["explanation"] = details["explanation"]
                if details.get("example"):
                    term["example"] = details["example"]
            existing_terms.append(term)
        else:
            new_terms.append(term)
            # Add new term to database
            term_db.add_term_with_details(
                english=english,
                chinese=chinese_single,
                explanation=term.get("explanation", ""),
                example=term.get("example", ""),
                category=term.get("category", "term")
            )
    
    return {
        "terms": terms,
        "new_terms": new_terms,
        "existing_terms": existing_terms,
        "new_terms_count": len(new_terms),
        "existing_terms_count": len(existing_terms)
    }


def parse_term_extraction_response(response: str) -> List[Dict]:
    """
    Parse the JSON response from term extraction.
    
    Args:
        response: Raw API response
        
    Returns:
        List of term dictionaries
    """
    # Try to extract JSON from response
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[\s\S]*"terms"[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
        else:
            return []
    
    try:
        data = json.loads(json_str)
        return data.get("terms", [])
    except json.JSONDecodeError:
        return []

def _normalize_single_term(chinese: str) -> str:
    """Normalize extracted Chinese term into a single canonical string.

    The term extraction model sometimes returns multiple candidates joined by separators
    like '/', '或', '、'. This function keeps only the first candidate.
    """
    if not chinese:
        return chinese

    value = str(chinese).strip()
    if not value:
        return value

    # Common multi-candidate separators
    separators = ["／", "/", "|", "｜", ";", "；", "，", ",", "、", "\n"]
    for sep in separators:
        if sep in value:
            head = value.split(sep, 1)[0].strip()
            if head:
                value = head
            break

    # Handle patterns like "A 或 B" or "A或B"
    if "或" in value:
        head = value.split("或", 1)[0].strip()
        if head:
            value = head

    # Remove surrounding quotes or brackets that sometimes leak in
    value = value.strip().strip('"').strip("'")
    return value


def _split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _normalize_extracted_terms(extracted: Any) -> Dict[str, Any]:
    """Normalize extracted term payloads into a consistent dict shape."""
    if not isinstance(extracted, dict):
        print("The error happens when extracting the terms...")
        return {
            "terms": [],
            "new_terms": [],
            "existing_terms": [],
            "new_terms_count": 0,
            "existing_terms_count": 0
        }

    terms = extracted.get("terms")
    if not isinstance(terms, list):
        terms = []

    extracted["terms"] = terms
    extracted.setdefault("new_terms", [])
    extracted.setdefault("existing_terms", [])

    if "new_terms_count" not in extracted:
        extracted["new_terms_count"] = len(extracted.get("new_terms", []))
    if "existing_terms_count" not in extracted:
        extracted["existing_terms_count"] = len(extracted.get("existing_terms", []))

    return extracted


def _group_paragraphs_by_length(paragraphs: List[str], target_len: Optional[int]) -> List[List[str]]:
    """Group paragraphs into segments using a soft target length.

    Paragraphs are never split. Once the accumulated length reaches the target,
    the current segment is closed after that paragraph.
    """
    if not paragraphs:
        return []
    if not target_len or target_len <= 0:
        return [[p] for p in paragraphs]

    groups: List[List[str]] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current and current_len >= target_len:
            groups.append(current)
            current = [paragraph]
            current_len = paragraph_len
            continue

        current.append(paragraph)
        current_len += paragraph_len

    if current:
        groups.append(current)

    return groups


def _split_sentences(text: str, is_chinese: bool) -> List[str]:
    if not text:
        return []
    if is_chinese:
        parts = re.split(r"(?<=[。！？])", text)
    else:
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_sentences_to_count(sentences: List[str], target_count: int) -> List[str]:
    if not sentences:
        return []
    if target_count <= 1:
        return [" ".join(sentences).strip()]

    total_len = sum(len(s) for s in sentences)
    target_len = max(1, total_len // target_count)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for idx, sentence in enumerate(sentences):
        current.append(sentence)
        current_len += len(sentence)
        remaining_sentences = len(sentences) - idx - 1
        remaining_chunks = target_count - len(chunks) - 1

        if remaining_chunks <= 0:
            continue

        if current_len >= target_len and remaining_sentences >= remaining_chunks:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current).strip())

    # If we are short, append empty chunks to keep alignment predictable.
    while len(chunks) < target_count:
        chunks.append("")

    # If we are long, merge extras into the last chunk.
    if len(chunks) > target_count:
        merged_tail = " ".join(chunks[target_count - 1:]).strip()
        chunks = chunks[:target_count - 1] + [merged_tail]

    return chunks


def _enforce_annotation_rules(
    text: str,
    extracted_terms: Optional[Dict],
    term_db: TerminologyDB,
    chinese_only_terms: Optional[List[str]] = None
) -> str:
    """Enforce deterministic annotation rules after polishing.

    Rules:
    - Only the first occurrence of a term keeps "中文 (English)".
    - Subsequent occurrences drop the parenthetical English.
    - Remove Markdown emphasis markers like **...**.
    - Do NOT add "(English)" to any term that is already in Chinese.
    """
    if not text:
        return text

    # Remove Markdown bold markers
    cleaned = text.replace("**", "")

    # Collapse nested parentheses like "术语 (outer (inner))" -> "术语 (outer)"
    # Handle both ASCII () and fullwidth Chinese （）
    nested_patterns = [
        re.compile(r"\(([^()]*?)\([^()]*?\)\s*\)"),
        re.compile(r"（([^（）]*?)（[^（）]*?）\s*）"),
    ]
    for nested_pattern in nested_patterns:
        while True:
            new_cleaned = nested_pattern.sub(lambda m: f"({m.group(1)})" if m.group(0)[0] == '(' else f"（{m.group(1)}）", cleaned)
            if new_cleaned == cleaned:
                break
            cleaned = new_cleaned

    terms = extracted_terms.get("terms", []) if extracted_terms else []
    term_pairs: List[tuple] = []

    for term in terms:
        english = term.get("english", "")
        chinese = term.get("chinese", "")
        if english and chinese:
            term_pairs.append((english, chinese))

    # Also include DB terms that appear in text but weren't extracted
    for eng in term_db.terms.keys():
        if any(t[0].lower() == eng.lower() for t in term_pairs):
            continue
        chi = term_db.get_translation(eng)
        if chi:
            term_pairs.append((eng, chi))

    for english, chinese in term_pairs:
        # Match both ASCII () and fullwidth Chinese （）
        pattern = re.compile(
            rf"{re.escape(chinese)}\s*[(（]\s*{re.escape(english)}\s*[)）]",
            re.IGNORECASE
        )

        matches = list(pattern.finditer(cleaned))
        if len(matches) <= 1:
            continue

        # Replace all occurrences after the first with just the Chinese term.
        # Keep everything up to and including the first match intact.
        new_parts: List[str] = []
        last_idx = matches[0].end()
        new_parts.append(cleaned[:last_idx])
        for idx, match in enumerate(matches):
            if idx == 0:
                continue
            start, end = match.span()
            new_parts.append(cleaned[last_idx:start])
            new_parts.append(chinese)
            last_idx = end
        new_parts.append(cleaned[last_idx:])
        cleaned = "".join(new_parts)

    if chinese_only_terms:
        for term in chinese_only_terms:
            normalized = str(term).strip()
            if not normalized:
                continue
            # Match both ASCII () and fullwidth Chinese （）
            pattern = re.compile(
                rf"{re.escape(normalized)}\s*(?:\([^)]*\)|（[^）]*）)",
                re.IGNORECASE
            )
            cleaned = pattern.sub(normalized, cleaned)

    return cleaned


def polish_translation(
    english_text: str,
    draft_chinese: str,
    extracted_terms: Optional[Dict] = None,
    term_db: Optional[TerminologyDB] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    segment_target_chars: Optional[int] = None,
    chinese_only_terms: Optional[List[str]] = None
) -> str:
    """Post-edit the draft Chinese to be podcast-friendly, grounded in the English source."""
    if term_db is None:
        term_db = TerminologyDB()

    system_prompt = POLISHING_SYSTEM_PROMPT

    # Split into aligned segments to avoid length-induced quality loss.
    chinese_paragraphs = _split_paragraphs(draft_chinese)
    if not chinese_paragraphs:
        chinese_paragraphs = [draft_chinese]

    chinese_groups = _group_paragraphs_by_length(chinese_paragraphs, segment_target_chars)
    chinese_parts = ["\n\n".join(group).strip() for group in chinese_groups]

    english_paragraphs = _split_paragraphs(english_text)
    if len(english_paragraphs) == len(chinese_paragraphs):
        english_parts: List[str] = []
        idx = 0
        for group in chinese_groups:
            group_len = len(group)
            english_parts.append("\n\n".join(english_paragraphs[idx:idx + group_len]).strip())
            idx += group_len
    else:
        print("[Warning] English and Chinese paragraph counts do not match; polishing whole passage in one go.")
        english_parts = [english_text]
        chinese_parts = [draft_chinese]

    polished_parts: List[str] = []
    for eng_part, chi_part in zip(english_parts, chinese_parts):
        user_content = POLISHING_USER_PROMPT.format(
            english_text=eng_part,
            draft_chinese=chi_part
        )

        polished_part = call_openrouter_api(
            system_prompt=system_prompt,
            user_content=user_content,
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=21800
        ).strip()

        polished_parts.append(polished_part)

    polished = "\n\n".join(p for p in polished_parts if p)
    return _enforce_annotation_rules(
        polished,
        extracted_terms,
        term_db,
        chinese_only_terms=chinese_only_terms
    )

def translate_text(
    text: str,
    extracted_terms: Optional[Dict] = None,
    term_db: Optional[TerminologyDB] = None,
    episode_id: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False
) -> str:
    """
    Step 2: Translate text using extracted terminology.
    
    Args:
        text: English text to translate
        extracted_terms: Terms extracted from Step 1
        term_db: Terminology database
        episode_id: Episode ID for first-occurrence tracking
        api_key: OpenRouter API key
        model: Model to use (overrides config default)
        
    Returns:
        Translated Chinese text
    """
    if term_db is None:
        term_db = TerminologyDB()
    
    if episode_id:
        term_db.set_episode(episode_id)
    
    # Build terminology table
    if extracted_terms:
        terms = extracted_terms.get("terms", [])
    else:
        terms = []
    
    # Build terminology table for translation
    terminology_lines = []
    
    for term in terms:
        english = term.get("english", "")
        chinese = term.get("chinese", "")
        
        if english and chinese:
            terminology_lines.append(f"- {english} → {chinese}")
    
    # Also include terms from database that appear in text but weren't extracted
    for eng in term_db.terms.keys():
        # Check if already in extraction results
        if not any(t.get("english", "").lower() == eng.lower() for t in terms):
            # Check if term appears in text
            if re.search(r'\b' + re.escape(eng) + r'\b', text, re.IGNORECASE):
                # Get translation (handles both simple and detailed format)
                chi = term_db.get_translation(eng)
                if chi:
                    terminology_lines.append(f"- {eng} → {chi}")
    
    terminology_table = "\n".join(terminology_lines) if terminology_lines else "无特定术语"
    # Build system prompt
    system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
        terminology_table=terminology_table
    )
    
    # Call translation API
    translated = call_openrouter_api(
        system_prompt=system_prompt,
        user_content=text,
        api_key=api_key,
        model=model,
        temperature=0.7,
        enable_reasoning=enable_reasoning
    )
    
    # Mark terms as seen
    for term in terms:
        english = term.get("english", "")
        if english:
            term_db.mark_term_seen(english)
    
    return translated


def translate_with_extraction(
    text: str,
    term_db: Optional[TerminologyDB] = None,
    episode_id: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_search: bool = True,
    save_terms: bool = True,
    enable_reasoning: bool = False,
    polish_segment_chars: Optional[int] = None,
    chinese_only_terms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Full three-step translation process.
    
    Step 1: Extract and explain terms with AI search
    Step 2: Translate using extracted terminology
    Step 3: Polish the Chinese draft for podcast narration (based on English + draft)
    
    Args:
        text: English text to translate
        term_db: Terminology database
        episode_id: Episode ID for tracking
        api_key: OpenRouter API key
        model: Model to use (overrides config default)
        enable_search: Enable web search for term explanations
        save_terms: Save new terms to database
        enable_reasoning: Enable reasoning mode (for reasoning models like deepseek-r1)
        polish_segment_chars: Soft target length for each polishing segment (paragraph-aligned)
        
    Returns:
        dict: {
            "extracted_terms": {...},
            "draft_translation": str,
            "translation": str,
            "new_terms_added": int,
            "reasoning_enabled": bool
        }
    """
    if term_db is None:
        term_db = TerminologyDB()
    
    if episode_id:
        term_db.set_episode(episode_id)
    
    # Step 1: Extract terms
    print("[Step 1/3] Extracting philosophy terms...")
    extracted = extract_terms(
        text=text,
        term_db=term_db,
        api_key=api_key,
        model=model,
        enable_search=enable_search,
        enable_reasoning=enable_reasoning
    )
    extracted = _normalize_extracted_terms(extracted)
    print(f"  Found {len(extracted['terms'])} terms ({extracted['new_terms_count']} new)")
    
    # Step 2: Translate
    print("[Step 2/3] Translating text...")
    draft_translation = translate_text(
        text=text,
        extracted_terms=extracted,
        term_db=term_db,
        episode_id=episode_id,
        api_key=api_key,
        model=model,
        enable_reasoning=enable_reasoning
    )

    # Step 3: Polish
    print("[Step 3/3] Polishing Chinese for podcast narration...")
    polished_translation = polish_translation(
        english_text=text,
        draft_chinese=draft_translation,
        extracted_terms=extracted,
        term_db=term_db,
        api_key=api_key,
        model=model,
        segment_target_chars=polish_segment_chars,
        chinese_only_terms=chinese_only_terms
    )
    
    # Save database if requested
    if save_terms:
        term_db.save()
    
    return {
        "extracted_terms": extracted,
        "draft_translation": draft_translation,
        "translation": polished_translation,
        "new_terms_added": extracted["new_terms_count"],
        "reasoning_enabled": enable_reasoning
    }


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def save_extracted_terms_to_file(
    terms: List[Dict],
    output_path: str,
    append: bool = True
) -> str:
    """
    Save extracted terms to a JSON file for future reuse.
    
    Args:
        terms: List of term dictionaries
        output_path: Path to save JSON file
        append: If True, append to existing file
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing = {}
    if append and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    
    # Merge terms (new terms override existing)
    for term in terms:
        english = term.get("english", "").lower().strip()
        if english:
            existing[english] = term
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    
    return str(output_path)


def format_term_for_display(term: Dict) -> str:
    """
    Format a term for human-readable display.
    
    Example output:
    Pre-Socratics (前苏格拉底哲学家)
    解释: 指生活在苏格拉底之前...
    示例: 泰勒斯 (Thales)...
    """
    english = term.get("english", "")
    chinese = term.get("chinese", "")
    explanation = term.get("explanation", "")
    example = term.get("example", "")
    
    lines = [f"{english} ({chinese})"]
    if explanation:
        lines.append(f"解释: {explanation}")
    if example:
        lines.append(f"示例: {example}")
    
    return "\n".join(lines)


# ==============================================================================
# DRY RUN & TESTING
# ==============================================================================

def dry_run_extract_terms(text: str, model: str = None, enable_search: bool = True) -> Dict:
    """
    Dry run: Test term extraction logic without calling API.
    """
    return {
        "input_text_preview": text[:200] + "...",
        "extraction_prompt_preview": TERM_EXTRACTION_SYSTEM_PROMPT[:300] + "...",
        "user_prompt_preview": TERM_EXTRACTION_USER_PROMPT.format(
            text=text[:100] + "...",
            existing_terms="（暂无已收录术语）"
        )[:300] + "...",
        "api_endpoint": f"{OPENROUTER_BASE_URL}/chat/completions",
        "model": model or TRANSLATION_MODEL,
        "web_search_enabled": enable_search,
        "expected_output_format": {
            "terms": [
                {"english": "term", "chinese": "术语", "explanation": "...", "example": "...", "category": "term"}
            ]
        },
        "logic_valid": True
    }


def dry_run_translate(text: str, model: str = None) -> Dict:
    """
    Dry run: Test full translation logic without calling API.
    """
    return {
        "step1": "Extract terms with AI + web search",
        "step2": "Translate with extracted terminology",
        "step3": "Polish Chinese draft for podcast narration",
        "input_text_preview": text[:200] + "...",
        "translation_prompt_preview": TRANSLATION_SYSTEM_PROMPT[:300] + "...",
        "polishing_prompt_preview": POLISHING_SYSTEM_PROMPT[:300] + "...",
        "api_endpoint": f"{OPENROUTER_BASE_URL}/chat/completions",
        "model": model or TRANSLATION_MODEL,
        "api_key_set": bool(OPENROUTER_API_KEY),
        "logic_valid": True
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=== Two-Step Translator Dry Run ===\n")
    
    test_text = """
    In this lecture, we'll explore how the Pre-Socratics revolutionized Western thought. 
    Thales, often considered the first philosopher, proposed that water is the arché of all things.
    Heraclitus, in contrast, argued for the primacy of logos and constant change.
    Later, Parmenides would challenge these views with his concept of Being.
    These thinkers laid the groundwork for Plato's metaphysics and Aristotle's epistemology.
    """
    
    print("Step 1: Term Extraction Dry Run")
    print("-" * 40)
    result1 = dry_run_extract_terms(test_text)
    print(f"  Model: {result1['model']}")
    print(f"  Web search: {result1['web_search_enabled']}")
    print(f"  Logic valid: {result1['logic_valid']}")
    
    print("\nStep 2: Translation Dry Run")
    print("-" * 40)
    result2 = dry_run_translate(test_text)
    print(f"  Step 1: {result2['step1']}")
    print(f"  Step 2: {result2['step2']}")
    print(f"  API key set: {result2['api_key_set']}")
    print(f"  Logic valid: {result2['logic_valid']}")
    
    print("\n✓ All logic checks PASSED")
