"""
Philosophy terminology database for translation consistency.
Tracks terms across episodes and handles bracket notation for first occurrences.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Set
from datetime import datetime

from config import PHILOSOPHY_TERMS_PATH, EPISODE_TERMS_PATH, DATA_DIR


class TerminologyDB:
    """
    Manages philosophy terminology with episode-aware first-occurrence tracking.
    
    Features:
    - Pre-loaded philosophy glossary (English -> Chinese)
    - Track which terms have been seen in which episodes
    - First occurrence: add bracket notation
    - Subsequent occurrences: no brackets
    """
    
    def __init__(
        self,
        terms_path: Optional[Path] = None,
        episode_history_path: Optional[Path] = None
    ):
        """
        Initialize terminology database.
        
        Args:
            terms_path: Path to philosophy terms JSON
            episode_history_path: Path to episode term history JSON
        """
        self.terms_path = terms_path or PHILOSOPHY_TERMS_PATH
        self.episode_history_path = episode_history_path or EPISODE_TERMS_PATH
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.terms: Dict[str, str] = self._load_terms()
        self.episode_history: Dict[str, Dict] = self._load_episode_history()
        
        # Current episode tracking
        self._current_episode: Optional[str] = None
        self._current_episode_seen: Set[str] = set()
    
    def _load_terms(self) -> Dict[str, str]:
        """Load philosophy terms from JSON file."""
        if self.terms_path.exists():
            with open(self.terms_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print("The collected term path does not exist!!!")
        return {}
    
    def _load_episode_history(self) -> Dict[str, Dict]:
        """Load episode term history from JSON file."""
        if self.episode_history_path.exists():
            with open(self.episode_history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print("The episode term path does not exist!!!")
        return {}
    
    def save(self):
        """Save both terms and episode history to disk."""
        with open(self.terms_path, "w", encoding="utf-8") as f:
            json.dump(self.terms, f, ensure_ascii=False, indent=2)
        
        with open(self.episode_history_path, "w", encoding="utf-8") as f:
            json.dump(self.episode_history, f, ensure_ascii=False, indent=2)
    
    def set_episode(self, episode_id: str):
        """
        Set current episode for term tracking.
        
        Args:
            episode_id: Unique identifier for the episode
        """
        self._current_episode = episode_id
        
        # Load previously seen terms for this episode
        if episode_id in self.episode_history:
            self._current_episode_seen = set(
                self.episode_history[episode_id].get("terms_seen", [])
            )
        else:
            self._current_episode_seen = set()
            self.episode_history[episode_id] = {
                "terms_seen": [],
                "created_at": datetime.now().isoformat()
            }
    
    def is_first_occurrence(self, english_term: str) -> bool:
        """
        Check if this is the first occurrence of a term in the current episode.
        
        Args:
            english_term: English philosophy term
            
        Returns:
            True if first occurrence, False otherwise
        """
        if not self._current_episode:
            # No episode set, assume first occurrence
            return True
        
        normalized = english_term.lower().strip()
        
        # Check if seen in any previous episode
        for ep_id, ep_data in self.episode_history.items():
            if ep_id != self._current_episode:
                if normalized in ep_data.get("terms_seen", []):
                    return False
        
        # Check if seen in current episode
        return normalized not in self._current_episode_seen
    
    def mark_term_seen(self, english_term: str):
        """
        Mark a term as seen in the current episode.
        
        Args:
            english_term: English philosophy term
        """
        if not self._current_episode:
            return
        
        normalized = english_term.lower().strip()
        self._current_episode_seen.add(normalized)
        
        # Update episode history
        if "terms_seen" not in self.episode_history[self._current_episode]:
            self.episode_history[self._current_episode]["terms_seen"] = []
        
        if normalized not in self.episode_history[self._current_episode]["terms_seen"]:
            self.episode_history[self._current_episode]["terms_seen"].append(normalized)
    
    def add_term(self, english: str, chinese: str):
        """
        Add a new term to the dictionary (simple format).
        
        Args:
            english: English term
            chinese: Chinese translation
        """
        normalized = english.lower().strip()
        self.terms[normalized] = chinese
    
    def add_term_with_details(
        self,
        english: str,
        chinese: str,
        explanation: str = "",
        example: str = "",
        category: str = "term"
    ):
        """
        Add a term with detailed information (explanation, example).
        
        Args:
            english: English term
            chinese: Chinese translation
            explanation: Brief explanation of the term
            example: Usage example
            category: Category (philosopher/term/concept/school/text-specific)
        """
        normalized = english.lower().strip()
        
        # Store as detailed format
        self.terms[normalized] = {
            "chinese": chinese,
            "explanation": explanation,
            "example": example,
            "category": category
        }
    
    def get_translation(self, english_term: str) -> Optional[str]:
        """
        Get Chinese translation for an English term.
        
        Handles both simple format (str) and detailed format (dict).
        
        Args:
            english_term: English philosophy term
            
        Returns:
            Chinese translation or None if not found
        """
        normalized = english_term.lower().strip()
        value = self.terms.get(normalized)
        
        if value is None:
            return None
        
        # Handle both formats
        if isinstance(value, dict):
            return value.get("chinese", "")
        else:
            return value
    
    def get_term_details(self, english_term: str) -> Optional[Dict]:
        """
        Get full details for a term.
        
        Args:
            english_term: English philosophy term
            
        Returns:
            Dict with chinese, explanation, example, category or None
        """
        normalized = english_term.lower().strip()
        value = self.terms.get(normalized)
        
        if value is None:
            return None
        
        # Convert simple format to detailed
        if isinstance(value, str):
            return {
                "english": normalized,
                "chinese": value,
                "explanation": "",
                "example": "",
                "category": "term"
            }
        else:
            return {
                "english": normalized,
                **value
            }
    
    def format_term(self, english_term: str, chinese_term: str) -> str:
        """
        Format term with bracket notation if first occurrence.
        
        Args:
            english_term: Original English term
            chinese_term: Chinese translation
            
        Returns:
            Formatted string: "中文 (English)" for first, "中文" for subsequent
        """
        if self.is_first_occurrence(english_term):
            self.mark_term_seen(english_term)
            return f"{chinese_term} ({english_term})"
        else:
            return chinese_term
    
    def format_term_with_explanation(self, english_term: str) -> str:
        """
        Format term with explanation for display.
        
        Example output:
        Pre-Socratics (前苏格拉底哲学家)
        解释: 指生活在苏格拉底之前...
        示例: 泰勒斯 (Thales)...
        
        Args:
            english_term: English philosophy term
            
        Returns:
            Formatted string with explanation and example
        """
        details = self.get_term_details(english_term)
        if not details:
            return f"{english_term} (未找到)"
        
        lines = [f"{details['english']} ({details['chinese']})"]
        
        if details.get("explanation"):
            lines.append(f"解释: {details['explanation']}")
        
        if details.get("example"):
            lines.append(f"示例: {details['example']}")
        
        return "\n".join(lines)
    
    def get_all_seen_terms(self) -> Dict[str, Set[str]]:
        """Get all terms seen across all episodes."""
        result = {}
        for ep_id, ep_data in self.episode_history.items():
            result[ep_id] = set(ep_data.get("terms_seen", []))
        return result
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        # Count detailed vs simple terms
        detailed_count = sum(1 for v in self.terms.values() if isinstance(v, dict))
        simple_count = len(self.terms) - detailed_count
        
        return {
            "total_terms": len(self.terms),
            "detailed_terms": detailed_count,
            "simple_terms": simple_count,
            "total_episodes": len(self.episode_history),
            "current_episode": self._current_episode,
            "current_episode_terms_seen": len(self._current_episode_seen)
        }


def create_default_philosophy_terms() -> Dict[str, str]:
    """
    Create default philosophy terminology dictionary.
    
    Returns:
        Dict mapping English terms to Chinese translations
    """
    return {
        # Epistemology 认识论
        "epistemology": "认识论",
        "empiricism": "经验主义",
        "rationalism": "理性主义",
        "skepticism": "怀疑主义",
        "a priori": "先验的",
        "a posteriori": "后验的",
        "knowledge": "知识",
        "belief": "信念",
        "justification": "辩护",
        "truth": "真理",
        
        # Metaphysics 形而上学
        "metaphysics": "形而上学",
        "ontology": "本体论",
        "substance": "实体",
        "essence": "本质",
        "existence": "存在",
        "being": "存在",
        "reality": "实在",
        "causality": "因果性",
        "determinism": "决定论",
        "free will": "自由意志",
        
        # Ethics 伦理学
        "ethics": "伦理学",
        "morality": "道德",
        "virtue": "美德",
        "deontology": "义务论",
        "consequentialism": "后果主义",
        "utilitarianism": "功利主义",
        "categorical imperative": "绝对命令",
        "moral realism": "道德实在论",
        
        # Logic 逻辑学
        "logic": "逻辑学",
        "syllogism": "三段论",
        "fallacy": "谬误",
        "deduction": "演绎",
        "induction": "归纳",
        "validity": "有效性",
        "soundness": "可靠性",
        
        # Phenomenology 现象学
        "phenomenology": "现象学",
        "consciousness": "意识",
        "intentionality": "意向性",
        "qualia": "感质",
        "subjective experience": "主观经验",
        
        # Existentialism 存在主义
        "existentialism": "存在主义",
        "absurdity": "荒谬",
        "authenticity": "本真性",
        "bad faith": "自欺",
        "angst": "焦虑",
        "dasein": "此在",
        
        # Philosophy of Mind 心灵哲学
        "philosophy of mind": "心灵哲学",
        "dualism": "二元论",
        "materialism": "唯物主义",
        "physicalism": "物理主义",
        "functionalism": "功能主义",
        "mind-body problem": "身心问题",
        
        # Ancient Philosophy 古代哲学
        "eudaimonia": "幸福",
        "logos": "逻各斯",
        "telos": "目的",
        "arête": "卓越",
        "phronesis": "实践智慧",
        "nous": "理智",
        
        # Philosophers 哲学家
        "plato": "柏拉图",
        "aristotle": "亚里士多德",
        "kant": "康德",
        "hegel": "黑格尔",
        "nietzsche": "尼采",
        "heidegger": "海德格尔",
        "wittgenstein": "维特根斯坦",
        "husserl": "胡塞尔",
        "sartre": "萨特",
        "descartes": "笛卡尔",
        "hume": "休谟",
        "locke": "洛克"
    }


def dry_run_terminology() -> dict:
    """
    Dry run: Test terminology database logic without persistence.
    
    Returns:
        dict: Test results
    """
    # Create temporary in-memory database
    db = TerminologyDB()
    
    # Test 1: Add default terms
    default_terms = create_default_philosophy_terms()
    for eng, chi in default_terms.items():
        db.add_term(eng, chi)
    
    # Test 2: Set episode
    db.set_episode("test_ep001")
    
    # Test 3: First occurrence check
    test_term = "epistemology"
    is_first = db.is_first_occurrence(test_term)
    translation = db.get_translation(test_term)
    formatted = db.format_term(test_term, translation)
    
    # Test 4: Second occurrence (should not have brackets)
    is_second_first = db.is_first_occurrence(test_term)
    formatted_second = db.format_term(test_term, translation)
    
    # Test 5: Stats
    stats = db.get_stats()
    
    return {
        "terms_loaded": len(db.terms),
        "test_term": test_term,
        "translation": translation,
        "first_occurrence_check": is_first,
        "formatted_first": formatted,
        "second_occurrence_check": is_second_first,
        "formatted_second": formatted_second,
        "expected_first": f"{translation} ({test_term})",
        "expected_second": translation,
        "stats": stats,
        "logic_valid": (
            is_first == True and 
            is_second_first == False and
            formatted == f"{translation} ({test_term})" and
            formatted_second == translation
        )
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=== Terminology Database Dry Run ===\n")
    
    result = dry_run_terminology()
    
    print(f"Terms loaded: {result['terms_loaded']}")
    print(f"\nTest term: '{result['test_term']}'")
    print(f"Translation: '{result['translation']}'")
    print(f"\nFirst occurrence:")
    print(f"  is_first: {result['first_occurrence_check']}")
    print(f"  formatted: '{result['formatted_first']}'")
    print(f"  expected:  '{result['expected_first']}'")
    print(f"\nSecond occurrence:")
    print(f"  is_first: {result['second_occurrence_check']}")
    print(f"  formatted: '{result['formatted_second']}'")
    print(f"  expected:  '{result['expected_second']}'")
    print(f"\nStats: {result['stats']}")
    print(f"\n✓ Logic check: {'PASSED' if result['logic_valid'] else 'FAILED'}")
