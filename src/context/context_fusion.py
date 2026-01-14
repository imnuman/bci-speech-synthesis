"""
Context Fusion for BCI Decoding
Uses ambient speech to improve neural decoder accuracy
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ContextWindow:
    """Container for recent context."""
    text: str
    timestamp: float
    confidence: float


class ContextFusion:
    """
    Fuses ambient speech context with BCI decoder output.

    The idea: If someone asks "Are you thirsty?", the BCI decoder
    should be more likely to output "yes" or "no" than "bathroom".

    Methods:
    1. Keyword matching - boost probabilities for relevant words
    2. LLM refinement - use small LLM to reason about context
    """

    # Question type to response mapping
    QUESTION_PATTERNS = {
        'binary': {
            'keywords': ['are you', 'do you', 'is it', 'can you', 'did you'],
            'boost': ['yes', 'no']
        },
        'need': {
            'keywords': ['need', 'want', 'like'],
            'boost': ['yes', 'no', 'help']
        },
        'comfort': {
            'keywords': ['thirsty', 'water', 'drink'],
            'boost': ['yes', 'no', 'water']
        },
        'pain': {
            'keywords': ['hurt', 'pain', 'uncomfortable'],
            'boost': ['yes', 'no', 'pain']
        },
        'temperature': {
            'keywords': ['cold', 'hot', 'warm'],
            'boost': ['yes', 'no', 'cold', 'hot']
        },
        'fatigue': {
            'keywords': ['tired', 'sleep', 'rest'],
            'boost': ['yes', 'no', 'tired']
        },
        'bathroom': {
            'keywords': ['bathroom', 'toilet', 'restroom'],
            'boost': ['yes', 'no', 'bathroom']
        },
        'food': {
            'keywords': ['hungry', 'food', 'eat'],
            'boost': ['yes', 'no', 'hungry']
        }
    }

    def __init__(
        self,
        context_window_sec: float = 10.0,
        boost_factor: float = 2.0
    ):
        """
        Initialize context fusion.

        Args:
            context_window_sec: Time window for context relevance
            boost_factor: Probability boost multiplier for relevant words
        """
        self.context_window_sec = context_window_sec
        self.boost_factor = boost_factor

        self._context_history: List[ContextWindow] = []

    def add_context(self, text: str, timestamp: float, confidence: float = 1.0):
        """Add new context to history."""
        self._context_history.append(ContextWindow(
            text=text.lower(),
            timestamp=timestamp,
            confidence=confidence
        ))

        # Trim old context
        cutoff = timestamp - self.context_window_sec
        self._context_history = [
            c for c in self._context_history
            if c.timestamp > cutoff
        ]

    def get_recent_context(self) -> str:
        """Get concatenated recent context."""
        return ' '.join(c.text for c in self._context_history)

    def detect_question_type(self, context: str) -> Optional[str]:
        """Detect the type of question being asked."""
        context_lower = context.lower()

        for qtype, pattern in self.QUESTION_PATTERNS.items():
            for keyword in pattern['keywords']:
                if keyword in context_lower:
                    return qtype

        return None

    def refine(
        self,
        decoder_output: Dict[str, float],
        context: Optional[str] = None
    ) -> str:
        """
        Refine decoder output using context.

        Args:
            decoder_output: Dict mapping words to probabilities
            context: Optional context string (uses history if None)

        Returns:
            Most likely intent word
        """
        if context is None:
            context = self.get_recent_context()

        if not context:
            # No context, return argmax
            return max(decoder_output, key=decoder_output.get)

        # Detect question type
        qtype = self.detect_question_type(context)

        # Apply boosts
        refined = decoder_output.copy()

        if qtype and qtype in self.QUESTION_PATTERNS:
            boost_words = self.QUESTION_PATTERNS[qtype]['boost']
            for word in boost_words:
                if word in refined:
                    refined[word] *= self.boost_factor

        # Normalize
        total = sum(refined.values())
        refined = {k: v / total for k, v in refined.items()}

        return max(refined, key=refined.get)

    def refine_with_probs(
        self,
        decoder_output: Dict[str, float],
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """Refine and return full probability distribution."""
        if context is None:
            context = self.get_recent_context()

        if not context:
            return decoder_output

        qtype = self.detect_question_type(context)
        refined = decoder_output.copy()

        if qtype and qtype in self.QUESTION_PATTERNS:
            boost_words = self.QUESTION_PATTERNS[qtype]['boost']
            for word in boost_words:
                if word in refined:
                    refined[word] *= self.boost_factor

        # Normalize
        total = sum(refined.values())
        return {k: v / total for k, v in refined.items()}


class LLMContextFusion:
    """
    Advanced context fusion using a local LLM.

    Uses Llama-3-8B-Instruct or similar to reason about
    what the user might want to say given the conversation.
    """

    SYSTEM_PROMPT = """You are helping a person with speech difficulties communicate.
Given the recent conversation context and their brain-decoded intent probabilities,
determine what they most likely want to say.

Available responses: {vocab}

Rules:
1. Consider what makes sense given the conversation
2. Weight the brain signal probabilities heavily
3. Output ONLY the single most likely word from the vocabulary"""

    def __init__(
        self,
        model_path: str = 'models/llama-3-8b-instruct.gguf',
        vocab: List[str] = None
    ):
        """
        Initialize LLM fusion.

        Args:
            model_path: Path to GGUF model file
            vocab: List of possible output words
        """
        self.model_path = model_path
        self.vocab = vocab or ['yes', 'no', 'help', 'water', 'bathroom',
                               'pain', 'tired', 'hungry', 'cold', 'hot']
        self._llm = None

    def load(self) -> bool:
        """Load LLM model."""
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=32  # Offload to GPU
            )
            return True

        except ImportError:
            print("llama-cpp-python not installed")
            return False
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return False

    def refine(
        self,
        decoder_output: Dict[str, float],
        context: str
    ) -> str:
        """Use LLM to refine decoder output."""
        if self._llm is None:
            # Fallback to simple argmax
            return max(decoder_output, key=decoder_output.get)

        # Format prompt
        sorted_probs = sorted(decoder_output.items(), key=lambda x: -x[1])
        probs_str = ', '.join(f"{w}: {p:.1%}" for w, p in sorted_probs[:5])

        prompt = f"""Context: "{context}"

Brain signal probabilities: {probs_str}

Most likely response:"""

        response = self._llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            stop=['\n']
        )

        # Extract word from response
        result = response['choices'][0]['text'].strip().lower()

        # Validate against vocabulary
        for word in self.vocab:
            if word in result:
                return word

        # Fallback to argmax
        return max(decoder_output, key=decoder_output.get)


if __name__ == '__main__':
    # Test context fusion
    fusion = ContextFusion(boost_factor=2.0)

    # Simulated decoder output
    decoder_output = {
        'yes': 0.3,
        'no': 0.25,
        'help': 0.15,
        'water': 0.1,
        'bathroom': 0.05,
        'pain': 0.05,
        'tired': 0.05,
        'hungry': 0.03,
        'cold': 0.01,
        'hot': 0.01
    }

    # Test with different contexts
    contexts = [
        "Are you thirsty?",
        "Do you need to use the bathroom?",
        "Are you in pain?",
        "Is it too cold in here?"
    ]

    print("Context Fusion Test")
    print(f"Original top: {max(decoder_output, key=decoder_output.get)}")
    print()

    for context in contexts:
        result = fusion.refine(decoder_output, context)
        probs = fusion.refine_with_probs(decoder_output, context)
        print(f"Context: \"{context}\"")
        print(f"  Result: {result}")
        print(f"  Top-3: {sorted(probs.items(), key=lambda x: -x[1])[:3]}")
        print()
