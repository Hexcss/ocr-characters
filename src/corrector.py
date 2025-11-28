import os
import difflib

class SpellCorrector:
    def __init__(self, wordlist_path='data/words.txt'):
        self.words = set()
        self.word_list = []
        
        # Load dictionary
        if os.path.exists(wordlist_path):
            with open(wordlist_path, 'r') as f:
                # Read words, strip whitespace, convert to uppercase to match OCR output
                content = f.read().splitlines()
                self.word_list = [w.upper() for w in content]
                self.words = set(self.word_list)
            print(f"✅ Spell Checker Loaded: {len(self.words)} words.")
        else:
            print(f"⚠️ Warning: {wordlist_path} not found. Spell check will do nothing.")

    def correct(self, text):
        """
        Input: "HEL10"
        Output: "HELLO"
        """
        text = text.upper().strip()
        
        # 1. If it's already a valid word, don't touch it.
        if text in self.words:
            return text
        
        # 2. Find the closest match using Sequence Matcher (Levenshtein distance)
        # cutoff=0.6 means "must be at least 60% similar"
        matches = difflib.get_close_matches(text, self.word_list, n=1, cutoff=0.6)
        
        if matches:
            # We found a close match!
            print(f"   (Auto-Corrected '{text}' -> '{matches[0]}')")
            return matches[0]
        
        # 3. No close match found? Return original.
        return text

# Simple test block
if __name__ == "__main__":
    corrector = SpellCorrector()
    print(corrector.correct("W0RLD")) # Should print WORLD
    print(corrector.correct("PYTH0N")) # Should print PYTHON