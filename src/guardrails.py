class InputGuardrail:
    BLOCKED = [
        "ignore previous instructions",
        "ignore all instructions",
        "you are now",
        "pretend you are",
        "act as a different",
        "forget everything",
        "jailbreak",
        "system prompt",
        "disregard your",
    ]

    def validate(self, text: str):
        if not text or not text.strip():
            return False, "Please enter question"
        if len(text) < 3:
            return False, "Too short"
        if len(text) > 2000:
            return False, "Too long"
            
        lower_text = text.lower()
        for blocked in self.BLOCKED:
            if blocked in lower_text:
                return False, "Invalid input"
                
        return True, text.strip()

class OutputGuardrail:
    def check(self, answer: str, context: str):
        if not answer:
            return {"safe": False, "confidence": "low"}
        if len(answer) < 20:
            return {"safe": False, "confidence": "low"}
            
        # Basic word overlap check
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        overlap = answer_words.intersection(context_words)
        if len(overlap) > 0:
            return {"safe": True, "confidence": "high"}
            
        return {"safe": True, "confidence": "low"}
