from abc import ABC, abstractmethod

class BaseInspector(ABC):
    def __init__(self, model_path, vectorizer_path=None, output_dir="plots/inspect"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.output_dir = output_dir

    @abstractmethod
    def analyze(self):
        """Run the full inspection and generate plots/reports."""
        pass
