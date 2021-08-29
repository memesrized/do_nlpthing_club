from pathlib import Path


class PathLoader:
    def __init__(self, path=None):
        if not path:
            self.main_dir = Path(__file__).resolve().parent.parent
        else:
            self.main_dir = Path(path)

        self.configs = self.main_dir / "configs"
        self.data = self.main_dir / "data"
        self.models = self.main_dir / "models"
        self.reports = self.main_dir / "reports"
        self.logs = self.main_dir / "logs"
