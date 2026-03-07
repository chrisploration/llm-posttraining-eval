class PostTrainError(Exception):
    """Base for all project errors"""

class ConfigError(PostTrainError):
    """Invalid / Missing configuration"""

class CheckpointError(PostTrainError):
    """Checkpoint save / load failure"""

class EvalError(PostTrainError):
    """Evaluation pipeline failure"""

class DataError(PostTrainError):
    """Training / Evaluation data issue"""