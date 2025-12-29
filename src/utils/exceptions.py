# ✔ Exception

# Exception is a built-in Python class

# Automatically available — no import needed

class NewsClassifierException(Exception):
    def __init__(self, error_message: str,):
        super().__init__(error_message)