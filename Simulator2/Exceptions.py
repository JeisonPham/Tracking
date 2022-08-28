class DuplicateNameException(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Name already exists in rendering window: {name}")
