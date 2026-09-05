class Node:

    def __init__(self, name: str, index: int, type: str, summary: str):
        self.name = name
        self.index = int(index)
        self.type = str(type)
        self.summary = summary if summary else ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, index={self.index}, type={self.type})"

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self):
        return hash((self.name, self.index, self.type))

    def to_doc(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "type": self.type,
            "summary": self.summary,
        }

    @classmethod
    def from_doc(cls, doc):
        return cls(
            name=doc["name"][0],
            index=doc["index"][0],
            type=doc["type"][0],
            summary=doc["summary"][0],
        )

    @classmethod
    def from_df_row(cls, row):
        return cls(
            name=row.get("name", ""),
            index=row.get("index"),
            type=row.get("type", ""),
            summary=row.get("summary", ""),
        )
