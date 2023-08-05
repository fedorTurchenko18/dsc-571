from pydantic import BaseModel


class EkoUser(BaseModel):
    ciid: str