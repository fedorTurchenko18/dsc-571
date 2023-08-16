from pydantic import BaseModel


class EkoUser(BaseModel):
    ciid: str

class EkoSales(BaseModel):
    sales: str

class EkoCustomers(BaseModel):
    customers: str