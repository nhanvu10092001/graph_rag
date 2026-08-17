from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class File:    
    documents_path: Optional[str] = None
    documents_name: Optional[str] = None
    collection_name: Optional[str] = None
    metadata: Optional[dict] = None