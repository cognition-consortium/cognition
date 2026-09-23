#!/usr/bin/env python


from typing import Optional, Dict, List
from enum import Enum
from beartype import beartype


class IDHStatus(Enum):
    """Enum for IDH status to ensure type safety."""
    WILDTYPE = "Wildtype"
    MUTANT = "Mutant"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str):
        """Resolve a string to an IDHStatus enum member."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid IDHStatus: {value}")


# https://braintumor.org/brain-tumors/about-brain-tumors/brain-tumor-types/
class TumorType(Enum):
    """Enum representing valid tumor types with associated IDH status."""
    ASTROCYTOMA = ("Astrocytoma", IDHStatus.MUTANT)
    ATRT = ("Atypical Teratoid/Rhabdoid Tumor", IDHStatus.WILDTYPE)
    CENTRAL_NEUROCYTOMA = ("Central Neurocytoma", IDHStatus.WILDTYPE)
    CPP = ("Choroid Plexus Papilloma", IDHStatus.WILDTYPE)
    DMG = ("Diffuse Midline Glioma", IDHStatus.WILDTYPE)
    DLGT = ("Diffuse Leptomeningeal Glioneuronal Tumor", IDHStatus.WILDTYPE)
    DNT = ("Dysembryoplastic Neuroepithelial Tumor", IDHStatus.WILDTYPE)
    EPENDYMOMA = ("Ependymoma", IDHStatus.WILDTYPE)
    ETMR = ("Embryonal Tumour With Multilayered Rosettes", IDHStatus.WILDTYPE)
    GANGLIOGLIOMA = ("Ganglioglioma", IDHStatus.WILDTYPE)
    GLIOBLASTOMA = ("Glioblastoma", IDHStatus.WILDTYPE)
    HEMANGIOBLASTOMA = ("Hemangioblastoma", IDHStatus.WILDTYPE)
    HGNET = ("High-grade Neuroepithelial Tumor", IDHStatus.WILDTYPE)
    LYMPHOMA = ("Lymphoma", IDHStatus.WILDTYPE)
    MEDULLOBLASTOMA = ("Medulloblastoma", IDHStatus.WILDTYPE)
    MENINGIOMA = ("Meningioma", IDHStatus.WILDTYPE)
    MYB_LGG = ("MYB LGG", IDHStatus.WILDTYPE)
    
    NON_TUMOR_BRAIN = ("Non tumor brain", None)
    NON_TUMOR_BLOOD = ("Non tumor blood", None)
    
    OLIGODENDROGLIOMA = ("Oligodendroglioma", IDHStatus.MUTANT)
    PLASMACYTOMA = ("Plasmacytoma", IDHStatus.MUTANT)
    PILOCYTIC_ASTROCYTOMA = ("Pilocytic Astrocytoma", IDHStatus.WILDTYPE)
    PRPT = ("Papillary Tumor Pineal Region", IDHStatus.WILDTYPE)
    PXA = ("Pleomorphic Xanthoastrocytoma", IDHStatus.WILDTYPE)
    SUBEPENDYMOMA = ("Subependymoma", IDHStatus.WILDTYPE)

    
    def __init__(self, label: str, idh_status: Optional[IDHStatus]):
        self._label = label
        self._idh_status = idh_status

    @property
    def name(self) -> str:
        return self._label

    @property
    def idh_status(self) -> Optional[IDHStatus]:
        return self._idh_status

    def verify_idh_status(self, status: Optional[IDHStatus]) -> bool:
        if self.idh_status is None or status is None:
            return True
        return self.idh_status == status

    def __str__(self):
        status = self.idh_status.value if self.idh_status else "Unknown"
        return f"{self.name}" #  (IDH: {status})

    @classmethod
    def from_name(cls, name: str) -> "TumorType":
        for tumor in cls:
            if tumor.name.lower() == name.lower():
                return tumor
        raise ValueError(f"Unknown tumor type: {name}")

    @classmethod
    def get_all(cls) -> list["TumorType"]:
        return list(cls)



class TumorTypeDatabase:
    def __init__(self):
        pass
    
    @staticmethod
    def get_by_idh_status(_idh_status):
        out = []
        
        for _ in list(TumorType):
            if _.idh_status == _idh_status:
                out.append(_)
        
        return out
    
    @staticmethod
    def __iter__():
        for _ in list(TumorType):
            yield _
    
    @staticmethod
    def get_by_name(_name):
        try:
            return TumorType.from_name(_name)
        except ValueError:
            return None


tumor_db = TumorTypeDatabase()


# Usage example
if __name__ == "__main__":
    g = TumorType.from_name("Glioblastoma")
    a = TumorType.from_name("Astrocytoma")
    
    print(g)
    print(a)
    
    #e = TumorType.from_name("Non existent")
    
    # Display all tumors
    print("All tumor types:")
    for tumor in tumor_db:
        print(f"  {tumor}")
    
    # Query by IDH status
    print(f"\nIDH Mutant tumors:")
    for tumor in tumor_db.get_by_idh_status(IDHStatus.MUTANT):
        print(f"  {tumor}")
    
    # Get specific tumor
    glioblastoma = tumor_db.get_by_name("Glioblastoma")
    if glioblastoma:
        print(f"\nFound: {glioblastoma}")

    # Get non existing type
    nonexistent = tumor_db.get_by_name("NoNeXisTenT")
    print(f"\nReponse on finding non-existent: {nonexistent}")
    
    # Example of IDH status verification
    print(f"\nIDH Status Verification Examples:")
    astrocytoma = tumor_db.get_by_name("Astrocytoma")
    if astrocytoma:
        print(f"Astrocytoma with Mutant IDH: {astrocytoma.verify_idh_status(IDHStatus.MUTANT)}")  # True
        print(f"Astrocytoma with Wildtype IDH: {astrocytoma.verify_idh_status(IDHStatus.WILDTYPE)}")  # False
        print(f"Astrocytoma with None IDH: {astrocytoma.verify_idh_status(None)}")  # True (unknown is valid)
    
    non_tumor = tumor_db.get_by_name("Non tumor brain")
    if non_tumor:
        print(f"Non-tumor brain with any IDH status: {non_tumor.verify_idh_status(IDHStatus.MUTANT)}")  # True (no defined status)
