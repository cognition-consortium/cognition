#!/usr/bin/env python3

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Union, Dict, List, Iterator, Optional
import glob
from tqdm import tqdm
import re
from abc import ABC, abstractmethod
from beartype import beartype
from beartype.typing import ClassVar
import json
import os

from .utils import interruptible


class BaseMethylscape(ABC):
    """Abstract base class for MethylScape parsers."""
    
    @beartype
    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialize with HTML report filepath."""
        self.filepath = Path(filepath)
        self.sentrix_id: str = None
        self.classification: str = ""
        self.class_score: float = 0.0
        self._parse()

    @abstractmethod
    def _parse(self) -> None:
        """Parse the HTML file and extract data. Must be implemented by subclasses."""
        pass

    @beartype
    def get_sentrix_id(self) -> str:
        """Get the Sentrix ID."""
        return self.sentrix_id
        
    @beartype
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Return parsed data as dictionary."""
        return {
            'array_sentrix_id': self.sentrix_id,
            'array_methylscape_bethesda_class': self.classification,
            'array_methylscape_bethesda_class_score': self.class_score
        }

    def __str__(self):
        if self.sentrix_id is not None:
            return f"{self.classification}[p={round(self.class_score,4)}]"
    
    @beartype
    def to_dataframe(self) -> pd.DataFrame:
        """Return parsed data as single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


class MethylscapeBv2(BaseMethylscape):
    """MethylScape Bethesda v2 classifier parser."""
    
    # Class constants with type annotations
    CLASSES: ClassVar[List[str]] = [
        "A_IDH", "A_IDH_HG", "AB_EWSR1_BEND2", "ACVR1", "AG_MYB", "ANTCON", "ARMS",
        "ASTROBL_CXXC5", "ASTROBL_MN1", "ATRT_MYC", "ATRT_SHH", "ATRT_TYR", "Bcor_ITD",
        "BRCA_A", "BRCA_B", "CHGL", "CHORD", "CNET", "CN", "COAD", "CONTR_ADENOPIT",
        "CONTR_CEBM", "CONTR_CORPRAL", "CONTR_HEMI", "CONTR_INFLAM", "CONTR_PINEAL",
        "CONTR_PONS", "CONTR_REACT", "CPC_AD", "CPC_PED", "CPH_ADM", "CPH_PAP", "CPP_AD",
        "CPP_PED", "CRINET", "CTRL_BLOOD", "CTRL_HYPOTHAL", "CTRL_OPTIC", "DG_F3T3_0",
        "DG_G34", "DGONC", "DIG_DIA", "DLGNT_1", "DLGNT_2", "DMT_SMARCB1", "DMG_EGFR",
        "DMG_K27", "DNT", "EFT_CIC", "ENDOSTL", "EP300_BCOR", "EPN_ACVR1", "EPN_MPE",
        "EPN_PFA_1A", "EPN_PFA_1B", "EPN_PFA_1C", "EPN_PFA_1D", "EPN_PFA_1E", "EPN_PFA_1F",
        "EPN_PFA_2A", "EPN_PFA_2C", "EPN_PFB_1", "EPN_PFB_2", "EPN_PFB_3", "EPN_PFB_4",
        "EPN_PFB_5", "EPN_SPINE", "EPN_SPINE_MYCN", "EPN_SPINE_SE_A", "EPN_SPINE_SE_B",
        "EPN_ST_ZFTA_FUS_C", "EPN_ST_ZFTA_FUS_D", "EPN_ST_ZFTA_RELA_A", "EPN_ST_ZFTA_RELA_B",
        "EPN_YAP", "EPEN_SUBEPN_PF", "EPEN_SUBEPN_ST", "ERMS", "ET_BRD4_LEUTX", "ET_PLAG",
        "EVNCYT", "EWS", "GBM_CBM", "GBM_MES_ATYP", "GBM_MES_TYP", "GBM_PNC", "GBM_RTK_I",
        "GBM_RTK_II", "GCT_GERM_A", "GCT_GERM_KIT", "GCT_YOLKSAC", "GG", "GNT_KinF_A",
        "HGAP", "HGG_B", "HGG_E", "HGG_F", "HMB", "HPAF", "IHG", "IHG_EMB", "ICMT_A",
        "ICMT_B", "ICMT_C", "IO_MEPL", "KIRC", "LCH", "LUAD", "MB_G34_I", "MB_G34_II",
        "MB_G34_III", "MB_G34_IV", "MB_G34_V", "MB_G34_VI", "MB_G34_VII", "MB_G34_VIII",
        "MB_MYO", "MB_SHH_1", "MB_SHH_2", "MB_SHH_3", "MB_SHH_4", "MB_WNT", "MELAN",
        "MELCYT", "MMNST", "MNG_BEN_1", "MNG_BEN_2", "MNG_BEN_3", "MNG_INT_A", "MNG_INT_B",
        "MNG_MAL", "MNG_SMARCE1", "MPNST", "MPNST_ATYP", "MYB_B", "MYB_C", "MYGNT",
        "NB_FOXR2", "NB_MYCN", "NB_TMM_NEG", "NB_TMM_POS", "NFIB_PLEX", "ONB", "O_IDH",
        "O_SARC_IDH", "PATZ1", "PA_CORT", "PA_INF_GFR", "PA_MID", "PA_PF", "PA_PF_A",
        "PB_FOXR2", "PB_GRP1A", "PB_GRP1B", "PB_RB", "PGNNT", "PIN_CYT", "PITAD_ACTH",
        "PITAD_FSH_LH", "PITAD_PRL", "PITAD_STH_DENSE1", "PITAD_STH_DENSE2", "PITAD_STH_SPA",
        "PITUI", "PLAGL1_FUS", "PLASMACYT", "PLNTY", "PPTID_A", "PPTID_B", "PPTR_A",
        "PPTR_B", "PXA", "RB", "RB_MYCN", "RGNT", "RMS_MYOD1", "SCHW", "SFT_HMPC",
        "SNUC_IDH2", "ULB_cl", "pedHGG_A", "pedHGG_MYCN", "pedHGG_RTK1A", "pedHGG_RTK1B",
        "pedHGG_RTK1C", "pedHGG_RTK2A", "pedHGG_RTK2B"
    ]
    
    class_mapping_to_tumor_types = {
        'O_SARC_IDH': 'Oligodendroglioma',
        'O_IDH': 'Oligodendroglioma',
        
        'A_IDH': 'Astrocytoma',
        'A_IDH_HG': 'Astrocytoma',
        
        'GBM_CBM': 'Glioblastoma',
        'GBM_MES_ATYP': 'Glioblastoma',
        'GBM_MES_TYP': 'Glioblastoma',
        'GBM_PNC': 'Glioblastoma',
        'GBM_RTK_I': 'Glioblastoma',
        'GBM_RTK_II': 'Glioblastoma'
        
        #'pedHGG_A': 'Glioblastoma'  # Pediatric high-grade glioma
    }
    
    @beartype
    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialize MethylScape Bv2 parser."""
        super().__init__(filepath)
    
    def _parse(self) -> None:
        """Parse the HTML file and extract Bv2-specific data."""

        fn_json = str(self.filepath) + '.json'
        
        if os.path.exists(fn_json):
            with open(fn_json, 'r') as f:
                data = json.load(f)
            
                self.sentrix_id = data['sentrix_id']
                self.classification = data['classification']
                self.class_score = data['class_score']

        else:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser') # lxml
            
            # Find the target div
            div = soup.find('div', id='bethesda-classifier-v2')
            if not div:
                raise ValueError(f"Could not find bethesda-classifier-v2 div in {self.filepath}")
                
            tables = div.find_all('table', limit=2)
            if len(tables) < 2:
                raise ValueError(f"Expected at least 2 tables in {self.filepath}")
            
            # Extract Sentrix ID from filename using regex
            match = re.search(r'-Bv2_([^.]+)\.html', str(self.filepath))
            if not match:
                raise ValueError(f"Could not extract Sentrix ID from filename: {self.filepath}")
            self.sentrix_id = match.group(1)
            
            # Extract classification and score from HTML
            self.classification = tables[1].find('tbody').find('tr').find_all('td', limit=2)[1].get_text(strip=True)
            score_text = tables[0].find('tbody').find_all('tr', limit=5)[4].find_all('td', limit=2)[1].get_text(strip=True)

            self.class_score = float(score_text)

            with open(fn_json, 'w') as f2:
                json.dump({'sentrix_id': self.sentrix_id, 'classification': self.classification, 'class_score': self.class_score}, f2)

    
    @classmethod
    def map_to_tumor_type(cls, mnp_label):
        try:
            return cls.class_mapping_to_tumor_types[mnp_label]
        except:
            return "Other"
    
    @beartype
    def get_classification(self) -> str:
        """Get tumor type for this sample's classification."""
        return self.map_to_tumor_type(self.classification)


class MethylscapeDatabase:
    """Database manager for MethylScape files."""
    
    @beartype
    def __init__(self, base_path: Union[str, Path] = "data/DNA_methylation/methylscape/") -> None:
        """Initialize database with base path."""
        self.base_path = Path(base_path)
        self.idx = {}
        self._scan_database()
    
    def _scan_database(self) -> None:
        """Scan for available MethylScape files and parse them."""
        bv2_pattern = str(self.base_path / "Bv2" / "*.html")
        bv2_files = glob.glob(bv2_pattern, recursive=True)
        
        if bv2_files:
            # Skippable with 'q': the index stays partial for the rest of the
            # session, so get_by_sentrix_id returns None for whatever was not
            # reached. Only skip when the MethylScape lookups do not matter.
            with interruptible() as quit_event:
                for filepath in tqdm(bv2_files, desc="Parsing MethylScape Bv2 files (q to skip)"):
                    if quit_event.is_set():
                        tqdm.write(f"Parsing MethylScape Bv2 files interrupted by user ({len(self.idx)}/{len(bv2_files)} parsed)")
                        break

                    try:
                        parser = MethylscapeBv2(filepath)
                        self.idx[parser.get_sentrix_id()] = parser
                    except Exception as e:
                        print(f"Warning: Failed to parse {filepath}: {e}")

    
    @beartype
    def __len__(self) -> int:
        """Return number of successfully parsed files."""
        return len(self.idx)

    @beartype
    def __iter__(self) -> Iterator[pd.Series]:
        """Iterate over database rows."""
        df = self.to_dataframe()
        return iter(row for _, row in df.iterrows())

    @beartype
    def get_by_sentrix_id(self, sentrix_id: Optional[str])-> Optional[MethylscapeBv2]:
        """Get Methylscape by Sentrix ID."""
    
        return self.idx.get(sentrix_id)
           
    @beartype
    def get_classification_by_sentrix_id(self, sentrix_id: str) :
        """Get Methylscape by Sentrix ID."""
        matches = self.get_by_sentrix_id(sentrix_id)
        if matches is not None:
            return matches.get_classification()
        else:
            return None
    
    @beartype
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all parsed data to a combined DataFrame."""
        if not self.idx:
            return pd.DataFrame()

        return pd.DataFrame([parser.to_dict() for parser in self.idx.values()])
    
    @beartype
    def get_tumor_type_summary(self) -> pd.DataFrame:
        """Get summary of tumor types in the database."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        df['tumor_type'] = df['array_methylscape_bethesda_class'].apply(
            MethylscapeBv2.map_to_tumor_type
        )
        return df['tumor_type'].value_counts().reset_index()





# Usage example
if __name__ == "__main__":
    # Create database instance
    methylscape_db =  MethylscapeDatabase("data/DNA_methylation/methylscape/")
    
    print(f"Database contains {len(methylscape_db)} files")
    
    if len(methylscape_db) > 0:
        # Show summary
        print("\nFirst few records:")
        summary_df = methylscape_db.to_dataframe()
        print(summary_df.head())
        
   
