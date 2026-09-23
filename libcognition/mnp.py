#!/bin/python

import pandas as pd
import glob
from tqdm import tqdm
from functools import reduce

from .utils import interruptible


class mnp():
    def __init__(self, fn):
        self.data = pd.read_csv(fn)
        self.data.rename(columns={'Unnamed: 0':'Class'}, inplace=True)
    
    def get_sentrix_id(self):
        out = self.data.columns[1]
        
        # do some regex matching?!
        
        return out

    def __str__(self):
        dat = self.data
        dat = dat.sort_values(by=dat.columns[1], ascending=False)
        row = dat.iloc[0]
        #print(row)
        
        result = f"{row[dat.columns[0]]}[p={round(row[dat.columns[1]],4)}]"
        
        return result

    def check_rows(self):
        assert(self.data.shape[0] == len(self.classes))
        assert(self.data.shape[1] == 2)
        
        assert self.data['Class'].isin(self.classes).all()


class mnp_v12_8(mnp):
    classes = ['A_IDH_LG', 'AG_MYB', 'LGG_MYB_B', 'LGG_MYB_C', 'LGG_MYB_D', 'GTAKA', 'HGAP', 'ATRT_MYC',
    'ATRT_SHH', 'ATRT_TYR', 'CHGL', 'CHORDM', 'CN', 'CNS_NB_FOXR2', 'CNS_SARC_DICER', 'CPC_PED', 'CPC_AD',
    'CPP_AD', 'CPP_PED', 'CPH_ADM', 'CPH_PAP', 'DGONC', 'DIG_DIA', 'DLGNT_1', 'DLGNT_2', 'GNT_A', 'DMG_K27',
    'DNET', 'CNS_SARC_CIC', 'ONB', 'SNUC_IDH2', 'CNS_BCOR_FUS', 'EPN_MPE', 'EPN_PF_SE', 'EPN_SPINE',
    'EPN_SPINE_MYCN', 'EPN_SPINE_SE_B', 'EPN_SPINE_SE_A', 'NET_PLAGL1_FUS', 'EPN_ST_SE', 'EPN_YAP',
    'ETMR_C19MC', 'ETMR_Atyp', 'EVNCYT', 'EWS', 'GBM_CBM', 'DHG_G34', 'A_IDH_HG', 'INFLAM_ENV', 'GBM_MES_TYP',
    'GBM_MES_ATYP', 'pedHGG_MYCN', 'pedHGG_RTK1A', 'pedHGG_RTK1B', 'pedHGG_RTK1C', 'pedHGG_RTK2A',
    'pedHGG_RTK2B', 'GBM_RTK1', 'GBM_RTK2', 'DMG_EGFR', 'GCT_GERM_A', 'GCT_GERM_KIT', 'GCT_TERA', 'GCT_YOLKSAC',
    'GG', 'CTRL_REACTIVE', 'pedHGG_A', 'pedHGG_B', 'CNS_BCOR_ITD', 'NET_CXXC5', 'ABM_MN1', 'GBM_PNC', 'HGG_B',
    'HGG_E', 'HGG_F', 'NET_PATZ1', 'ET_PLAG', 'HMB', 'SFT_HMPC', 'IHG', 'IO_MEPL', 'LCH', 'LIPN', 'ET_BRD4_LEUTX',
    'MB_MYO', 'MB_SHH_4', 'MB_SHH_IDH', 'MB_SHH_1', 'MB_SHH_2', 'MB_SHH_3', 'MB_WNT', 'MB_G34_I', 'MB_G34_II',
    'MB_G34_III', 'MB_G34_IV', 'MB_G34_V', 'MB_G34_VI', 'MB_G34_VII', 'MB_G34_VIII', 'MMNST', 'MELN', 'MET_MEL',
    'MNG_BEN_1', 'MNG_BEN_2', 'MNG_BEN_3', 'MNG_SMARCE1', 'MNG_INT_A', 'MNG_INT_B', 'MNG_MAL', 'MPNST_TYP',
    'MPNST_ATYP', 'MYXGNT', 'CTRL_ADENOPIT', 'CTRL_CBM', 'CTRL_CORPCAL', 'CTRL_HEMI', 'CTRL_HYPOTHAL', 'CTRL_OPTIC',
    'CTRL_PIN', 'CTRL_PONS', 'O_IDH', 'PA_CORT', 'PA_INF', 'PA_INF_FGFR', 'PA_MID', 'PB_GRP1A', 'PB_GRP1B', 'PB_GRP2',
    'PB_FOXR2', 'DLBCL', 'PLASMACYT', 'EPN_PFA_1A', 'EPN_PFA_1B', 'EPN_PFA_1C', 'EPN_PFA_1D', 'EPN_PFA_1E', 'EPN_PFA_1F',
    'EPN_PFA_2A', 'EPN_PFA_2B', 'EPN_PFA_2C', 'EPN_PFB_1', 'EPN_PFB_2', 'EPN_PFB_3', 'EPN_PFB_4', 'EPN_PFB_5',
    'CAUDEQU_NET', 'PGNT', 'PIN_CYT', 'PIN_RB', 'PITAD_ACTH', 'PITAD_GON', 'PITAD_PRL', 'PITAD_STH_DENSE1',
    'PITAD_STH_DENSE2', 'PITAD_STH_SPARSE', 'PITAD_TSH', 'PITUI', 'PLNTY', 'PPTID_A', 'PPTID_B', 'PTPR_A', 'PTPR_B',
    'PXA', 'RB', 'RB_MYCN', 'EPN_ST_ZFTA_FUS_C', 'EPN_ST_ZFTA_FUS_D', 'EPN_ST_ZFTA_FUS_E', 'RGNT', 'SCHW', 'SEGA',
    'EPN_ST_ZFTA_RELA_A', 'EPN_ST_ZFTA_RELA_B', 'CNS_SCHW_VGLL', 'NB_MYCN', 'NB_TMM_NEG', 'NB_TMM_POS', 'CTRL_BLOOD',
    'NFIB_PLEX', 'OLIGOSARC_IDH', 'ERMS', 'CRINET', 'ARMS', 'RMS_MYOD1']
    
    class_mapping_to_tumor_types = {
        'OLIGOSARC_IDH': 'Oligodendroglioma',
        'O_IDH': 'Oligodendroglioma',
        
        'A_IDH_LG': 'Astrocytoma',
        'A_IDH_HG': 'Astrocytoma',
        
        'GBM_CBM': 'Glioblastoma',
        'GBM_MES_ATYP': 'Glioblastoma',
        'GBM_MES_TYP': 'Glioblastoma',
        'GBM_PNC': 'Glioblastoma',
        'GBM_RTK1': 'Glioblastoma',
        'GBM_RTK2': 'Glioblastoma'
        
        #'pedHGG_A': 'Glioblastoma' # this is difficult, as can also be considered pedriatic cases?
    }
    
    def __init__(self, fn):
        super().__init__(fn)
        self.check_rows()
    
    @classmethod
    def map_to_tumor_type(cls, mnp_label):
        try:
            return cls.class_mapping_to_tumor_types[mnp_label]
        except:
            return "Other"
        


class mnp_db:
    _path = "data/DNA_methylation/epignostix/"
    
    def __init__(self):
        self.idx = {}
        self.db_scan()
    
    def db_scan(self):
        db_files = pd.DataFrame({'fn_v12.8': glob.glob(self._path + '/v12.8/*_cal.csv', recursive=True)})
        
        # Skippable with 'q': the index stays partial for the rest of the
        # session, so get_by_sentrix_id returns None for whatever was not
        # reached. Only skip when the MNP lookups do not matter for the run.
        with interruptible() as quit_event:
            # Wrap iterrows with tqdm and add description
            for index, row in tqdm(db_files.iterrows(),
                                  total=len(db_files),
                                  desc="Parsing MNP v12.8 files (q to skip)"):
                if quit_event.is_set():
                    tqdm.write(f"Parsing MNP v12.8 files interrupted by user ({len(self.idx)}/{len(db_files)} parsed)")
                    break

                obj = mnp_v12_8(row['fn_v12.8'])
                self.idx[obj.get_sentrix_id()] = obj

    def get_by_sentrix_id(self, sentrix_id):
        return self.idx.get(sentrix_id)



