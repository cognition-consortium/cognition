#!/usr/bin/env python

from . import __version__

import functools
from typing import Literal
import os
import glob
import gzip
import html
from urllib.parse import unquote
import numpy as np
import hdf5plugin
import h5py
import pandas as pd
import random
import re
import requests
import shutil
import socket
import sys
import tempfile
import time
import termios
import tty
import zipfile
import threading

from mepylome import MethylData, ReferenceMethylData, CNV


from threading import Lock
file_lock = Lock()

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from beartype import beartype
from idattools.idat import IDATreader
from pathlib import Path
from tqdm.auto import tqdm
tqdm.pandas()

from pymetharray.files import IdatDataset, SampleSheet
from pymetharray.models import ArrayType
from pymetharray.models import Channel

from .utils import *
from .tumortypes import *
from .tumorlocations import TumorLocation
from .mnp import *
from .methylscape import *


import logging
for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
    logger.setLevel(logging.WARNING)



class idat:
    """One idat file on disk: where it is, how to obtain it, what its header says.

    Everything here is a property of the file itself and needs no database. The
    comparisons against the db columns (does the sentrix id match the row, is
    the array type the expected one) live in database.check_idats.

    `filename` and `channel` are the db values this file was derived from; they
    are kept because the file cannot reconstruct them: _path has a trailing
    '.gz' stripped where the db 'Filename' may still carry it, and the physical
    channel of a file is not authoritative from its name (see detected_channel).
    """

    @beartype
    def __init__(self, path: str, verify_file: bool = True,
                 filename: str | None = None, channel: str | None = None):
        self._path = path
        self._filename = filename
        self._channel = channel
        self._header_cache = None

        if verify_file:
            if os.path.exists(path):
                try:
                    decoy = IDATreader(Path(path))
                except:
                    raise Exception(path + ": Invalid IDAT")

    @property
    def path(self) -> str:
        return self._path

    @beartype
    def file_exists(self) -> bool:
        return os.path.exists(self._path)

    @beartype
    def _read_header(self) -> dict:
        """Sentrix id and probe count from the idat header, read once per object.

        header_only keeps this cheap: no intensities are decoded. Both values
        come from the same read because opening the file twice for two
        properties would double the I/O over the full database.
        """
        if self._header_cache is None:
            channel_enum = {'Grn': Channel.GREEN, 'Red': Channel.RED}.get(self._channel)
            if channel_enum is None:
                raise ValueError(f"{self._path}: unknown channel '{self._channel}', cannot read header")

            dataset = IdatDataset(self._path, channel_enum, header_only=True)
            self._header_cache = {
                # corner case with some odd hacked sentrix id's in geo files
                'sentrix_id':  re.sub(r'^([^_]+_[^_]+)_1$', r'\1', dataset.get_sentrix_id()),
                'n_snps_read': dataset.n_snps_read,
            }

        return self._header_cache

    @property
    @beartype
    def sentrix_id(self) -> str:
        return self._read_header()['sentrix_id']

    @property
    @beartype
    def array_type(self) -> str:
        return str(ArrayType.from_probe_count(self._read_header()['n_snps_read']))

    @property
    @beartype
    def detected_channel(self) -> str:
        """The physical channel, from the control-probe intensities.

        The _Grn/_Red suffix in the file name is not authoritative; this reads
        what the file actually is (see utils.idat_channel), which is what
        catches swapped Grn/Red files.
        """
        return idat_channel(self._path)

    @beartype
    def restore_from_backup(self) -> bool:
        """Move <path>.bak back into place; returns whether there was one."""
        backup_path = self._path + ".bak"

        if not os.path.exists(backup_path):
            return False

        print(f"WARNING: found backup file, restoring: {backup_path} => {self._path}")
        shutil.move(backup_path, self._path)
        return True

    @beartype
    def download(self, url: str):
        """Fetch this idat from *url*, dispatching on the url's protocol."""
        protocol = url.split("://", 1)[0]
        
        print("DOWNLOAD")

        if protocol in ["http", "https"]:
            if url.find("ftp.ebi.ac.uk/pub") > -1 or url.find("amazonaws.com") > -1:
                self.download_mtab_zip_by_url(url, self._zip_member_name())
            else:
                self.download_by_url(url)
        elif protocol == "gdc":
            self.gdc_download(url)
        else:
            raise Exception(f"{self._path}\n\nUnknown protocol: {url}")

    @beartype
    def _zip_member_name(self) -> str:
        """Name this idat carries inside an mtab zip: db Filename + _<channel>.idat.

        Built from the db 'Filename' rather than from _path: the zip member
        follows the db value, which may still end in '.gz' where _path has that
        suffix stripped.
        """
        if self._filename is None or self._channel is None:
            raise ValueError(f"{self._path}: filename/channel unknown, cannot resolve zip member")

        return self._filename.strip() + "_" + self._channel + ".idat"

    @beartype
    def gdc_download(self, url: str):
        code = url.split("//",1)[1]
        
        try:
            gdc_client_download(code, self._path)
        except:
            print("GDC client errored out") # happens too often to throw exception

    @beartype
    def download_mtab_zip_by_url(self, url: str, idat: str):
        basename_zip = os.path.basename(url)
        basename_idat = os.path.basename(idat)
        fn_zip = "cache/" + basename_zip
        fn_idat = "cache/" + basename_idat

        if not os.path.exists(os.path.dirname(self._path)):
            os.makedirs(os.path.dirname(self._path))

        if not os.path.exists(fn_zip):
            print("Downloading [zip]: "+url+"")
            response = requests.get(url)
            
            if response.status_code == 200:
                # make path recursively if not exists
                if not os.path.exists("cache/"):
                    os.makedirs(os.path.dirname("cache/"))
                
                with open(fn_zip, 'wb') as fh:
                    fh.write(response.content)
        
        if not os.path.exists(fn_idat):
            with zipfile.ZipFile(fn_zip, "r") as zip_ref:
                for member in zip_ref.infolist():
                    # Ignore directories
                    if member.is_dir():
                        pass
                    else:
                        filename = os.path.basename(member.filename)

                        # Process only .idat and .idat.gz files
                        if filename.endswith(".idat") or filename.endswith(".idat.gz"):
                            out_path = os.path.join('cache/', filename)

                            # Extract and write file
                            with zip_ref.open(member) as src, open(out_path, "wb") as dst:
                                dst.write(src.read())


        if os.path.exists(fn_idat):
            try:
                decoy = IDATreader(Path(fn_idat))
            except:
                raise Exception(fn_idat + ": Invalid IDAT?")

            return shutil.move(fn_idat, self._path)
        else:
            raise Exception('Failed to extract file')
    
    @beartype
    def download_by_url(self, url: str):
        # URL kan HTML- en/of percent-encoded zijn (bijv. &amp; of %5F) -> decoderen
        url = html.unescape(url)
        url = unquote(url)

        print(f"Downloading [idat]: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            # make path recursively if not exists
            if not os.path.exists(os.path.dirname(self._path)):
                os.makedirs(os.path.dirname(self._path))

            # write to file
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(response.content)
            tmp.close()

            if is_gz_file(tmp.name):
                with gzip.open(tmp.name, 'rb') as f_in:
                    with open(tmp.name + ".idat", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        os.unlink(tmp.name)
                        tmp.name = tmp.name + ".idat"
            try:
                tmp2 = IDATreader(Path(tmp.name))
            except:
                raise Exception(f"{url} -> {tmp.name}: Invalid IDAT?")

            return shutil.move(tmp.name, self._path)  # Path.rename cannot handle symlinks

        else:
            raise Exception('Failed to download file')
    
    
    def __str__(self):
        return "path: " + self._path


class database:
    _db_file = "data/database.txt"
    _idat_path = "data/DNA_methylation/idat/"
    _hdf5_path = "data/DNA_methylation/hdf5/"
    _hdf5_mepylome_path = "data/DNA_methylation/hdf5-mepylome/"
    _cnv_path = "data/DNA_methylation/CNV-mepylome"
    _idat_normals_path = "data/DNA_methylation/idat_normals"

    _master_hdf5_pymetharray = f"data/DNA_methylation/master_database_merged_table.v{__version__}.hdf5"

    _master_hdf5_mepylome = {
        'mvalues_illumina': f"data/DNA_methylation/master_database_m_m_i_merged_table.v{__version__}.hdf5",  # normalization type i, m-values
        'mvalues_swan':     f"data/DNA_methylation/master_database_m_m_s_merged_table.v{__version__}.hdf5",  # normalization type s, m-values, seemingly the best
        'mvalues_noob':     f"data/DNA_methylation/master_database_m_m_n_merged_table.v{__version__}.hdf5",  # normalization type n, m-values

        'betas_illumina':   f"data/DNA_methylation/master_database_m_b_i_merged_table.v{__version__}.hdf5",  # normalization type i, betas
        'betas_swan':       f"data/DNA_methylation/master_database_m_b_s_merged_table.v{__version__}.hdf5",  # normalization type s, betas
        'betas_noob':       f"data/DNA_methylation/master_database_m_b_n_merged_table.v{__version__}.hdf5",  # normalization type n, betas
        
        'intensities_illumina':   f"data/DNA_methylation/master_database_m_i_i_merged_table.v{__version__}.hdf5",  # normalization type i, intensities (control-scaled, matches CNV)
        #'intensities_swan':       f"data/DNA_methylation/master_database_m_i_s_merged_table.v{__version__}.hdf5",  # normalization type s, intensities
        #'intensities_noob':       f"data/DNA_methylation/master_database_m_i_n_merged_table.v{__version__}.hdf5",  # normalization type n, intensities

    }

    # Master CNV matrix: bins x samples, per-bin Median (log2 ratio). The shared
    # bin grid is only a few thousand rows and CNV is generated for a subset of
    # samples, so this stays small -> a single parquet file is more efficient
    # (binary columns + zstd, no per-cell text formatting / single-threaded gzip)
    # than both a gzip CSV and the chunked HDF5 master used for the m-values.
    _master_cnv = f"data/DNA_methylation/master_database_cnv_bins.v{__version__}.hg38.parquet"

    def __init__(self):
        self._samples = []
                
        self.mnp_db_v12_8 = mnp_db()
        self.methylscape_db_Bv2 = MethylscapeDatabase("data/DNA_methylation/methylscape/")
        
        dfs = (pd.read_csv(self._db_file, sep="\t")
               .dropna(how='all')
               .loc[lambda df: df["Discarded"].isin(["", "-"]) | df["Discarded"].isna()])
        
        assert(dfs.shape[0] > 0)

        @beartype
        def error_on_duplicates(df, column: str):
            is_duplicate_non_null = df[column].notna() & df[column].duplicated(keep=False)
            duplicate_rows = df[is_duplicate_non_null]
        
            if not duplicate_rows.empty:
                display_df = duplicate_rows[[column, "Dataset"]].sort_values(by=[column, 'Dataset'])
        
                msg = (
                    f"Remove duplicates in '{column}' before proceeding:\n"
                    f"{display_df.to_string(index=False)}"
                )
                raise ValueError(msg)

        for column in ["Sample ID", "Filename", "Sentrix ID"]:
            error_on_duplicates(dfs, column)
        
        with tqdm(total=dfs.shape[0], desc="Parsing database") as pbar:   
            for row in dfs.iterrows():
                row = row[1]
                
                if not pd.isna(row['Filename']):
                    row['idat'] = {'Grn': None, 'Red': None}
                    for channel in ['Grn', 'Red']:
                        row['idat'][channel] = idat(self._idat_path + row['Dataset'] + "/" + re.sub(r'.gz$', '', row['Filename'] ).strip() + "_" + channel + ".idat", False,
                                                    filename=row['Filename'], channel=channel)

                    row['hdf5-pymetharray'] = self._hdf5_path + row['Dataset'] + "/" + re.sub(r'.gz$', '', row['Filename'] ).strip() + ".hdf5"
                    row['hdf5-mepylome'] = self._hdf5_mepylome_path + row['Dataset'] + "/" + re.sub(r'.gz$', '', row['Filename'] ).strip() + ".hdf5"
                    
                    self.add_sample(row)

                else:
                    pass # print("Warning: missing Filename: " + str(row))
                
                pbar.update(1)
        

    @classmethod
    def get_database_file_columns(cls):
        with open(cls._db_file, 'r') as fh:
            header = fh.readline()
        
        return re.sub(r'^[ \f\r\v\n]+|[ \f\r\v\n]+$', '', header).split("\t")

    
    @functools.cached_property
    @beartype
    def get_all_probes_pymetharray(self) -> list:
        raise Exception("Error!")
        all_probes = set()

        samples = self.to_df().dropna(subset=['Array type']).drop_duplicates(subset='Array type', keep='first')
        for index, row in samples.iterrows():
            df = pd.read_hdf(row['hdf5-pymetharray'], 'methylation')
            
            cgps = [re.sub(r'_.*$', '', _) for _ in df.index] # strip suffixes in epicv2
            all_probes |= set(cgps)
        
        return sorted(list(all_probes))


    @functools.cached_property
    @beartype
    def get_all_probes_mepylome(self) -> list:
        all_probes = set()

        samples = self.to_df().dropna(subset=['Array type']).drop_duplicates(subset='Array type', keep='first')
        for index, row in samples.iterrows():
            #print(f"::{row['hdf5-mepylome']}")
            df = pd.read_hdf(row['hdf5-mepylome'])
            
            cgps = [re.sub(r'_.*$', '', _) for _ in df.index] # strip suffixes in epicv2
            all_probes |= set(cgps)
            print(f" ==> {len(all_probes)}")

        return sorted(list(all_probes))


    @functools.cached_property
    @beartype
    def get_all_cnv_bins(self) -> list:
        """Bin identifiers for the shared CNV bin grid, as 'chr:start-end'.

        All array types are re-gridded onto one shared 450k bin grid (see
        libcognition.mepylome_helpers), so every per-sample cnv_bins.hg38.csv
        carries exactly these bins. This is the CNV analogue of
        get_all_probes_mepylome and serves as the master matrix' row index. It
        is derived from the shared grid (not from any sample file), so it does
        not depend on which CNV outputs happen to exist yet.
        """
        from .mepylome_helpers import mepylome_shared_bins

        bins = pd.DataFrame(mepylome_shared_bins()).sort_values('bins_index')
        return [f"{c}:{s}-{e}" for c, s, e in
                zip(bins['Chromosome'], bins['Start'], bins['End'])]



    def check(self, check_pymetharray: bool,
                    check_mepylome: Literal[False] | list[str],
                    autocomplete: bool = False,
                    check_cnv: bool = False,
                    force_validating_sentrix_ids: bool = False,
                    force_checking_concordance_individual_master_hdfs: bool = False):
        hostname = socket.gethostname()
        
        print("Hostname: " + hostname)
        print("GPU:", re.match(r"gpu", hostname))
        print("Autoocomplete:", autocomplete)
        if hostname.find("snellius.surf") == -1 and not re.match(r"^int[0-9]$", hostname) and not re.match(r"rad-hpc", hostname) and not re.match(r"gpu", hostname) and not re.match(r".+cluster", hostname): # idats and hdf5s are not stored over there
            self.check_idats(autocomplete, force_validating_sentrix_ids) # 1a. check if all entries have idats
        else:
            print("Not testing presence of *.idat files (we're on snellius)", flush=True)
        
        self.check_mnp_ids(autocomplete)
        
        
        # 1b. check if no idats exist beyond the current db
        self.validate_all_idat_files_are_in_db(autocomplete)
        # 1c. check if metadata is appended to db:
        self.check_empty_values()
        
        
        
        if hostname.find("snellius.surf") == -1 and not re.match(r"^int[0-9]$", hostname) and not re.match(r"rad-hpc", hostname) and not re.match(r"gpu", hostname) and not re.match(r".+cluster", hostname): # idats and hdf5s are not stored over there
            # 2. per sample hdf5's
            # 2a. check if all entries have hdf5
            self.check_hdf5s(check_pymetharray, check_mepylome, autocomplete)

            # check if the normals are simlinked (in)to ./data/DNA methylation/idat_normals/<platform>
            # must run BEFORE check_cnvs: that step builds the CNV reference from
            # this symlink pool, so a not-yet-reconciled pool would feed a stale
            # (wrong-count) reference into the CNV generation below.
            self.validate_normal_idats(autocomplete)

            # 2c. check if all entries have CNV calls
            if check_cnv:
                self.check_cnvs(autocomplete)
        else:
            # even without the full per sample hdf5 collection, the data mount must be there
            self.check_hdf5_sentinels()


        # 2b. check if no hdf5 exist beyond the current db
        self.validate_all_hdf5_files_are_in_db(
            validate_pymetharray=check_pymetharray, # to be deprecated
            validate_mepylome=True,
            autocomplete= autocomplete
        )


        # 3. master hdf5 (m-values)
        # 3a. check if all entries are in mater hdf5 + check if no columns in master hdf5 exist beyond the current db
        if isinstance(check_mepylome, list):
            for suffix in check_mepylome:
                print("autocomplete:", autocomplete)
                self.check_master_hdf5_mepylome(suffix, autocomplete)
        if check_pymetharray:
            self.check_master_hdf5_pymetharray(autocomplete)
        # 3b. master CNV matrix (bins x samples, per-bin Median), opt-in.
        if check_cnv:
            self.check_cnv_master(autocomplete)
        
        
        # 4. check if contents are identcal by random sampling and comparing
        if check_pymetharray:
            if force_checking_concordance_individual_master_hdfs:
                self.verify_hdf5_in_sync_with_master_hfd5(2000, 10000)

    
    def check_mnp_ids(self, autocomplete: bool = False):
        metadata = self.to_df()

        error = False
        if autocomplete:
            nonmatching = set(self.mnp_db_v12_8.idx.keys()) - set(metadata['Sentrix ID'])
            
            for x in nonmatching:
                print(f"Moving: {x} -> {x}.bak")
                os.rename(x, x + '.bak')
                error = True

        if error:
            raise Exception("MNP files were moved")
        
        
    
    @beartype
    def check_idats(self, autocomplete: bool = False,
                          force_validating_sentrix_ids: bool = False):
        errors = []
        entries = list(self)

        # parallel existence checks — I/O-bound, threads release the GIL on syscalls
        pairs = [(entry, ch) for entry in entries for ch in ['Grn', 'Red']]

        def _check(entry_channel):
            entry, channel = entry_channel
            return entry, channel, entry['idat'][channel].file_exists()

        with ThreadPoolExecutor(max_workers=32) as executor:
            existence = list(tqdm(
                executor.map(_check, pairs),
                total=len(pairs),
                desc="Checking *.idat presence",
            ))

        for entry, channel, exists in existence:
            if not exists:
                print("NOT EXISTS")
                if autocomplete:
                    print("AUTOCOMPLETE")
                    if not entry['idat'][channel].restore_from_backup():
                        print("NOT RESTORED FROM BACKUP")
                        if pd.isna(entry['url ' + channel]):
                            print("A")
                            errors.append(f"Non existing entry {entry['Filename']} ({entry['idat'][channel]}) with no existing url to download file")
                        else:
                            print(f"B: [{entry['url ' + channel].strip()}]")
                            entry['idat'][channel].download(entry['url ' + channel].strip())
                else:
                    errors.append(f"idat missing: {entry['idat'][channel]}")

        if force_validating_sentrix_ids:
            with interruptible() as quit_event:
                for entry in tqdm(entries, desc="Validating sentrix IDs (q to skip)"):
                    if quit_event.is_set():
                        tqdm.write("Validation interrupted by user")
                        break

                    grn = entry['idat']['Grn']
                    red = entry['idat']['Red']

                    try:
                        grn_channel = grn.detected_channel
                        red_channel = red.detected_channel
                    except Exception as e:
                        errors.append(f"Channel check failed for sample id [{entry['Sample ID']}]: {e}")
                    else:
                        if grn_channel != 'Grn':
                            errors.append(f"Channel mismatch for sample id [{entry['Sample ID']}]: {grn.path} is not a Grn channel (detected {grn_channel})")
                        if red_channel != 'Red':
                            errors.append(f"Channel mismatch for sample id [{entry['Sample ID']}]: {red.path} is not a Red channel (detected {red_channel})")

                    if grn.sentrix_id != red.sentrix_id:
                        errors.append(f"idat files do not belong to each other !!: {grn.sentrix_id} & {red.sentrix_id}")

                    if entry['Sentrix ID'] != grn.sentrix_id:
                        errors.append(f"Sentrix id for sample id [{entry['Sample ID']}] database mismatch [{entry['Sentrix ID']}]: {grn.sentrix_id}")

                    if entry['Array type'] != grn.array_type:
                        errors.append(f"Array type for sample id [{entry['Sample ID']}] database mismatch [{entry['Array type']}]: {grn.array_type}")

        if len(errors) > 0:
            raise Exception("\n".join(["Checking idats resulted in errors:", ""] + errors))

   
    @beartype
    def verify_hdf5_in_sync_with_master_hfd5(self, random_samples: int = 100, random_cpgs: int = 15) -> dict:
        metadata = self.to_df()
        metadata = metadata.sample(n=min(random_samples, len(metadata)), replace=False)
        
        all_probes = self.get_all_probes_pymetharray
        mvalues = self.read_master_hdf5(self._master_hdf5_pymetharray, all_probes, metadata)

        num_matches = 0
        num_mismatches = 0

        k = 0
        for sample in tqdm(mvalues.columns, desc="Checking concordance individual & master hdf5"):
            metadata_sample = metadata[metadata['Filename'] == sample].iloc[0]
            sampled_cpgs = random.sample(list(all_probes), min(random_cpgs, len(all_probes)))
            
            df_master = mvalues.loc[sampled_cpgs, sample]
            df_single = (
                    pd.read_hdf(metadata_sample['hdf5-pymetharray'], 'methylation')['m_value']
                    .astype('float16')
                    .rename(sample)
                    .reindex(sampled_cpgs)
                )

            matches = (df_master == df_single) | (df_master.isna() & df_single.isna())
            num_matches = matches.sum().sum() 
            num_mismatches = (~matches).sum().sum()

            if k % 10 == 0:
                print({'num_matches': num_matches, 'num_mismatches': num_mismatches, 'array_type': metadata_sample['Array type']}) #v2's have sometimes aggregated cpgs

            if num_mismatches > 0 and metadata_sample['Array type'] != 'epicv2':
                raise Exception(
                f"Data mismatch in sample '{sample}': {num_matches} matches, {num_mismatches} mismatches"
            )
            
            k += 1

        out = {'num_matches': num_matches, 'num_mismatches': num_mismatches}
        #print(out)
        
        return out

    
    def check_empty_values(self):# needs to be done after downloading idats - as these are needed to calc them
        @beartype
        def error_on_none(df, column: str):
            is_none = df[column].isna()
            duplicate_rows = df[is_none]
        
            if not duplicate_rows.empty:
                display_df = duplicate_rows[["Filename", "Sentrix ID", "Dataset", "Array type"]]
        
                msg = (
                    f"The following '{column}' entries are empty:\n"
                    f"{display_df.to_string(index=False)}"
                )
                raise ValueError(msg)

        dfs = self.to_df()
        for column in ["Sample ID", "Filename", "Sentrix ID", "Array type"]:
            error_on_none(dfs, column)


    
    # one file per array type / dataset layout; if these are absent the hdf5 tree
    # is not mounted at all, which no amount of autocompletion can fix here
    _hdf5_mepylome_sentinels = [ # quick and dirty - should be the first entry from the database for each platform type
        "CPTAC-3/C3L-00104-01_A.hdf5",
        "GSE136361/GSM4047406_200394970005_R05C01.hdf5",
        "Verheul-Cell-Lines/203175830152_R04C01.hdf5",
    ]

    def check_hdf5_sentinels(self):
        errors = []

        for sentinel in self._hdf5_mepylome_sentinels:
            path = self._hdf5_mepylome_path + sentinel
            if not os.path.exists(path):
                errors.append(f"hdf5 (mepylome) sentinel missing: {path}")

        if len(errors) > 0:
            raise Exception("\n".join(["Checking hdf5 sentinels resulted in errors:", ""] + errors))

        print(f"Checked {len(self._hdf5_mepylome_sentinels)} hdf5 (mepylome) sentinels - all present", flush=True)


    def check_hdf5s(self, check_pymetharray: bool,
                    check_mepylome: Literal[False] | list[str], autocomplete: bool = False):
        errors = []
        print(f"autocomplete: {autocomplete}")
        
        for entry in tqdm(self, desc="Checking individual hdf5's"):
            # check for mepylome hdf5
            if check_mepylome and not os.path.exists(entry['hdf5-mepylome']):
                if autocomplete:
                    if os.path.exists(entry['hdf5-mepylome']+".bak"):
                        print(f"  Restoring hdf5-mepylome from backup: {entry['hdf5-mepylome']}.bak => {entry['hdf5-mepylome']}")
                        shutil.move(entry['hdf5-mepylome']+".bak", entry['hdf5-mepylome'])
                    else:
                        print(f"  Building hdf5-mepylome: {entry['hdf5-mepylome']}")
                        tmp = entry['hdf5-mepylome'] + '.tmp'
                        df, _, _ = idat_to_data_container_mepylome(
                            entry['idat']['Grn'].path,
                            entry['idat']['Red'].path,
                            keys=[str(_) for _ in self._master_hdf5_mepylome.keys()],
                        )
                        os.makedirs(os.path.dirname(entry['hdf5-mepylome']), exist_ok=True)
                        df.to_hdf(tmp, key='data', mode='w', complevel=1, complib='blosc')

                        with pd.HDFStore(tmp, mode='r') as store:
                            storer = store.get_storer('data')
                            assert tuple(storer.shape) == df.shape, f"Shape mismatch after write: {storer.shape} vs {df.shape}"

                        shutil.move(tmp, entry['hdf5-mepylome'])
                    
                else:
                    errors.append(f"hdf5 (mepylome) missing: {entry['hdf5-mepylome']}")


            # check for former pymetharray hdf5
            if check_pymetharray and not os.path.exists(entry['hdf5-pymetharray']):
                if autocomplete:
                    if os.path.exists(entry['hdf5-pymetharray']+".bak"):
                        print(f"  Restoring hdf5-pymetharray from backup: {entry['hdf5-pymetharray']}.bak => {entry['hdf5-pymetharray']}")
                        shutil.move(entry['hdf5-pymetharray']+".bak", entry['hdf5-pymetharray'])
                    else:
                        data_container, _, _ = idat_to_data_container_pymetharray(
                            entry['idat']['Grn'].path,
                            entry['idat']['Red'].path,
                            retain_uncorrected_probe_intensities=True,
                            pval=True,
                        )
                        data_container.export(entry['hdf5-pymetharray']+'.tmp')
        
                        succes = False
                        with pd.HDFStore(entry['hdf5-pymetharray']+'.tmp', mode='r') as store:
                            succes = True
                        if succes:
                            shutil.move(entry['hdf5-pymetharray']+'.tmp', entry['hdf5-pymetharray'])
                            
                        del data_container

                else:
                    errors.append(f"hdf5 missing: {entry['hdf5-pymetharray']}")

        if len(errors) > 0:
            raise Exception("\n".join(["Checking hdf5s resulted in errors:", ""] + errors))


    @beartype
    def check_cnvs(self, autocomplete: bool = False, threads: int = 24):
        # The IDAT -> CNV pipeline (hg38 manifests + shared bin grid) lives in
        # libcognition.mepylome_helpers, shared with sandbox/test_cnv_single.py.
        from .mepylome_helpers import mepylome_idat_to_cnv_database_export, mepylome_output_paths, mepylome_shared_bins

        reference_dir = Path(self._idat_normals_path)

        # (1) Scan all rows to determine which CNV files are missing
        todo = []
        for entry in tqdm(self, desc="Checking individual CNV files"):
            if pd.isna(entry['Filename']):
                raise ValueError("should not happen")

            dataset = entry['Dataset']
            # hg38 output filenames (.cnv_bins.hg38.csv / .cnv_detail.hg38.csv /
            # .normals.hg38.txt), as produced by mepylome_helpers.
            _, cnv_bins_file, cnv_detail_file, cnv_normals_file = mepylome_output_paths(
                dataset, entry['Filename']
            )

            if cnv_bins_file.exists() and cnv_detail_file.exists() and cnv_normals_file.exists():
                continue
            
            if True: #entry['Tumor Type'] in [tumor_db.get_by_name("Oligodendroglioma"), tumor_db.get_by_name("Astrocytoma")]:
                todo.append((entry, cnv_bins_file))

        # (2) Either generate the missing files (autocomplete) or report them as errors
        if not autocomplete:
            if len(todo) > 0:
                errors = [f"CNV missing: {cnv_bins_file}" for _, cnv_bins_file in todo]
                # skip for now -- raise Exception("\n".join(["Checking CNVs resulted in errors:", ""] + errors))
            return

        if not todo:
            return

        print(f"Going to generate {len(todo)} CNV file(s)")

        # Prerequisites that affect every sample: fail immediately (and clearly)
        # rather than crash mid-loop on each sample. The reference pool must
        # exist, and mepylome_shared_bins() validates+loads the common 450k bin grid.
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference (normals) dir not found: {reference_dir}")
        mepylome_shared_bins()

        # Skippable with 'q': whatever is not generated now simply stays on the
        # todo list of the next run, so stopping here loses no work.
        with interruptible() as quit_event:
            for entry, cnv_bins_file in tqdm(todo, desc="Autocompleting missing CNV's (q to skip)"):
                if quit_event.is_set():
                    tqdm.write("Autocompleting CNV's interrupted by user")
                    break

                array_type = entry['Array type']
                dataset = entry['Dataset']
                filename = entry['Filename']

                # IDAT basename (same dir as the Grn idat, sample-named).
                basename = Path(entry['idat']['Grn'].path).parent / filename

                # hg38 build + shared (pre-defined) 450k bins; per-array-type
                # reference under <reference_dir>/<array_type>/. verbose=False keeps
                # the tqdm progress bar clean. Any error aborts immediately — these
                # are serious and not worth continuing past.
                mepylome_idat_to_cnv_database_export(basename, dataset, filename, array_type, reference_dir,
                            verbose=False)


    @beartype
    def validate_normal_idats(self, autocomplete: bool = False):
        """Reconcile the CNV reference symlink pool with the current db.

        The CNV reference (normals) live at <_idat_normals_path>/<array_type>/
        as symlinks into the real per-sample idats. The reference set is exactly
        the samples flagged 'Include as CNV reference' in the db (a curated
        whitelist, only allowed on normals); this keeps that directory in sync:

          * a symlink there that is NOT backed by a flagged sample (unflagged,
            renamed, removed from the db) is deleted;
          * a missing symlink for a flagged sample is (re)created.

        Only symlinks are ever removed: a real (non-symlink) file in the pool is
        left in place and reported instead, so nothing irreplaceable is deleted
        because of a stale db row.

        A link is considered correct when it is a symlink resolving to the same
        real idat as the db entry, regardless of whether it is stored as a
        relative or absolute link (existing links of either style are left
        untouched); newly created links are relative for portability.

        Without autocomplete the discrepancies are raised as errors (like the
        other check_* methods); with autocomplete they are fixed in place.

        The db 'Include as CNV reference' column is the single source of truth
        for the reference set; this materializes it into the symlink pool, which
        mepylome_helpers then consumes wholesale (no separate whitelist).
        """
        normals_base = Path(self._idat_normals_path)

        # (1) Desired symlinks: {link_path: real_idat_path} for every sample
        #     flagged 'Include as CNV reference', both channels. That column is
        #     the curated reference whitelist (the db parser only allows it on
        #     normals), so it -- not the tumor type -- decides the pool. The link
        #     basename is taken from the real idat name so it matches the db's
        #     own path construction (gz already stripped there).
        desired: dict[Path, Path] = {}
        for entry in self:
            if entry['Include as CNV reference'] is not True:
                continue
            if not (pd.isna(entry['Discarded']) or str(entry['Discarded']).strip() in ("", "-")):
                continue  # a discarded sample never enters the reference pool
            if pd.isna(entry['Array type']):
                continue  # can't place it without knowing the platform folder

            array_type = entry['Array type']
            for channel in ('Grn', 'Red'):
                real = Path(entry['idat'][channel].path)
                desired[normals_base / array_type / real.name] = real

        # (2) Everything currently in the pool (any array-type subdir).
        existing: set[Path] = set()
        if normals_base.exists():
            for sub in normals_base.iterdir():
                if sub.is_dir():
                    existing.update(p for p in sub.iterdir() if not p.is_dir())

        errors = []

        # (3) Stray entries: present but not desired. Symlinks are deleted; a
        #     real file is never touched, only reported.
        strays = sorted(existing - set(desired))
        for path in tqdm(strays, desc="Pruning stray normal symlinks"):
            if path.is_symlink():
                if autocomplete:
                    path.unlink()
                    print(f"Removed stray normal symlink: {path}")
                else:
                    errors.append(f"Stray normal symlink (not a current non-tumor sample): {path}")
            else:
                errors.append(f"Unexpected non-symlink in normals pool (left in place): {path}")

        # (4) Desired links that are missing or point elsewhere: (re)create as a
        #     relative symlink.
        for link, real in tqdm(sorted(desired.items()), desc="Validating normal idats"):
            target_real = real.resolve()
            if link.is_symlink() and link.resolve() == target_real:
                continue  # already correct (relative or absolute)
            if autocomplete:
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    link.unlink()
                rel = os.path.relpath(target_real, start=link.parent.resolve())
                link.symlink_to(rel)
                print(f"Linked normal: {link} -> {rel}")
            else:
                errors.append(f"Missing/incorrect normal symlink: {link} -> {real}")

        if len(errors) > 0:
            raise Exception("\n".join(["Validating normal idats resulted in errors:", ""] + errors))


    # @deprecated("phase pymetharray out")
    @beartype
    def check_master_hdf5_pymetharray(self, autocomplete: bool = False):
        self.check_master_hdf5(
            fn=self._master_hdf5_pymetharray,
            reader=self.read_individual_hdf5s_threaded,
            autocomplete=autocomplete
        )

    @beartype
    def check_master_hdf5_mepylome(self, key, autocomplete: bool = False):
        fn = self._master_hdf5_mepylome[key]
        self.check_master_hdf5(
            fn=fn,
            reader=lambda m, use_tqdm=True: self.read_individual_hdf5s_threaded_mepylome(m, key=key, use_tqdm=use_tqdm),
            autocomplete=autocomplete
        )

    @beartype
    def check_cnv_master(self, autocomplete: bool = False):
        """Sync the master CNV matrix (bins x samples, per-bin Median) with the db.

        The CNV analogue of check_master_hdf5_mepylome, but written as a single
        parquet file (_master_cnv): the shared bin grid is only a few thousand
        rows and CNV is generated for a subset of samples, so the matrix is small
        and a parquet is more efficient/portable than the chunked HDF5 master
        (and far cheaper to write than a gzip CSV: no per-cell text formatting,
        no single-threaded gzip).

        Only samples whose per-sample CNV bins file exists are included (CNV is
        not generated for every entry). At this size an incremental append/drop
        buys little, so when out of sync (and autocomplete is set) the master is
        fully rebuilt; without autocomplete the discrepancies are raised as
        errors, matching the other master checks.
        """
        from .mepylome_helpers import mepylome_output_paths

        fn = self._master_cnv

        metadata_all = self.to_df()
        # Samples that actually have CNV output on disk (bins file present).
        has_cnv = metadata_all.apply(
            lambda r: (not pd.isna(r['Filename']))
                      and mepylome_output_paths(r['Dataset'], r['Filename'])[1].exists(),
            axis=1,
        )
        metadata_cnv = metadata_all[has_cnv]
        wanted = set(metadata_cnv['Filename'])

        if os.path.exists(fn):
            # Cheap staleness check: read only the parquet footer/schema to get
            # the sample columns (no row data is touched). 'bin' is the stored
            # index column, so drop it.
            import pyarrow.parquet as pq
            present = [c for c in pq.read_schema(fn).names if c != 'bin']
            to_drop = [c for c in present if c not in wanted]
            to_append = [c for c in wanted if c not in set(present)]

            if not to_drop and not to_append:
                print(f"Master CNV in sync: {fn}")
                return

            if not autocomplete:
                errors = [f"master CNV: column '{c}' not present in database and should be dropped" for c in to_drop]
                errors += [f"master CNV: database entry '{c}' not present in master CNV and should be inserted" for c in to_append]
                raise Exception("\n".join(["", "The master CNV is not in sync with the database", ""] + errors))

            print(f"Going to rebuild master CNV  --  to_drop: {len(to_drop)}  --  to_append: {len(to_append)}")
            self.rebuild_cnv_master(fn, metadata_cnv)
        else:
            if autocomplete:
                self.rebuild_cnv_master(fn, metadata_cnv)
            else:
                raise Exception(f"No master CNV present at all: {fn}")

    def rebuild_cnv_master(self, fn, metadata_cnv):
        """(Re)build the master CNV parquet from the per-sample CNV bins files.

        Phase-level debug prints (read -> in RAM -> parquet write -> move) with
        elapsed times: to_parquet is a single atomic call with no progress hook,
        so a real per-cent progress bar during the write is not available; the
        per-sample tqdm in read_individual_cnvs_threaded covers the read phase.
        """
        # Phase 1: read every per-sample CNV into the bins x samples matrix.
        t0 = time.perf_counter()
        print(f"[CNV master] reading {len(metadata_cnv)} per-sample CNV file(s) ...")
        values = self.read_individual_cnvs_threaded(metadata_cnv)
        mem_mb = values.memory_usage(deep=True).sum() / 1e6
        print(f"[CNV master] DataFrame in RAM: shape={values.shape}  "
              f"~{mem_mb:.0f} MB  ({time.perf_counter() - t0:.1f}s)")

        if os.path.dirname(fn):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
        # Phase 2: write parquet (zstd). Atomic move into place so a crash
        # mid-write never leaves a partial master.
        tmp = f"{fn}.{random.randint(0, 9999):04}.tmp"
        t1 = time.perf_counter()
        print(f"[CNV master] writing parquet (zstd) -> {tmp} ...")
        values.to_parquet(tmp, compression='zstd')
        print(f"[CNV master] parquet written: {os.path.getsize(tmp) / 1e6:.0f} MB  "
              f"({time.perf_counter() - t1:.1f}s)")
        shutil.move(tmp, fn)
        print(f"[CNV master] done: {fn}  {values.shape}  "
              f"(total {time.perf_counter() - t0:.1f}s)")

    @beartype
    def check_master_hdf5(self, fn: str, reader, autocomplete: bool = False):
        # @deprecated: werkend maar inefficiënt, gebruik merge2
        def merge1(df_main, df_append):
            df_main.index.name = None
            df_append.index.name = None

            df_main = pd.concat([df_main, df_append], axis=1, copy=False)

            del(df_append)
            df_main.index.name = 'Filename'

            return df_main

        def merge2(df_main, df_append): # fastest, with higher peak memory usage
            df_main.index.name = None
            df_append.index.name = None

            index = df_main.index
            cols_main = df_main.columns
            cols_append = df_append.columns

            df_main = df_main.to_numpy()
            df_append = df_append.to_numpy()

            # np.hstack() is ideal for horizontal stacking (columns)
            df_main = np.hstack((df_main, df_append))
            del(df_append)

            df_main = pd.DataFrame(
                df_main,
                index=index,
                columns=list(cols_main) + list(cols_append)
            )

            df_main.index.name = 'Filename'
            return df_main

        errors = []

        metadata_all = self.to_df()

        if os.path.exists(fn):
            with pd.HDFStore(fn, mode='r') as store:
                columns = store['mvalue_table_all_columns']['Filename'].tolist()

            to_drop = [col for col in tqdm(columns, desc="Checking columns") if col not in metadata_all['Filename'].values]
            to_append = metadata_all[~metadata_all['Filename'].isin(columns)]

            if len(to_append) > 0 or len(to_drop) > 0:

                if autocomplete:
                    print(f"Going to update master hdf5  --  to_drop: {len(to_drop)}  --  to_append:  --  {len(to_append)}")
                    all_probes = self.get_all_probes_mepylome
                    mvalues_all = self.read_master_hdf5(fn, all_probes)

                if len(to_drop) > 0:
                    print(f"Going to to_drop: {len(to_drop)}")
                    if autocomplete:
                        mvalues_all = mvalues_all.drop(columns=to_drop)
                        self.write_master_hdf5(mvalues_all, fn)
                    else:
                        for elem in to_drop:
                            errors.append(f"master hdf5: found column '{elem}' not present in database and should be dropped")

                if len(to_append) > 0:
                    if autocomplete:
                        print(f"Going to append:  --  {len(to_append)}")

                        slice_size = 600
                        with tqdm(total=len(to_append), desc="Exporting chunks") as pbar:
                            for i in range(0, len(to_append), slice_size):
                                to_append_slice = to_append.iloc[i:i + slice_size]

                                #Y = self.read_individual_hdf5s(to_append_slice, use_tqdm=True)
                                Y = reader(to_append_slice, use_tqdm=True)


                                #probes_pymetharray = self.get_all_probes_pymetharray
                                #print(f"probes_pymetharray ({len(probes_pymetharray)}): {probes_pymetharray[:10]}")
                                probes_mepylome = self.get_all_probes_mepylome
                                print(f"probes_mepylome    ({len(probes_mepylome)}): {probes_mepylome[:10]}")
                                print(f"M {mvalues_all.shape}, first 10 rows: {mvalues_all.index[:10].tolist()}")
                                print(f"Y {Y.shape}, first 10 rows: {Y.index[:10].tolist()}")

                                assert mvalues_all.index.equals(Y.index), "pre merging"
                                mvalues_all = merge2(mvalues_all, Y)

                                print("*M", mvalues_all.shape)
                                print(mvalues_all.head(n=2))

                                assert mvalues_all.index.equals(Y.index), "post merging"
                                mvalues_all.index.name = 'Filename'

                                del(Y)

                                self.write_master_hdf5(mvalues_all, fn)
                                pbar.update(len(to_append_slice))
                    else:
                        for elem in to_append['Filename']:
                            errors.append(f"master hdf5: found database entry '{elem}' not present in master hdf5 and should be inserted")

            if len(errors) > 0:
                raise Exception("\n".join(["", "There master hdf5 is not in sync with the database", ""] + errors))

        else:
            if autocomplete:
                self.full_rebuild_master_hdf5(fn, reader)
            else:
                raise Exception(f"No master hdf5 present at all: {fn}")

    @beartype
    def read_individual_hdf5s_threaded(self, metadata_selection: pd.DataFrame, threads = 8, use_tqdm:bool = True) -> pd.DataFrame:
        def load_threadsafe(i, sample):
            with file_lock:
                return i, pd.read_hdf(sample['hdf5-pymetharray'], 'methylation')

        def process_sample(idx_sample):
            i, sample_t = idx_sample
            j, sample = sample_t

            k, raw = load_threadsafe(i, sample) #pd.read_hdf(sample.hdf5, 'methylation')
            if sample['Array type'] == "epicv2":
                raw = epicv2_to_epic(raw)
            col = (
                raw[['m_value']]
                .reindex(all_probes, copy=False)
                .to_numpy(copy=False, dtype='float16')
                .ravel()
            )
            
            del raw
            return i, col

        all_probes = self.get_all_probes_pymetharray
        #metadata_selection = metadata_selection.head(n=125)
        
        data = np.empty((len(all_probes), len(metadata_selection)), dtype='float16')
        with ThreadPoolExecutor(max_workers=threads) as executor: # goes a bit faster, ~2 times,
            futures = {executor.submit(process_sample, (i, sample)): i
                       for i, sample in enumerate(metadata_selection.iterrows())}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Importing mvalues"):
                i, col = future.result()
                data[:, i] = col

        print(f"1. {data.shape}")
        data = pd.DataFrame(data)
        print(f"2. {data.shape}")
        data.index = all_probes
        data.index.name = 'Filename'
        print(f"3. {data.shape}")
        
        data.columns = metadata_selection['Filename']
        print(f"4. {data.shape}")

        return data



    def read_individual_hdf5s_threaded_mepylome(self, metadata_selection: pd.DataFrame,
                                                key: str , # e.g. 'mvalues_illumina'
                                                threads: int = 8,
                                                use_tqdm: bool = True) -> pd.DataFrame:
        file_lock = threading.Lock()

        def load_threadsafe(i, sample, key):
            with file_lock:
                #print(f"Loading {sample['hdf5-mepylome']} for key '{key}'")
                data = pd.read_hdf(sample['hdf5-mepylome'])
                #print(f" => {data.shape}")
                col = next(c for c in data.columns if c.endswith(f'_{key}'))
                data = data[[col]].rename(columns={col: col[:-(len(key) + 1)]})
                return i, data

        def process_sample(idx_sample, key):
            i, sample_t = idx_sample
            j, sample = sample_t

            k, raw = load_threadsafe(i, sample, key)
            if sample['Array type'] == "epicv2":
                raw = epicv2_to_epic(raw)
            col = (
                raw.iloc[:, [0]]
                .reindex(all_probes, copy=False)
                .to_numpy(copy=False, dtype='float16')
                .ravel()
            )
            del raw
            return i, col

        all_probes = self.get_all_probes_mepylome

        data = np.empty((len(all_probes), len(metadata_selection)), dtype='float16')
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(process_sample, (i, sample), key): i
                    for i, sample in enumerate(metadata_selection.iterrows())}

            iterator = as_completed(futures)
            if use_tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Importing mvalues")
            for future in iterator:
                i, col = future.result()
                data[:, i] = col

        data = pd.DataFrame(data)
        data.index = all_probes
        data.index.name = 'Filename'
        data.columns = metadata_selection['Filename']
        return data


    @beartype
    def read_individual_cnvs_threaded(self, metadata_selection: pd.DataFrame,
                                      threads: int = 8,
                                      use_tqdm: bool = True) -> pd.DataFrame:
        """Read per-sample CNV bins into a bins x samples matrix (per-bin Median).

        The CNV analogue of read_individual_hdf5s_threaded_mepylome: instead of
        per-sample HDF5 it reads the CSV output (*.cnv_bins.hg38.csv) written by
        mepylome_idat_to_cnv_database_export. Each sample contributes its per-bin 'Median' (log2
        ratio), reindexed onto the shared bin grid (get_all_cnv_bins) so all
        samples align row-for-row. No file lock is needed (unlike the HDF5
        reader): pandas reads distinct CSV files thread-safely.
        """
        from .mepylome_helpers import mepylome_output_paths

        all_bins = self.get_all_cnv_bins

        def process_sample(idx_sample):
            i, sample_t = idx_sample
            _, sample = sample_t
            _, bins_file, _, _ = mepylome_output_paths(sample['Dataset'], sample['Filename'])
            df = pd.read_csv(bins_file)
            # Single bin key 'chr:start-end', matching get_all_cnv_bins.
            df.index = [f"{c}:{s}-{e}" for c, s, e in
                        zip(df['Chromosome'], df['Start'], df['End'])]
            col = (
                df[['Median']]
                .reindex(all_bins, copy=False)
                .to_numpy(copy=False, dtype='float32')
                .ravel()
            )
            del df
            return i, col

        data = np.empty((len(all_bins), len(metadata_selection)), dtype='float32')
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(process_sample, (i, sample)): i
                       for i, sample in enumerate(metadata_selection.iterrows())}

            iterator = as_completed(futures)
            if use_tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Importing CNV bins")
            for future in iterator:
                i, col = future.result()
                data[:, i] = col

        data = pd.DataFrame(data)
        data.index = all_bins
        data.index.name = 'bin'
        data.columns = metadata_selection['Filename']
        return data


    @beartype
    def read_individual_hdf5s(self, metadata_selection: pd.DataFrame, use_tqdm:bool = True) -> pd.DataFrame:
        all_probes = self.get_all_probes_pymetharray
        
        data = np.empty((len(all_probes), len(metadata_selection)), dtype='float16')
        iterator = tqdm(metadata_selection.iterrows(), total=len(metadata_selection), desc="Importing mvalues") if use_tqdm else metadata_selection.iterrows()

        for i, (_, sample) in enumerate(iterator):
            raw = pd.read_hdf(sample['hdf5-pymetharray'], 'methylation')
            if sample['Array type'] == "epicv2":
                raw = epicv2_to_epic(raw)
            data[:, i] = (
                raw[['m_value']]
                .reindex(all_probes, copy=False)
                .to_numpy(copy=False, dtype='float16')
                .ravel()
            )

            del(raw)


        # data = np.reshape(data, shape=(len(data), len(all_probes))) - mem inefficient
        
        data = pd.DataFrame(data)
        data.index = all_probes
        data.index.name = 'Filename'
        
        data.columns = metadata_selection['Filename']

        return data


    @beartype
    def read_master_hdf5(self, fn: str, 
                         all_probes: list[str],
                         metadata_selection: pd.DataFrame | None = None) -> pd.DataFrame:
        print(f"Reading: {fn}")
        with pd.HDFStore(fn, mode='r') as store:
            print("k",len(all_probes))
        
            

            with tqdm(total = 5, desc="Reading master hdf5") as pbar:
                mvalues = store['mvalue_table_all']
                pbar.update(1)
                
                mvalues.index = store['mvalue_table_all_index']['Filename'].tolist()
                pbar.update(1)
                
                cols = store['mvalue_table_all_columns']['Filename'].tolist()
                mvalues.columns = cols
                pbar.update(1)
                
                if len(all_probes) != len(mvalues) or not np.array_equal(mvalues.index.values, all_probes):
                    mvalues = mvalues.reindex(all_probes, copy=False) # horibly slow and expensive, avoid at all cost
                pbar.update(1)

                if metadata_selection is not None:
                    selection = metadata_selection['Filename'].dropna().tolist()
                    
                    mvalues = mvalues[selection]
                pbar.update(1)

                mvalues.index.name = 'Filename'
                return mvalues
        

    @beartype
    def write_master_hdf5(self, mvalues: pd.DataFrame, fn):
        fn_tmp = f"{fn}.{random.randint(0, 9999):04}.tmp"

        with pd.HDFStore(fn_tmp, mode='w') as store:
            index_out = mvalues.index.to_frame(index=False, name='Filename')
            columns_out = mvalues.columns.to_frame(index=False, name='Filename')

            idx_old = mvalues.index # hdf5 can't cope with indexes and colnames consisting of strings, once exported chunkwise in frames of this size. we have to remove it, but deep copies of the whole dataframe are too expensive, so reset them here and restore later
            columns_old = mvalues.columns
            
            mvalues.index = pd.RangeIndex(len(mvalues)) # dropping is more expensive, this does not trigger an export error
            mvalues.columns = [None] * mvalues.shape[1]

            store.put('/mvalue_table_all_index', index_out, format='table')
            store.put('/mvalue_table_all_columns', columns_out, format='table')

            # Initialize the table
            chunk_size = 10000
            store.put('/mvalue_table_all', mvalues.iloc[0:0], format='table')
        
            # Write in chunks
            print(f"exporting to {fn_tmp}")
            for start in tqdm(range(0, len(mvalues), chunk_size), desc = f"Exporting mvalues:"):
                end = start + chunk_size
                chunk = mvalues.iloc[start:end]
                store.append('/mvalue_table_all', chunk)
                del(chunk)

            # whacky hacky
            mvalues.index = idx_old
            mvalues.columns = columns_old


        success = False
        ncol = False
        nrow = False
        with pd.HDFStore(fn_tmp, mode='r') as store:
            success = True
            storer = store.get_storer('mvalue_table_all')
            if storer.nrows == mvalues.shape[0]:
                nrow = True
            
            if storer.ncols == mvalues.shape[1]:
                ncol = True
            

        if success and ncol and nrow:
            os.chmod(fn_tmp, 0o664)
            shutil.move(fn_tmp, fn)
        else:
            if not success:
                raise Exception(f"Corrupt file: {fn_tmp}")
            if not ncol:
                raise Exception(f"Incorrect dimentsions - ncol")
            if not nrow:
                raise Exception(f"Incorrect dimentsions - nrow")
            
    
    def full_rebuild_master_hdf5(self, fn, reader):
        metadata_all = self.to_df()
        values = reader(metadata_all)
        
        print(f"X. {values.shape}")
        self.write_master_hdf5(values, fn)
        

    @beartype
    # autocomplete: bool = False,
    def validate_all_idat_files_are_in_db(self, autocomplete: bool = False) -> bool:
        # Reference
        df_reference = self.to_df()
        df_reference['UID'] = df_reference['Dataset'] + '/' + df_reference['Filename']

        df_scan = pd.DataFrame({'idat': glob.glob(self._idat_path + '**/*.idat', recursive=True)})
        if len(df_scan) > 0:
            df_scan['Filename'] = df_scan['idat'].apply(lambda p: re.sub(r'_(Grn|Red)\.idat$', '', os.path.basename(p)))
            df_scan['Dataset'] = df_scan['idat'].apply(lambda p: os.path.basename(os.path.dirname(p)))
            df_scan['UID'] = df_scan['Dataset'] + '/' + df_scan['Filename']

            # Check for presence in reference
            df_scan['not_missing'] = df_scan['UID'].progress_apply(lambda uid: uid in df_reference['UID'].tolist())

            # Get sorted list of missing file paths
            missing_files = sorted(df_scan[~df_scan['not_missing']]['idat'].tolist())
        
            if len(missing_files) > 0:
                if autocomplete:
                    print("There are *.idat files that are missing in data/database.txt - these are renamed by :")

                    for f in missing_files:
                        print(f"mv '{f}' => '{f}.bak'")
                        shutil.move(f, f + ".bak")
                else:
                    raise Exception("\n".join(["", "There are *.idat files that are missing in data/database.txt", ""] + ['mv ' + _ + " " + _ + ".bak" for _ in missing_files]))
        return True

    
    @beartype
    def validate_all_hdf5_files_are_in_db(self,
                                          validate_pymetharray: bool = True,
                                          validate_mepylome: bool | list[str] = True,
                                          autocomplete: bool = False) -> bool:
        # Reference
        df_reference = self.to_df()
        df_reference['UID'] = df_reference['Dataset'] + '/' + df_reference['Filename']

        checks = []
        if validate_pymetharray:
            checks.append(self._hdf5_path)
        if validate_mepylome:
            checks.append(self._hdf5_mepylome_path)

        for path in checks:
            df_scan = pd.DataFrame({'HDF5': glob.glob(path + '**/*.hdf5', recursive=True)})
            if len(df_scan) > 0:
                df_scan['Filename'] = df_scan['HDF5'].apply(lambda p: re.sub(r'\.hdf5$', '', os.path.basename(p)))
                df_scan['Dataset'] = df_scan['HDF5'].apply(lambda p: os.path.basename(os.path.dirname(p)))
                df_scan['UID'] = df_scan['Dataset'] + '/' + df_scan['Filename']

                # Check for presence in reference
                df_scan['not_missing'] = df_scan['UID'].progress_apply(lambda uid: uid in df_reference['UID'].tolist())

                # Get sorted list of missing file paths
                missing_files = sorted(df_scan[~df_scan['not_missing']]['HDF5'].tolist())

                if len(missing_files) > 0:
                    if autocomplete:
                        print("There are *.hdf5 files that are missing in data/database.txt - these are renamed by :")

                        for s in missing_files:
                            print(f"mv '{s}' => '{s}.bak'")
                            shutil.move(s, s + ".bak")
                    else:
                        raise Exception("\n".join(["", "There are *.hdf5 files that are missing in data/database.txt", ""] + ['rm ' + _ for _ in missing_files]))

        return True
    

    @beartype
    def add_sample(self, row: pd.core.series.Series):
        def fix_survival_days(val) -> int:
            try:
                val = int(val)
            except (ValueError, TypeError):
                val = None

            if val is not None and val < 1:
                print(row)
                raise Exception(f"survival time does not make sense: {val}")

            return val

        def fix_survival_event(val) -> bool:
            if val in ['1',1,True,'True','true','t']:
                return True
            elif val in ['0',0,False,'False','false','f']:
                return False
            elif val in ['', '-', None, np.nan]:
                return None
            else:
                print(row)
                raise Exception("strange survival event value [1/0]: "+str(val))

        def fix_md5sum(val) -> bool:
            if val in ['', '-', None, np.nan]:
                return None
            elif re.match(r'^[a-fA-F0-9]{32}$', val):
                return val.upper()
            else:
                print(row)
                raise Exception(f"Invalid md5sum: {val}")
        
        def fix_dataset(val) -> str:
            if re.search(r'[_,/]', val):
                print(row)
                raise Exception(f"Invalid dataset name: {val} (must not contain '_', ',' or '/')")

            return val
        
        def fix_sex(val) -> str:
            if val in ['F','f','female','Female']:
                return "Female"
            elif val in ["M","m", "male", "Male"]:
                return "Male"
            elif val in ['', '-', None, np.nan] or pd.isna(val):
                return None
            else:
                print(row)
                raise Exception(f"Unknown sex: {val}")
        
        def fix_age(val) -> str:
            if val in ['', '-', None, np.nan] or pd.isna(val):
                return None
        
            try:
                age = float(val)
            except (ValueError, TypeError):
                raise Exception(f"Unknown age: {val}")
            if not (0.0 <= age <= 130.0):
                raise Exception(f"Age out of range (0-130): {age}")

            return age

        def fix_age_class(val, age) -> str:
            if val in ['', '-', None, np.nan] or pd.isna(val):
                return None
            elif val in ["Pediatric", "Adult"]:
                if age is not None:
                    if (val == "Pediatric" and age < 22) or (val == "Adult" and age >= 22):
                        return val
                    else:
                        print(row)
                        raise Exception(f"Age class '{val}' does not align with age: {age}")
                return val
            else:
                raise Exception(f"Unknown age: {val}")
        
        def fix_tumor_type(val):
            if val in ['', '-', None, np.nan]:
                return None
            else:
                tumor_type = tumor_db.get_by_name(val)
                
                if tumor_type:
                    return tumor_type
                else:
                    print(row)
                    raise Exception(f"Unknown tumor type: '{val}'")
            
        def fix_tumor_location(val):
            if val in ['', '-', None, np.nan] or pd.isna(val):
                return None
            else:
                try:
                    return TumorLocation.from_string(val)
                except ValueError:
                    print(row)
                    raise Exception(f"Unknown tumor location: '{val}'")

        def fix_idh_status(val_idh_status, val_tumor_type = None):
            try:
                column_idh_status = None if (val_idh_status in ['', '-', None, np.nan] or pd.isna(val_idh_status)) else IDHStatus.from_string(val_idh_status)
            except Exception:
                print(row)
                raise Exception(f"Tumor type: {val_tumor_type} -- does not align with IDH status: {val_idh_status}")
            tumor_type_idh_status  = None if val_tumor_type is None else val_tumor_type.idh_status

            if column_idh_status is None:
                return column_idh_status # None
            elif column_idh_status is not None and tumor_type_idh_status is None:
                return column_idh_status
            else:
                if column_idh_status == tumor_type_idh_status:
                    return column_idh_status
                else:
                    print(row)
                    raise Exception(f"Tumor type: {val_tumor_type} -- does not align with IDH status: {column_idh_status}")

        def fix_primary_or_recurrent(val):
            if val in ['', '-', None, np.nan] or pd.isna(val):
                return None
            elif val in ['Primary', 'Recurrent']:
                return val
            else:
                print(row)
                raise Exception(f"Unknown 'Primary or Recurrent' value [Primary/Recurrent]: {val}")
            
        def fix_grade(val, sentrix_id = None):
            if val in [None, np.nan, '', '-']:
                return None
            elif val in ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4',
                         'Grade I', 'Grade II', 'Grade III', 'Grade IV']:
                return val
            else:
                print(row)
                raise Exception(f"Incorrect grade: {val} (sentrix_id: {sentrix_id})")

        def fix_include_as_cnv_reference(val, tumor_type):
            # Whitelist flag for the CNV reference pool: empty/'-' means "not a
            # reference" (the common case), so a new sample never becomes a
            # reference until explicitly tagged. Only 'Yes' is restricted -- it
            # is only meaningful on a normal (Non tumor brain/blood) sample, so
            # a Yes elsewhere is a curation mistake and fails hard.
            if val in ['', '-', None, np.nan] or pd.isna(val):
                return None
            elif val in ['1', 1, True, 'True', 'true', 't', 'Yes', 'yes', 'y', 'Y']:
                normal_types = set([TumorType.NON_TUMOR_BRAIN, TumorType.NON_TUMOR_BLOOD])
                if tumor_type not in normal_types:
                    print(row)
                    raise Exception(f"'Include as CNV reference' is set on a non-normal sample (Tumor Type: {tumor_type})")
                return True
            elif val in ['0', 0, False, 'False', 'false', 'f', 'No', 'no', 'n', 'N']:
                return False
            else:
                print(row)
                raise Exception(f"strange 'Include as CNV reference' value [Yes/No]: {val}")

        def fix_fraction_failed(val, column):
            # Curated as a percentage string ("1.23%"); strip the trailing '%'
            # and keep the number on that same percentage scale.
            if val in ['', '-', None, np.nan] or pd.isna(val):
                return None

            val = str(val).strip()
            if val.endswith('%'):
                val = val[:-1].strip()

            try:
                fraction = float(val)
            except (ValueError, TypeError):
                print(row)
                raise Exception(f"Invalid '{column}' value: {val}")

            if not (0.0 <= fraction <= 100.0):
                print(row)
                raise Exception(f"'{column}' out of range (0-100%): {fraction}")

            return fraction

        def fix_fraction_detP_failed(val):
            return fix_fraction_failed(val, 'Fraction det-P')

        def fix_fraction_poobah_failed(val):
            return fix_fraction_failed(val, 'Fraction poobah')

        def fix_array_type(val):
            if val in ['', '-', None, np.nan] or (isinstance(val, float) and np.isnan(val)):
                return None
            val = str(val).strip()
            if val not in ('450k', 'epic', 'epicv2'):
                print(row)
                raise Exception(f"unknown array type: {val}")
            return val

        row = row.copy()  # voorkom dat het originele DataFrame wordt gemuteerd
        row['Dataset'] = fix_dataset(row['Dataset'])
        
        try:
            row['WHO Grade'] = fix_grade(row['WHO Grade'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['Survival days'] = fix_survival_days(row['Survival days'])
            row['Survival event'] = fix_survival_event(row['Survival event'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")
        
        if (row['Survival days'] is None) != (row['Survival event'] is None):
            print(row)
            raise Exception("unclear survival data")
        
        try:
            row['Sex'] = fix_sex(row['Sex'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")
        
        try:
            row['Age'] = fix_age(row['Age'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['Age Class'] = fix_age_class(row['Age Class'], row['Age'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['Tumor Type'] = fix_tumor_type(row['Tumor Type'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            # row.get(): tolerate the column not existing yet, so the db still
            # parses before the 'Include as CNV reference' column is added.
            row['Include as CNV reference'] = fix_include_as_cnv_reference(
                row.get('Include as CNV reference', None), row['Tumor Type']
            )
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['Tumor Location'] = fix_tumor_location(row['Tumor Location'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['IDH Status'] = fix_idh_status(row['IDH Status'], row['Tumor Type'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['Primary or Recurrent'] = fix_primary_or_recurrent(row['Primary or Recurrent'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            row['Array type'] = fix_array_type(row['Array type'])
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        try:
            # row.get(): tolerate the column not existing yet, so the db still
            # parses before the 'Fraction poobah' column is added.
            row['Fraction det-P'] = fix_fraction_detP_failed(row.get('Fraction det-P', None))
            row['Fraction poobah'] = fix_fraction_poobah_failed(row.get('Fraction poobah', None))
        except Exception as e:
            raise Exception(f"Error in sample {row['Sentrix ID']}: {e}")

        
        row['md5sum Grn'] = fix_md5sum(row['md5sum Grn'])
        row['md5sum Red'] = fix_md5sum(row['md5sum Red'])
        
        row['MNP v12.8'] = self.mnp_db_v12_8.get_by_sentrix_id(row['Sentrix ID'])
        row['Methylscape Bv2'] = self.methylscape_db_Bv2.get_by_sentrix_id(None if row['Sentrix ID'] is np.nan else row['Sentrix ID'])
        
        self._samples.append(row)

    @beartype
    def get_mnp_v12_8_by_sentrix_id(self, sentrix_id: str, top_n: int = 1) -> None | pd.DataFrame:
        if sentrix_id in self.mnp_db_v12_8.mnp_table_v12_8.columns:
            filtered = self.mnp_db_v12_8.mnp_table_v12_8[sentrix_id].sort_values(ascending=False)
            if top_n == 0:
                return filtered.to_frame()
            else:
                return filtered.head(top_n).to_frame()

        else:
            return None

    def __iter__(self):
        for sample in self._samples:
            yield sample

    def __len__(self):
        return len(self._samples)

    def to_df(self):
        out = pd.DataFrame({})
        for key in self._samples[0].items():
            out[key[0]] = [_[key[0]] for _ in self._samples]

        return out

    @beartype
    def export_snakemake_manifest(self, path: str = "data/snakemake_manifest.tsv") -> pd.DataFrame:
        """Write a flat per-sample manifest for a Snakemake-driven build.

        Additive helper — it does not touch any sync/check behaviour. It flattens
        the parsed, non-discarded entries (whose idat/hdf5 fields are objects and
        whose Tumor Type is a TumorType enum) into plain string columns, so a
        Snakefile can drive the per-sample builds without importing libcognition
        per job. One row per sample.
        """
        rows = []
        for entry in self._samples:
            tumor_type = entry['Tumor Type']
            rows.append({
                'Dataset':          entry['Dataset'],
                'Filename':         entry['Filename'],
                'Sample ID':        entry['Sample ID'],
                'Sentrix ID':       entry['Sentrix ID'],
                'Array type':       entry['Array type'],
                'Tumor Type':       None if tumor_type is None else tumor_type.name,
                'idat_grn':         entry['idat']['Grn'].path,
                'idat_red':         entry['idat']['Red'].path,
                'hdf5_mepylome':    entry['hdf5-mepylome'],
                'hdf5_pymetharray': entry['hdf5-pymetharray'],
            })

        manifest = pd.DataFrame(rows)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        manifest.to_csv(path, sep="\t", index=False)
        return manifest



