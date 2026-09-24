#!/usr/bin/env python

import glob
import gzip
import h5py
import os
import pickle
import re
import shutil
import subprocess
import sys
import tables
import termios
import threading
import tty

from contextlib import contextmanager

import numpy as np
import pandas as pd



from pathlib import Path


from beartype import beartype
from mepylome import IdatParser, MethylData
from mepylome.dtypes.arrays import ArrayType
from mepylome.dtypes.manifests import Manifest
from pymetharray.files import SampleSheet
from pymetharray.processing import SampleDataContainer

__all__ = [
    'MEPYLOME_KEYS',
    'idat_to_data_container_pymetharray',
    'idat_to_data_container_mepylome',
    'idat_channel',
    'interruptible',
    'verify_idat_channels',
    'idat_to_mvalues',
    'epicv2_to_epic',
    'gdc_client_download',
    'is_gz_file',
]


@contextmanager
def interruptible(keys=('q', 'Q')):
    """Yield an Event that gets set when the user presses one of *keys*.

    Lets a long loop be skipped without Ctrl-C (which would abort the whole
    run). cbreak gives single-char reads without Enter while keeping output
    processing intact for tqdm. The terminal settings are restored in a finally
    on the way out, so an exception in the loop body cannot leave the shell in
    cbreak mode.

    Without a terminal (batch jobs, `nbconvert --stdout | python -`) no watcher
    is started: tcgetattr only works on a tty and would otherwise raise inside
    the thread. The loop then simply runs to completion.

    Lives here rather than in database.py so mnp.py and methylscape.py can use
    it too; those are imported *by* database.py, so importing from there back
    would be circular.
    """
    quit_event = threading.Event()

    if not sys.stdin.isatty():
        yield quit_event
        return

    def _watch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not quit_event.is_set():
                if sys.stdin.read(1) in keys:
                    quit_event.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        yield quit_event
    finally:
        quit_event.set()  # stop watcher thread if the loop ended normally


# EXTENSION control probes target the four nucleotides. The C/G extension
# controls fluoresce in the green channel, the A/T extension controls in the
# red channel. See the Illumina array control probe annotation.
_GRN_EXTENSION_TYPES = ('Extension (C)', 'Extension (G)')
_RED_EXTENSION_TYPES = ('Extension (A)', 'Extension (T)')

# Cache Manifest objects per array type; loading them from disk is expensive.
_manifest_cache: dict = {}


def _extension_control_addresses(array_type: ArrayType) -> tuple[list, list]:
    """Return (green_addresses, red_addresses) of the EXTENSION control probes."""
    if array_type not in _manifest_cache:
        _manifest_cache[array_type] = Manifest(array_type)
    cdf = _manifest_cache[array_type].control_data_frame

    ext = cdf[cdf.Control_Type == 'EXTENSION']
    grn = ext[ext.Extended_Type.isin(_GRN_EXTENSION_TYPES)].Address_ID.astype(int).tolist()
    red = ext[ext.Extended_Type.isin(_RED_EXTENSION_TYPES)].Address_ID.astype(int).tolist()
    return grn, red


@beartype
def idat_channel(idat_file: str, ratio_threshold: float = 5.0) -> str:
    """
    Determine whether an IDAT file holds the Green (Grn) or Red channel.

    The physical channel of an IDAT file cannot be trusted from its file name
    alone. Instead this uses the intensity of the EXTENSION control probes:
    the C/G extension controls fluoresce in the green channel while the A/T
    extension controls fluoresce in the red channel. Within a single IDAT file
    the control group matching the file's channel is ~50-100x brighter than the
    other, which identifies the channel independent of the file name.

    Args:
        idat_file: path to a single Grn or Red IDAT file.
        ratio_threshold: minimum brightness ratio between the two control
            groups required to make a confident call. A pair that is closer
            than this is treated as ambiguous (e.g. corrupt or unexpected
            array).

    Returns:
        'Grn' or 'Red'.

    Raises:
        ValueError: if the array type / EXTENSION control probes cannot be
            resolved, or if the two control groups are too close to call.
    """
    parser = IdatParser(idat_file)
    array_type = ArrayType.from_probe_count(parser.n_snps_read)

    grn_addr, red_addr = _extension_control_addresses(array_type)
    if not grn_addr or not red_addr:
        raise ValueError(
            f"No EXTENSION control probes for array type '{array_type}' "
            f"(from {idat_file}); cannot determine channel."
        )

    intensities = pd.Series(
        np.asarray(parser.probe_means), index=np.asarray(parser.illumina_ids)
    )
    grn_signal = intensities.reindex(grn_addr).mean()
    red_signal = intensities.reindex(red_addr).mean()

    hi, lo = max(grn_signal, red_signal), min(grn_signal, red_signal)
    if not lo > 0 or hi / lo < ratio_threshold:
        raise ValueError(
            f"Ambiguous channel for {idat_file}: green control signal "
            f"{grn_signal:.1f} vs red control signal {red_signal:.1f} "
            f"(ratio below {ratio_threshold})."
        )

    return 'Grn' if grn_signal > red_signal else 'Red'


@beartype
def verify_idat_channels(idat_grn: str, idat_red: str) -> tuple[str, str]:
    """Guard against swapped Grn/Red arguments.
    The _Grn/_Red suffix in a file name is not authoritative, so verify the
    actual channel from the control-probe intensities (see idat_channel)
    before running any classifier.
    """
    grn_detected = idat_channel(idat_grn)
    red_detected = idat_channel(idat_red)

    if grn_detected == 'Grn' and red_detected == 'Red':
        return idat_grn, idat_red
    elif grn_detected == 'Red' and red_detected == 'Grn':
        return idat_red, idat_grn
    else:
        raise ValueError(
            f"Unable to determine channels: {idat_grn} -> {grn_detected}, "
            f"{idat_red} -> {red_detected}"
        )


def idat_to_data_container_pymetharray(idat_grn: str, idat_red: str,
                           retain_uncorrected_probe_intensities: bool = False,
                           pval: bool = False):
    """
    Load an idat pair, build and process a SampleDataContainer.

    Returns (data_container, array_type, sentrix_id).
    """
    
    print(f"retain_uncorrected_probe_intensities: {retain_uncorrected_probe_intensities}")
    print(f"pval: {pval}")
    
    ss = SampleSheet({'Grn': idat_grn, 'Red': idat_red}, recursive=False)
    print(f"idats: {idat_grn}")
    print(f"       {idat_red}")
    print(f"ss: {ss}")
    sample = [_ for _ in ss][0]
    sample.load()

    data_container = SampleDataContainer(
        idat_dataset_pair={
            'green_idat': sample.green_idat,
            'red_idat':   sample.red_idat,
        },
        manifest=sample.manifest,
        retain_uncorrected_probe_intensities=retain_uncorrected_probe_intensities,
        bit='float32',
        switch_probes=True,
        quality_mask=True,
        do_noob=True,
        pval=pval,
        poobah_decimals=3,
        poobah_sig=0.05,
        do_nonlinear_dye_bias=True,
        sesame=True,
        pneg_ecdf=False,
        file_format='hdf5',
    )
    data_container.process_all()

    array_type = str(sample.manifest.array_type)

    return data_container, array_type, sample.get_sentrix_id()


MEPYLOME_KEYS = [
    'mvalues_illumina',     'mvalues_swan',     'mvalues_noob',
    'betas_illumina',       'betas_swan',       'betas_noob',
    'intensities_illumina', 'intensities_swan', 'intensities_noob',  # total intensity per CpG (methylated + unmethylated)
]


@beartype
def idat_to_data_container_mepylome(
    idat_grn: str,
    idat_red: str,
    keys: list[str] #| None = None,
) -> tuple[pd.DataFrame, str, str]:
    """
    Load an idat pair with mepylome and return a combined DataFrame.

    keys: subset of MEPYLOME_KEYS, e.g. ['mvalues_swan'] or ['betas_illumina', 'mvalues_noob'].
    Each key encodes <type>_<prep> where type is 'mvalues', 'betas' or
    'intensities' and prep is 'illumina', 'swan' or 'noob'. The 'intensities'
    type is the total intensity per CpG (methylated + unmethylated), matching
    the signal mepylome's CNV uses.

    Returns (df, array_type, barcode) where df is float32 with columns
    '{barcode}_{key}', array_type is one of '450k'/'epic'/'epicv2'.
    """

    #if keys is None:
    #    keys = ['mvalues_swan']

    invalid = [k for k in keys if k not in MEPYLOME_KEYS]
    if invalid:
        raise ValueError(f"Invalid keys {invalid}. Must be from: {MEPYLOME_KEYS}")

    fn = [idat_grn, idat_red]

    # run each required prep only once (dict.fromkeys preserves order)
    unique_preps = dict.fromkeys(k.split('_', 1)[1] for k in keys)
    prep_data = {prep: MethylData(file=fn, prep=prep, seed=42) for prep in unique_preps}

    first = next(iter(prep_data.values()))
    barcode = first.sample_ids[0]
    array_type = str(first.array_type)  # '450k' / 'epic' / 'epicv2'

    frames = []
    for key in keys:
        dtype, prep = key.split('_', 1)
        md = prep_data[prep]
        if dtype == 'mvalues':
            src = md.mvalues
        elif dtype == 'betas':
            src = md.betas
        else:  # 'intensities' -> total intensity per CpG (methylated + unmethylated)
            src = md.methylated + md.unmethylated
        frames.append(src.rename(columns={barcode: f"{barcode}_{key}"}))

    ref = frames[0].index
    assert all(f.index.equals(ref) for f in frames[1:]), "Index mismatch across keys"

    return pd.concat(frames, axis=1).astype('float32'), array_type, barcode


def epicv2_to_epic(df):
    """
    #h5['chemistry'] = [re.sub(r'^.+_', '', s) for s in idx]
    # .drop(columns=['chemistry'])
    """
    cleaned_index = df.index.str.replace(r'_.+$', '', regex=True)
    df = df.groupby(cleaned_index).median()
    df.index.name = 'IlmnID'
    return df


@beartype
def idat_to_mvalues(idat_grn: str, idat_red: str) -> tuple[pd.DataFrame, str, str]:
    idat_grn, idat_red = verify_idat_channels(idat_grn, idat_red)

    df, array_type, sentrix_id = idat_to_data_container_mepylome(
        idat_grn, idat_red, keys=['mvalues_swan']
    )

    col = f"{sentrix_id}_mvalues_swan"
    mvalues = df[[col]].rename(columns={col: 'm_value'})

    if array_type == 'epicv2':
        mvalues = epicv2_to_epic(mvalues)

    return mvalues, array_type, sentrix_id


def gdc_client_download(uuid, path):
    tmpdir = "/tmp/gdc-client-tmp/"
    
    os.makedirs(tmpdir, exist_ok = True)
    os.makedirs(os.path.dirname(path), exist_ok = True)

    cli_cmd = ["gdc-client", "download", "-d", tmpdir, "--color_off", "--debug", uuid] # --debug is necessary to acquire explicit output filename
    #"--retry-amount", "5", "--wait-time",  "5", 
 
    print("Downloading (gdc-client): " + " ".join(cli_cmd) + "")

    idat = None
    succes = False

    with subprocess.Popen(cli_cmd, stdout=subprocess.PIPE, universal_newlines=True) as popen:
        for line in iter(popen.stdout.readline, ""):
            line = line.strip()
            print(line)

            if line.find("ERROR") > -1:
                raise Exception(line)
            #elif line.find(".idat") > -1 and line.find(".idat.parcel") == -1:
            #	idat.append(line)
            elif line.find("Downloading file to") > -1:
                idat = line.split("Downloading file to", 1)[1].strip().strip(":").strip()
            elif line.find("INFO: Successfully") > -1:
                print("download succesfully")
                succes = True

    if idat is None and succes == False:
        raise Exception("please check gdc-client output for file path")
    else:
        print("check1")
        print("idat", idat)
        if idat is None:
            print('check2: ' + tmpdir + uuid)
            # file is there, just find the path
            idat = [_ for _ in glob.glob(tmpdir + uuid + "/*.idat")]
            print(idat)
            idat = idat[0]
        
        shutil.move(idat, path)
        rmpath = os.path.dirname(idat)
        
        if len(rmpath) < 55:
            raise Exception("probably something odd happened")
        else:
            shutil.rmtree(rmpath)



def is_gz_file(filepath):
    print(filepath)
    with open(filepath, 'rb') as test_f:
        first_bytes = test_f.read(2)
        print(first_bytes)
        return first_bytes == b'\x1f\x8b'

