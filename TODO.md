# TODO

## Dependencies

- [ ] **Uitfaseren van `pymetharray`.** De code markeert dit zelf al als afbouw
      (`database.py`: `# @deprecated("phase pymetharray out")`,
      `validate_pymetharray=... # to be deprecated`). Resterend gebruik:
  - `utils.py` — `idat_to_data_container_pymetharray()`; enige caller is de
    hdf5-restore tak in `database.py::check_hdf5s()`. Vervanger voor de
    preprocessing is `idat_to_data_container_mepylome()`.
  - `database.py` — `IdatDataset` / `Channel` / `ArrayType` in de
    idat-header-read (`_read_header()`). Te vervangen door de `IDATreader`
    (idat-tools) die in dezelfde klasse al geconstrueerd wordt, plus
    `ArrayType` uit `mepylome.dtypes.arrays`. Section codes van beide
    parsers zijn identiek geverifieerd (barcode 402, chip label 404,
    n_probes 1000).
  - `SampleSheet` in `database.py` is al ongebruikt en kan zonder meer weg.

## Opruimen

- [ ] `database.py::idat.__init__` parseert het volledige IDAT-bestand enkel
      ter validatie en gooit het resultaat weg (`decoy = IDATreader(...)`),
      waarna `_read_header()` het bestand nogmaals opent. Twee reads, één
      gebruikt. `verify_file` staat default op `True` en wordt nergens
      uitgezet.
- [ ] Bare `except:` in diezelfde validatie maskeert de oorzaak van een
      parse-fout; `except Exception as e: ... from e` behoudt de traceback.
