# Data Provenance and Redistribution Status

## Included Data

This repository currently contains study data for two software projects:

- OpenStack
- Qt

Raw files:

```text
data/raw/openstack/openstack.csv
data/raw/qt/qt.csv
```

Processed files are stored under:

```text
data/processed/openstack/
data/processed/qt/
```

The packaged raw CSV files contain the following fields:

```text
dataset, commit_id, date, bug, fix, ns, nd, nf, entropy, la, ld, lt,
ndev, age, nuc, exp, rexp, sexp
```

## Provenance Status

The repository itself does **not currently contain a source citation,
download URL, or third-party license file that establishes the original
provenance and redistribution terms for the OpenStack and Qt datasets**.

Therefore, this document intentionally does not infer or invent an upstream
dataset source or license.

Before making the repository or release public, the dataset provenance and
redistribution terms should be verified against the original source from which
these files were obtained.

## Redistribution Guidance

Until the upstream license is verified:

- do not claim that the included data are covered by the repository's code license;
- do not apply a blanket software license to third-party data;
- keep the repository private for review preparation if necessary;
- if redistribution is not permitted, replace raw third-party data with a
  download/preparation script and cite the authoritative upstream source.

## Processed Data

Processed files were produced for this study from the included raw inputs.
However, whether processed derivatives may be redistributed can depend on the
license or terms of the original data source.

The same provenance verification should therefore be completed before public
release of both raw and processed data.

## Required Follow-Up Before Public Release

Document, for each dataset:

1. authoritative upstream source;
2. source URL or DOI;
3. original authors/maintainers;
4. applicable license or terms of use;
5. whether redistribution of raw data is permitted;
6. whether redistribution of processed derivatives is permitted;
7. required citation.

Once these facts are verified, update this file and add any required
third-party notices.
