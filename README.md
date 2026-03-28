# dtangle

Python implementation of **dtangle** for deconvolution of bulk expression profiles using reference cell types.
Original R package can be found at [https://github.com/gjhunt/dtangle](https://github.com/gjhunt/dtangle).

## Install

```bash
pip install dtangle
```

Requires Python >=3.11.

## Quick Start

```python
from dtangle import deconvolut

deconvolut(
	mixture_adata,
	reference_adata,
	"cell_type",
	markers={"A": ["g0"], "B": ["g1"]},
	n_markers=1,
	key_added="dtangle",
)

# Results are written to:
# > mixture_adata.obsm["dtangle"]
# > mixture_adata.uns["dtangle"]
```

## Development

```bash
uv venv
make install
make check

# Build docs
make docs
```

## Project Links

- Source: https://github.com/harryhaller001/dtangle
- Issues: https://github.com/harryhaller001/dtangle/issues

## License

GPL Version 3

## Citation

Original `dtangle` publication:

```bibtex
@article{10.1093/bioinformatics/bty926,
    author = {Hunt, Gregory J and Freytag, Saskia and Bahlo, Melanie and Gagnon-Bartsch, Johann A},
    title = {dtangle: accurate and robust cell type deconvolution},
    journal = {Bioinformatics},
    volume = {35},
    number = {12},
    pages = {2093-2099},
    year = {2018},
    month = {11},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/bty926},
    url = {https://doi.org/10.1093/bioinformatics/bty926},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/35/12/2093/48934914/bioinformatics_35_12_2093.pdf},
}
```
