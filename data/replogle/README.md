# Replogle K562 inputs

The analysis reads two preprocessed H5AD files from `raw/`:

- `Replogle-E-k562_pseudobulk.h5ad`, mean log1p expression per `(perturbation, batch)` group;
- `Replogle-E-k562_wilcoxon_batch_corrected.h5ad`, batch-stratified Wilcoxon differential expression of each perturbation against control.

## Source

Both files derive from the essential-gene K562 CRISPRi Perturb-seq screen of Replogle et al. (2022), *Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq*, Cell 185, 2559-2575, in the harmonized form distributed by scPerturb (Peidli et al., 2024, *Nature Methods*):

- Zenodo record [10.5281/zenodo.10044268](https://doi.org/10.5281/zenodo.10044268)
- file [`ReplogleWeissman2022_K562_essential.h5ad`](https://zenodo.org/records/10044268/files/ReplogleWeissman2022_K562_essential.h5ad?download=1), 1.55 GB, SHA-256 `412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789`.

## Building the inputs

From the repository root, in the `rcb` environment:

```sh
python -m experiments.replogle.preprocess
python -m experiments.replogle.run
```

`preprocess` downloads the scPerturb file, verifies its SHA-256 digest, and runs four crispyx steps:

1. quality control with `min_genes=100`, `min_cells_per_gene=100` and `min_cells_per_perturbation=20`, keeping 309,915 cells, 8,563 genes and 2,021 perturbations;
2. library-size normalization to `1e4` counts followed by `log1p`;
3. `cx.pb.aggregate(groupby=["perturbation", "batch"], method="mean_log1p", min_cells=5, random_state=0)`, giving 22,075 rows;
4. `cx.de.wilcoxon_test(perturbation_column="perturbation", control_label="control", batch_column="batch", corr_method="benjamini-hochberg")`, giving 2,021 rows with `pvalue_adj`, `z_score` and `logfoldchanges` layers.
