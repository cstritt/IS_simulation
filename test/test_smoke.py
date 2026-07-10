import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree

import abc_sampler
import niche_model


class SmokeTests(unittest.TestCase):
    def _write_tiny_inputs(self, tmp_path: Path):
        tree_path = tmp_path / "tiny_tree.nwk"
        tree_path.write_text("((A:0.1,B:0.1):0.05,(C:0.1,D:0.1):0.05);\n")

        metadata_path = tmp_path / "metadata.tsv"
        pd.DataFrame(
            {
                "GNUMBER": ["A", "B", "C", "D"],
                "LINEAGE_x": ["L1", "L1", "L2", "L2"],
            }
        ).to_csv(metadata_path, sep="\t", index=False)

        priors_path = tmp_path / "priors.yaml"
        priors_path.write_text(
            """
p_neutral:
  lower: 0.1
  upper: 0.3

gamma_shape:
  lower: 0.5
  upper: 1.0

gamma_scale:
  lower: 0.1
  upper: 0.2

r_birth:
  lower: 1.0e-4
  upper: 2.0e-4

r_loss:
  lower: 1.0e-4
  upper: 2.0e-4
"""
        )

        return tree_path, metadata_path, priors_path

    def test_single_simulation_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tree_path, metadata_path, _ = self._write_tiny_inputs(tmp_path)

            tree = Tree(str(tree_path), format=1)
            tip_names = sorted(tree.get_leaf_names())
            metadata = pd.read_csv(metadata_path, sep="\t")
            lineage_map = dict(zip(metadata["GNUMBER"], metadata["LINEAGE_x"]))

            params, cn, stats = niche_model.run_simulation(
                tree=tree,
                L=200,
                p_neutral=0.2,
                gamma_shape=0.8,
                gamma_scale=0.15,
                r_birth=1e-4,
                r_loss=1e-4,
                tip_names=tip_names,
                lineage_map=lineage_map,
                seed=42,
            )

            self.assertEqual(len(cn), len(tip_names))
            self.assertTrue(np.all(cn >= 0))
            self.assertEqual(params["r_birth"], 1e-4)
            self.assertTrue({"mean_cn", "gini_occupancy", "n_singletons"}.issubset(stats))

    def test_abc_sampler_smoke_n10(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tree_path, metadata_path, priors_path = self._write_tiny_inputs(tmp_path)
            outdir = tmp_path / "abc_out"

            params_df, stats_df, _, _ = abc_sampler.run_abc_sampler(
                str(tree_path),
                10,
                str(priors_path),
                str(metadata_path),
                str(outdir),
                n_workers=1,
            )

            self.assertEqual(len(params_df), 10)
            self.assertEqual(len(stats_df), 10)
            self.assertTrue(
                {"p_neutral", "gamma_shape", "gamma_scale", "r_birth", "r_loss"}.issubset(set(params_df.columns))
            )
            self.assertTrue({"mean_cn"}.issubset(set(stats_df.columns)))


if __name__ == "__main__":
    unittest.main()
