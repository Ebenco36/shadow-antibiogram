# src/controllers/AMR/evaluation/label_manager.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from src.controllers.AMR.config.experiment_config import (
    ExperimentConfig,
    EvaluationConfig,
)


class LabelManager:
    """
    Loads and manages ground-truth label mappings for antibiotics.

    This class is *config-driven*:
      - It receives an ExperimentConfig or EvaluationConfig.
      - It reads JSON paths from config.evaluation.ground_truth_paths.

    Expected ground_truth_paths in EvaluationConfig:
        {
          "broad": Path("datasets/antibiotic_broad_class_grouping.json"),
          "fine":  Path("datasets/antibiotic_class_grouping.json"),
          "who":   Path("datasets/antibiotic_class.json"),
        }

    Each JSON file is expected to map:
        group_name -> [list of antibiotic column names]

    We invert this into:
        antibiotic_name -> group_name

    For WHO we normalize key names like "AccessList" -> "Access".
    """

    def __init__(self, config: Union[ExperimentConfig, EvaluationConfig]):
        """
        Parameters
        ----------
        config : ExperimentConfig or EvaluationConfig
            If ExperimentConfig is provided, we use config.evaluation.
            If EvaluationConfig is provided, we use it directly.
        """
        self.logger = logging.getLogger(__name__)

        if isinstance(config, ExperimentConfig):
            self.evaluation_config: EvaluationConfig = config.evaluation
        elif isinstance(config, EvaluationConfig):
            self.evaluation_config = config
        else:
            raise TypeError(
                "LabelManager expects ExperimentConfig or EvaluationConfig, "
                f"got {type(config)}"
            )

        self._ground_truth_paths: Dict[str, Path] = {
            level: Path(path)
            for level, path in self.evaluation_config.ground_truth_paths.items()
        }

        self._label_maps: Dict[str, Dict[str, str]] = self._load_all()

    # ------------------------------------------------------------------ #
    # Internal JSON loading helpers
    # ------------------------------------------------------------------ #

    def _load_json(self, path: Path) -> Dict[str, List[str]]:
        """
        Load JSON mapping group_name -> [antibiotic1, antibiotic2, ...].

        Returns
        -------
        mapping : dict
            {group_name: [antibiotic_names...]}
        """
        if not path.exists():
            raise FileNotFoundError(f"Ground truth JSON not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in {path}, got {type(data)}")
        return data

    def _invert_grouping(self, mapping: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Invert {group_name: [antibiotic1, antibiotic2, ...]}
        into {antibiotic: group_name}.
        """
        inverted: Dict[str, str] = {}
        for group_name, antibiotics in mapping.items():
            for ab in antibiotics:
                if ab in inverted and inverted[ab] != group_name:
                    # Only warn; keep first assignment
                    self.logger.warning(
                        "Antibiotic %r assigned to multiple groups: %r and %r. "
                        "Keeping first assignment (%r).",
                        ab,
                        inverted[ab],
                        group_name,
                        inverted[ab],
                    )
                    continue
                inverted[ab] = group_name
        return inverted

    def _normalize_who_group_name(self, group_name: str) -> str:
        """
        Optional normalization for WHO JSON keys,
        e.g. 'WatchList' -> 'Watch', 'AccessList' -> 'Access', etc.
        """
        if group_name.endswith("List"):
            return group_name.replace("List", "")
        return group_name

    def _load_all(self) -> Dict[str, Dict[str, str]]:
        """
        Load and invert all ground-truth JSONs.

        Returns
        -------
        label_maps : dict
            {
              "broad": {antibiotic -> broad_group_name},
              "fine":  {antibiotic -> fine_class_name},
              "who":   {antibiotic -> who_category_name},
            }
        """
        label_maps: Dict[str, Dict[str, str]] = {}

        for level, path in self._ground_truth_paths.items():
            raw = self._load_json(path)

            # Normalize WHO keys (AccessList -> Access, etc.)
            if level == "who":
                normalized_raw = {
                    self._normalize_who_group_name(k): v for k, v in raw.items()
                }
                inverted = self._invert_grouping(normalized_raw)
            else:
                inverted = self._invert_grouping(raw)

            label_maps[level] = inverted

        return label_maps

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_label_maps(self) -> Dict[str, Dict[str, str]]:
        """
        Returns all label mappings:
            {
              "broad": {antibiotic -> broad_group},
              "fine":  {antibiotic -> fine_class},
              "who":   {antibiotic -> Access/Watch/Reserve},
            }
        """
        return self._label_maps
