# src/controllers/AMR/evaluation/external_evaluator.py

from typing import Dict, List

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class ExternalEvaluator:
    """
    Computes external clustering quality metrics (ARI, NMI) against
    multiple ground truth label sets.

    label_maps is expected to be:
        {
          "broad": {antibiotic -> broad_group},
          "fine":  {antibiotic -> fine_class},
          "who":   {antibiotic -> Access/Watch/Reserve},
        }
    """

    def __init__(self, label_maps: Dict[str, Dict[str, str]]):
        self.label_maps = label_maps

    def evaluate(
        self,
        partition: Dict[str, int],
        nodes: List[str],
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        partition : dict
            Mapping {node -> cluster_id}.
        nodes : list
            Ordered list of nodes / antibiotics (must match similarity matrix order).

        Returns
        -------
        scores : dict
            Keys like:
              "ARI_broad", "NMI_broad",
              "ARI_fine",  "NMI_fine",
              "ARI_who",   "NMI_who", ...
        """
        scores: Dict[str, float] = {}

        # Cluster labels aligned with nodes
        clustering_labels = [partition[n] for n in nodes]

        for level_name, label_map in self.label_maps.items():
            # Ground-truth labels in same node order
            truth_labels = [label_map.get(n) for n in nodes]

            # Filter out nodes without labels (None)
            paired = [
                (c, t)
                for c, t in zip(clustering_labels, truth_labels)
                if t is not None
            ]
            if not paired:
                continue

            c_lab, t_lab = zip(*paired)

            ari = adjusted_rand_score(t_lab, c_lab)
            nmi = normalized_mutual_info_score(t_lab, c_lab)

            scores[f"ARI_{level_name}"] = float(ari)
            scores[f"NMI_{level_name}"] = float(nmi)

        return scores
