import io
import re
import numpy as np
import pandas as pd
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt
from shadow_antibiogram.controllers.similarity.Metrics import (
    ACTJensenMetric, CFWSCosMetric, CFWSMetric, 
    ConditionalFractionMetric, CosineMetric, DiceMetric, IDFCFWSMetric, IDFCFWSPPMIMetric, IDFCosineFWSMetric, IDFCosinePPMIFWSMetric, 
    JaccardMetric, JensenShannonMetric, LiftMetric, NPMIIDFCFWSMetric, NPMIIDFCosineFWSMetric,
    OverlapMetric, ScaledLiftMetric, SimilarityMetric, TFIDFSimilarity, TverskyMetric, ACTCompositeMetric, 
    PhiMetric, PMIMetric, NPMIMetric, NPMI01Metric, MutualInformationMetric, NMIMetric, 
    YulesQMetric, YulesYMetric,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Callable, List, Optional, Tuple, Type, Dict, Union
from shadow_antibiogram.utils.helpers import plot_clustergram_with_dendrograms
alt.data_transformers.disable_max_rows()

class CoTestAnalyzer:
    def __init__(self, transactions: pd.DataFrame, antibiotic_cols: List[str]):
        self.transactions = transactions.copy()
        self.abx_cols = antibiotic_cols


    def _set_transaction(self, transactions: pd.DataFrame, antibiotic_cols: List[str]):
        self.transactions = transactions.copy()
        self.abx_cols = antibiotic_cols

    def remove_zero_columns(self) -> None:
        """
        Removes antibiotic columns from transactions and abx_cols where all values are 0.
        """
        zero_cols = [
            col for col in self.abx_cols if self.transactions[col].sum() == 0]
        if zero_cols:
            self.transactions.drop(columns=zero_cols, inplace=True)
            self.abx_cols = [
                col for col in self.abx_cols if col not in zero_cols]
        self.transactions.to_csv("complete_transactions.csv", index=False)

    def get_class_grouped_data(self, class_mapping: dict) -> pd.DataFrame:
        """
        Groups antibiotics by class and returns a new DataFrame.

        Args:
            class_mapping (dict): A dictionary mapping antibiotic columns to 
                                  their class.

        Returns:
            pd.DataFrame: A new DataFrame where columns are antibiotic classes.
        """
        class_df = pd.DataFrame()
        for abx_class, abx_list in class_mapping.items():
            # Find which antibiotics from the list are in our data
            valid_abx = [
                abx for abx in abx_list if abx in self.transactions.columns]
            if valid_abx:
                # If any antibiotic in the class was tested, mark the class as tested
                class_df[abx_class] = self.transactions[valid_abx].any(
                    axis=1).astype(int)
        return class_df

    def create_label_mapping(self, format_type: str = 'combined', remove_suffix: str = "_Tested") -> dict:
        """
        Creates a dictionary for cleaner labels in various formats.

        Args:
            format_type (str): The desired format for the labels. 
                            Options: 'abbr', 'full', 'combined'.
            remove_suffix (str): The suffix to remove from column names.

        Returns:
            dict: A mapping from old labels to new, cleaner labels.
        """
        label_map = {}
        for col in self.abx_cols:
            # First, remove the suffix to clean up the string
            cleaned_col = col.replace(remove_suffix, "")

            # Try to split the string into abbreviation and full name
            parts = cleaned_col.split(' - ', 1)

            if len(parts) == 2:
                abbr, full_name = parts
                if format_type == 'abbr':
                    label_map[col] = abbr
                elif format_type == 'full':
                    label_map[col] = full_name
                elif format_type == 'combined':
                    # Keep the "Abbr - Full Name" format
                    label_map[col] = cleaned_col
                else:
                    # Default to combined if format is invalid
                    label_map[col] = cleaned_col
            else:
                # If the label doesn't contain " - ", just use the cleaned version
                label_map[col] = cleaned_col

        return label_map

    ###########################################################
    ########### START OF METRICS IMPLEMENTATIONS ##############
    ###########################################################
    
    def _maybe_apply_fdr(self, metric, mat: pd.DataFrame, fdr: dict | None):
        if not fdr:
            return mat, None, None
        if not hasattr(metric, "fdr"):
            raise TypeError(f"{type(metric).__name__} does not support FDR (add FDRMixin).")
        res = metric.fdr(**fdr)
        pruned = pd.DataFrame(res.sim_pruned, index=mat.index, columns=mat.columns)
        edges  = metric.edges_from_fdr(res)
        return pruned, edges, res

    def compute_metric(
        self,
        metric_cls,
        *,
        right=None,
        fdr: dict | None = None,
        return_edges: bool = False,
        **kwargs
    ):
        metric = metric_cls(self.transactions, left_cols=self.abx_cols, right=right, **kwargs)
        mat = metric.compute()
        mat2, edges, _ = self._maybe_apply_fdr(metric, mat, fdr)
        return (mat2, edges) if return_edges else mat2

    def compute_metric_long(
        self,
        metric_cls,
        *,
        right=None,
        drop_self=True,
        triangle="upper",
        sort_by="similarity",
        ascending=False,
        round_to=3,
        topk=None,
        include_pvalues=False,
        fdr: dict | None = None,
        **kwargs
    ):
        metric = metric_cls(self.transactions, left_cols=self.abx_cols, right=right, **kwargs)
        mat = metric.compute()
        mat2, _, res = self._maybe_apply_fdr(metric, mat, fdr)
        df_long = metric.as_long(
            mat2, drop_self=drop_self, triangle=triangle, sort_by=sort_by,
            ascending=ascending, round_to=round_to, topk=topk
        )
        if include_pvalues and res is not None:
            p_df = pd.DataFrame(res.pvals, index=mat.index, columns=mat.columns).stack().rename("p_value").reset_index()
            q_df = pd.DataFrame(res.qvals, index=mat.index, columns=mat.columns).stack().rename("q_value").reset_index()
            p_df.columns = ["left","right","p_value"]; q_df.columns = ["left","right","q_value"]
            df_long = df_long.merge(p_df, on=["left","right"], how="left").merge(q_df, on=["left","right"], how="left")
        return df_long

    # ------------- convenience methods -------------
    # ------------- Our Similarity Metrics -------------
    def idfcfws(self, *, right: Optional[Union[List[str], str]] = None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(IDFCFWSMetric, right=right, **kwargs)
    
    def cosineidffws(self, *, right: Optional[Union[List[str], str]] = None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(IDFCosineFWSMetric, right=right, **kwargs)
    
    
    def idfppmicfws(self, *, right: Optional[Union[List[str], str]] = None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(IDFCFWSPPMIMetric, right=right, **kwargs)

    def cosineidfppmicfws(self, *, right: Optional[Union[List[str], str]] = None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(IDFCosinePPMIFWSMetric, right=right, **kwargs)

    
    def tfidf(self, *, right: Optional[Union[List[str], str]] = None, **kwargs) -> pd.DataFrame:
        """
        Compute TF–IDF cosine similarity between rows of the panel data.

        Args:
            right (list[str] | str | None): Optional right-hand subset (for rectangular sim).
            **kwargs: Extra args passed to TFIDFCosineMetric (e.g. smooth_idf=True).

        Returns:
            pd.DataFrame: Similarity matrix (square if right=None).
        """
        return self.compute_metric(TFIDFSimilarity, right=right, **kwargs)


    def npmiidfcfws(
        self,
        *,
        right: Optional[Union[List[str], str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute IDF-CFWS similarity matrix."""
        return self.compute_metric(NPMIIDFCFWSMetric, right=right, **kwargs)

    def npmicosineidffws(
        self,
        *,
        right: Optional[Union[List[str], str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute IDF-Cosine-FWS similarity matrix."""
        return self.compute_metric(NPMIIDFCosineFWSMetric, right=right, **kwargs)
    
    
    def jaccard(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(JaccardMetric, right=right, fdr=fdr, **kwargs)

    def jaccard_pairs(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, topk: Optional[int] = 20, **kwargs) -> pd.DataFrame:
        return self.compute_metric_long(JaccardMetric, right=right, topk=topk, fdr=fdr, **kwargs)

    def conditional_fraction(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(ConditionalFractionMetric, right=right, fdr=fdr, **kwargs)

    def lift(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(LiftMetric, right=right, fdr=fdr, **kwargs)

    def scaled_lift(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(ScaledLiftMetric, right=right, fdr=fdr, **kwargs)

    def dice(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(DiceMetric, right=right, fdr=fdr, **kwargs)

    def overlap(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(OverlapMetric, right=right, fdr=fdr, **kwargs)

    def cos(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(CosineMetric, right=right, fdr=fdr, **kwargs)

    def tversky(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, alpha: float = 0.5, beta: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(TverskyMetric, right=right, alpha=alpha, fdr=fdr, beta=beta, **kwargs)

    def cfws(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, alpha: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(CFWSMetric, right=right, alpha=alpha, fdr=fdr, **kwargs)

    def cfws_cos(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, alpha: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(CFWSCosMetric, right=right, alpha=alpha, fdr=fdr, **kwargs)

    def jensen_shannon(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(JensenShannonMetric, right=right, fdr=fdr, **kwargs)
    
    # ------- info-theoretic / association metrics -------

    def phi(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, **kwargs) -> pd.DataFrame:
        return self.compute_metric(PhiMetric, right=right, fdr=fdr, **kwargs)

    def pmi(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(PMIMetric, right=right, eps=eps, fdr=fdr, **kwargs)

    def npmi(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(NPMIMetric, right=right, eps=eps, fdr=fdr, **kwargs)

    def npmi01(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(NPMI01Metric, right=right, eps=eps, fdr=fdr, **kwargs)

    def mi(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(MutualInformationMetric, right=right, eps=eps, fdr=fdr, **kwargs)

    def nmi(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(NMIMetric, right=right, eps=eps, fdr=fdr, **kwargs)

    def yules_q(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(YulesQMetric, right=right, eps=eps, fdr=fdr, **kwargs)

    def yules_y(self, *, right: Optional[Union[List[str], str]] = None, fdr=None, eps: float = 0.5, **kwargs) -> pd.DataFrame:
        return self.compute_metric(YulesYMetric, right=right, eps=eps, fdr=fdr, **kwargs)


    # ----------------- ACT composite & friends -----------------

    def act(self,
            *,
            right: Optional[Union[List[str], str]] = None,
            weights: Tuple[float, float] = (0.5, 0.5),
            idf_alpha: float = 0.5,
            npmi_eps: float = 0.5,
            blend: str = "geo",
            use_support_gate: bool = True,
            gate_mode: str = "cap",
            gate_t: int = 5,
            gate_lambda: float = 12.0,
            compute_llr: bool = False,
            llr_eps: float = 0.5,
            **kwargs) -> pd.DataFrame:
        """
        ACT composite (IDF-CFWS ⊗ NPMI01). Set blend="add" for additive mix.
        If compute_llr=True, p-values will be available via analyzer.last_llr_pvalues.
        """
        metric = ACTCompositeMetric(
            self.transactions,
            left_cols=self.abx_cols,
            right=right,
            weights=weights,
            idf_alpha=idf_alpha,
            npmi_eps=npmi_eps,
            blend=blend,
            use_support_gate=use_support_gate,
            gate_mode=gate_mode,
            gate_t=gate_t,
            gate_lambda=gate_lambda,
            compute_llr=compute_llr,
            llr_eps=llr_eps,
            **kwargs
        )
        mat = metric.compute()
        # keep references for downstream filtering if requested
        self.last_act_components = {
            "idfcfws": pd.DataFrame(metric.idfcfws_, index=self.abx_cols if right is None else self.abx_cols,
                                    columns=self.abx_cols if right is None else (right if isinstance(right, list) else metric.right_labels)),
            "npmi01":  pd.DataFrame(metric.npmi01_,  index=self.abx_cols if right is None else self.abx_cols,
                                    columns=self.abx_cols if right is None else (right if isinstance(right, list) else metric.right_labels)),
            "gate":    None if metric.gate_ is None else pd.DataFrame(metric.gate_, index=self.abx_cols if right is None else self.abx_cols,
                                    columns=self.abx_cols if right is None else (right if isinstance(right, list) else metric.right_labels)),
        }
        if getattr(metric, "pvalues_", None) is not None:
            self.last_llr_G = pd.DataFrame(metric.llr_G_, index=mat.index, columns=mat.columns)
            self.last_llr_pvalues = pd.DataFrame(metric.pvalues_, index=mat.index, columns=mat.columns)
        else:
            self.last_llr_G = None
            self.last_llr_pvalues = None
        return mat
    
    
    def act_jensen(
        self,
        *,
        right: Optional[Union[List[str], str]] = None,
        weights: Tuple[float, float] = (0.5, 0.5),
        idf_alpha: float = 0.5,
        js_eps: float = 1e-12,
        blend: str = "geo",
        use_support_gate: bool = True,
        gate_mode: str = "cap",
        gate_t: int = 5,
        gate_lambda: float = 12.0,
        compute_llr: bool = False,
        llr_eps: float = 0.5,
        **kwargs
    ) -> pd.DataFrame:
        """
        ACT–Jensen composite (IDF-CFWS ⊗ JS01). Set blend="add" for additive mix.
        If compute_llr=True, p-values are available via analyzer.last_llr_pvalues.
        """
        metric = ACTJensenMetric(
            self.transactions,
            left_cols=self.abx_cols,
            right=right,
            weights=weights,
            idf_alpha=idf_alpha,
            js_eps=js_eps,
            blend=blend,
            use_support_gate=use_support_gate,
            gate_mode=gate_mode,
            gate_t=gate_t,
            gate_lambda=gate_lambda,
            compute_llr=compute_llr,
            llr_eps=llr_eps,
            # DO NOT pass **kwargs here unless your base/metric supports them
        )
        mat = metric.compute()

        # component matrices (use mat's index/cols to be robust to 'right' argument)
        self.last_actj_components = {
            "idfcfws": pd.DataFrame(metric.idfcfws_, index=mat.index, columns=mat.columns),
            "js01":    pd.DataFrame(metric.js01_,    index=mat.index, columns=mat.columns),
            "gate":    None if metric.gate_ is None else pd.DataFrame(metric.gate_, index=mat.index, columns=mat.columns),
        }

        if getattr(metric, "pvalues_", None) is not None:
            self.last_llr_G = pd.DataFrame(metric.llr_G_, index=mat.index, columns=mat.columns)
            self.last_llr_pvalues = pd.DataFrame(metric.pvalues_, index=mat.index, columns=mat.columns)
        else:
            self.last_llr_G = None
            self.last_llr_pvalues = None

        return mat
    

    ###########################################################
    ############ END OF METRICS IMPLEMENTATIONS ###############
    ###########################################################
    
    def compute_all_similarity_matrices(self, alpha: float = 0.5) -> Dict[str, pd.DataFrame]:
        return {
            'Jaccard': self.jaccard(),
            'ConditionalFraction': self.conditional_fraction(),
            'Lift': self.lift(),
            'ScaledLift': self.scaled_lift(),
            'CFWS': self.cfws(alpha=alpha),
            'CFWS_Lift': self.cfws_lift(alpha=alpha),
            'CFWS_Cos': self.cfws_cos(alpha=alpha),
            'JensenShannon': self.jensen_shannon(),
        }

    def format_metric_names(self, metrics: list[str]) -> list[str]:
        def format_metric(metric: str) -> str:
            parts = metric.split('_')
            formatted_parts = []
            for part in parts:
                if part.isupper():  # already an acronym
                    formatted_parts.append(part)
                elif part.lower() in {"cfws"}:  # known acronyms
                    formatted_parts.append(part.upper())
                else:
                    formatted_parts.append(part[0].upper() + part[1:])
            return '_'.join(formatted_parts)

        return [format_metric(m) for m in metrics]

    def stratified_all_similarity_matrices(
        self,
        by: Optional[str] = None,
        alpha: float = 0.5,
        filter_values: Optional[List[str]] = None,
        prefilter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        df = self.transactions

        # Apply prefilter if provided
        if prefilter is not None:
            df = prefilter(df)

        results = {}

        if by is None:
            if len(df) < 2:
                return {}
            analyzer = CoTestAnalyzer(df, self.abx_cols)
            try:
                results["all"] = analyzer.compute_all_similarity_matrices(
                    alpha=alpha)
            except Exception as e:
                print(f"Error computing matrices for full dataset: {e}")
        else:
            if by not in df.columns:
                raise ValueError(f"'{by}' column not found in transactions.")

            for val, group in df.groupby(by):
                if filter_values is not None and val not in filter_values:
                    continue
                if len(group) < 2:
                    continue
                analyzer = CoTestAnalyzer(group, self.abx_cols)
                try:
                    results[val] = analyzer.compute_all_similarity_matrices(
                        alpha=alpha)
                except Exception as e:
                    print(f"Error computing matrices for group '{val}': {e}")

        return results

    def mine_association_rules(self, min_support: float = 0.01, min_confidence: float = 0.3, min_lift: float = 1.0) -> pd.DataFrame:
        df = self.transactions[self.abx_cols].copy()
        df = df.fillna(0).astype(bool)
        frequent_itemsets = apriori(
            df, min_support=min_support, use_colnames=True)
        rules = association_rules(
            frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules = rules[(rules['confidence'] >= min_confidence)]
        return rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

    def stratified_similarity(self, by: str, metric_func: str = "jaccard", alpha: float = 0.5) -> Dict[str, pd.DataFrame]:
        results = {}
        if by not in self.transactions.columns:
            raise ValueError(f"'{by}' column not found in transactions.")
        for val, group in self.transactions.groupby(by):
            if len(group) < 2:
                continue
            analyzer = CoTestAnalyzer(group, self.abx_cols)
            method = getattr(analyzer, metric_func)
            try:
                if "alpha" in method.__code__.co_varnames:
                    results[val] = method(alpha=alpha)
                else:
                    results[val] = method()
            except Exception as e:
                print(f"Error computing metric for group '{val}': {e}")
        return results

    def stratified_association_rules(self, by: str, min_support: float = 0.01, min_confidence: float = 0.3, min_lift: float = 1.0, groups: List[str] = None) -> Dict[str, pd.DataFrame]:
        if by not in self.transactions.columns:
            raise ValueError(f"'{by}' column not found in transactions.")
        output = {}
        for val, group in self.transactions.groupby(by):
            if groups and val not in groups:
                continue
            if len(group) < 2:
                continue
            analyzer = CoTestAnalyzer(group, self.abx_cols)
            try:
                rules = analyzer.mine_association_rules(
                    min_support, min_confidence, min_lift)
                output[val] = rules
            except Exception as e:
                print(f"Error mining rules for group '{val}': {e}")
        return output

    def plot_similarity_comparison(self, matrices: Dict[str, pd.DataFrame], metrics: List[str]) -> alt.Chart:
        all_charts = []

        # Define custom color gradient

        custom_domain = [0.0, 0.1, 0.3, 0.6, 1.0]
        colorscale = ['#ffffff', '#e6eff8', '#b2cfea', '#669ed4', '#005eb8']

        for metric in metrics:
            matrix = matrices[metric].copy()
            matrix = matrix.reset_index().melt(
                id_vars='index', var_name='target', value_name='similarity')
            matrix.rename(columns={'index': 'source'}, inplace=True)

            chart = alt.Chart(matrix).mark_rect().encode(
                x=alt.X('target:N', sort=matrix['target'].unique(
                ).tolist(), title='Target'),
                y=alt.Y('source:N', sort=matrix['source'].unique(
                ).tolist(), title='Source'),
                color=alt.Color('similarity:Q',
                                scale=alt.Scale(
                                    domain=custom_domain,
                                    range=colorscale
                                ),
                                legend=alt.Legend(title="Similarity")),
                tooltip=['source:N', 'target:N', 'similarity:Q']
            ).properties(
                width=300,
                height=300,
                title=metric
            )

            all_charts.append(chart)

        return alt.hconcat(*all_charts).resolve_scale(color='shared').properties(
            title="Co-Testing Similarity Matrices"
        )

    def plot_stratified_network_graphs_grid(
        self,
        by: str,
        metric: Union[str, List[str]] = "Jaccard",
        threshold: float = 0.3,
        alpha: float = 0.5,
        groups: Optional[List[str]] = None,
        prefilter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        title_suffix: Optional[str] = None
    ) -> go.Figure:

        stratified = self.stratified_all_similarity_matrices(
            by=by, alpha=alpha, filter_values=groups, prefilter=prefilter
        )

        # Refilter stratified if group list is defined
        if groups is not None:
            stratified = {g: stratified[g] for g in groups if g in stratified}

        # Normalize metric list
        if isinstance(metric, str):
            metric = [metric]

        # Safety check: skip empty
        if len(stratified) == 0:
            raise ValueError(
                "No groups found after filtering. Check `groups` and `prefilter`.")

        num_groups = len(stratified)
        num_metrics = len(metric)

        # Calculate safe vertical spacing
        # ensures layout stays valid
        safe_vertical_spacing = min(0.05, 0.9 / max(1, num_groups - 1))
        print(safe_vertical_spacing)
        fig = make_subplots(
            rows=num_groups,
            cols=num_metrics,
            subplot_titles=[
                f"{group} - {met}" for group in stratified for met in metric
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.009
        )

        for row_idx, (group, matrices) in enumerate(stratified.items(), start=1):
            for col_idx, met in enumerate(metric, start=1):
                if met not in matrices:
                    continue

                sim_df = matrices[met].copy()
                sim_df.values[np.tril_indices_from(
                    sim_df)] = 0  # keep upper triangle

                # Create edges based on threshold
                edges = [
                    (i, j, sim_df.iloc[i, j])
                    for i in range(len(sim_df))
                    for j in range(len(sim_df))
                    if sim_df.iloc[i, j] >= threshold
                ]

                if not edges:
                    continue  # skip empty graphs

                G = nx.Graph()
                for node in sim_df.index:
                    G.add_node(node)
                for i, j, w in edges:
                    G.add_edge(sim_df.index[i], sim_df.columns[j], weight=w)

                pos = nx.spring_layout(G, seed=42)

                # Edge coordinates
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                # Node coordinates
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_labels = list(G.nodes())

                # Add edges
                fig.add_trace(
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    ),
                    row=row_idx, col=col_idx
                )

                # Add nodes
                fig.add_trace(
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_labels,
                        marker=dict(size=10, color='#005eb8'),
                        textposition="bottom center"
                    ),
                    row=row_idx, col=col_idx
                )

        fig.update_layout(
            height=max(1000, 1000 * num_groups),
            width=max(1300, 1300 * num_metrics),
            showlegend=False,
            hovermode='closest',
            title_text=f"Stratified Co-Testing Networks {title_suffix}",
            margin=dict(t=30)
        )

        output_formats = ['html', 'png', 'pdf']
        buffers = {}
        for fmt in output_formats:
            if fmt == "html":
                buf = io.StringIO()
                fig.write_html(buf)
                buf.seek(0)
                buffers[fmt] = buf.getvalue().encode("utf-8")  # str to bytes
            else:
                buf = io.BytesIO()
                fig.write_image(buf, format=fmt)
                buf.seek(0)
                buffers[fmt] = buf.getvalue()

        return buffers

    def plot_dendrogram(self, matrix: pd.DataFrame, method: str = "average") -> None:
        linked = linkage(1 - matrix.values, method=method)
        plt.figure(figsize=(10, 6))
        dendrogram(linked, labels=matrix.columns.tolist(), leaf_rotation=90)
        plt.title("Antibiotic Co-Testing Dendrogram")
        plt.tight_layout()
        plt.show()

    def plot_stratified_dendrograms(
        self,
        by: str,
        metrics: List[str] = ["CFWS"],
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (14, 6),
        groups: Optional[List[str]] = None,
        prefilter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        title_suffix: Optional[str] = None
    ) -> Dict[str, bytes]:

        stratified = self.stratified_all_similarity_matrices(
            by=by,
            alpha=alpha,
            filter_values=groups,
            prefilter=prefilter
        )

        if not stratified:
            raise ValueError(
                "No groups found after filtering. Check `groups` or `prefilter`.")

        title_suffix = title_suffix or ""

        n_rows = len(stratified)
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(
            figsize[0], figsize[1] * n_rows))

        # Normalize axes into 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        for i, (group, matrix_dict) in enumerate(stratified.items()):
            for j, metric in enumerate(metrics):
                ax = axes[i][j]
                if metric not in matrix_dict:
                    ax.set_visible(False)
                    continue

                mat = matrix_dict[metric].copy()
                np.fill_diagonal(mat.values, 0)

                try:
                    # Convert similarity to distance
                    dist = 1 - mat.values
                    condensed = squareform(dist, checks=False)

                    linked = linkage(condensed, method='average')
                    dendrogram(linked, labels=mat.index.tolist(),
                               ax=ax, leaf_rotation=90)
                    ax.set_title(f"{group} - {metric}", fontsize=10)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                    ax.axis('off')

        fig.suptitle(
            f"Stratified Dendrograms by {by}{title_suffix}", fontsize=16, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save to buffers
        output_formats = ['png', 'pdf', 'svg']
        buffers = {}
        for fmt in output_formats:
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, bbox_inches='tight')
            buf.seek(0)
            buffers[fmt] = buf.getvalue()

        plt.close(fig)
        return buffers
