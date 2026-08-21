"""Automated, reproducible bioactivity download from ChEMBL and PubChem BioAssay.

Uses the official public REST APIs. Results are cached under ``data/raw`` so a
re-run reuses them (``download.reuse_cache``) instead of re-hitting the network.
Only quantitative / outcome-bearing records with a SMILES are retained.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, List

import pandas as pd

from .config import Config
from .utils import get_logger

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_UA = {"User-Agent": "DEEPENTXAI/1.0 (research)"}


def _get(url: str, timeout: int = 60, retries: int = 3):
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(1.0)
    return None


class ChEMBLDownloader:
    """Pull IC50/EC50/Ki/Kd/MIC activities for the configured organisms."""

    def __init__(self, cfg: Config, log):
        self.cfg, self.log = cfg, log

    def run(self) -> pd.DataFrame:
        dl = self.cfg["download"]
        organisms = self.cfg["target"]["organisms"]
        types = dl["chembl_activity_types"]
        max_pages = int(dl["chembl_max_pages_per_query"])
        rows: List[Dict] = []
        for org in organisms:
            n0 = len(rows)
            for typ in types:
                offset = 0
                for _ in range(max_pages):
                    q = urllib.parse.urlencode({"target_organism": org, "standard_type": typ,
                                                "limit": 1000, "offset": offset})
                    d = _get(f"{CHEMBL}?{q}")
                    if not d:
                        break
                    acts = d.get("activities", [])
                    if not acts:
                        break
                    for a in acts:
                        smi = a.get("canonical_smiles")
                        val = a.get("standard_value")
                        if smi and val is not None and a.get("standard_units"):
                            rows.append({
                                "smiles": smi, "value": val, "units": a.get("standard_units"),
                                "type": typ, "relation": a.get("standard_relation"),
                                "chembl_id": a.get("molecule_chembl_id"),
                                "assay_id": a.get("assay_chembl_id"),
                                "organism": org, "source": "ChEMBL"})
                    offset += 1000
                    if d["page_meta"].get("next") is None:
                        break
            self.log.info(f"ChEMBL {org}: +{len(rows) - n0} (total {len(rows)})")
        return pd.DataFrame(rows)


class PubChemDownloader:
    """Pull ACTIVE/INACTIVE compounds from Enterobacteriaceae PubChem bioassays."""

    def __init__(self, cfg: Config, log):
        self.cfg, self.log = cfg, log

    def _aids(self, org: str, n: int) -> List[str]:
        q = urllib.parse.urlencode({"db": "pcassay", "term": f'"{org}"[Target Organism]',
                                    "retmax": n, "retmode": "json", "sort": "relevance"})
        d = _get(f"{EUTILS}?{q}")
        return (d or {}).get("esearchresult", {}).get("idlist", [])

    def _cids(self, aid: str, kind: str) -> List[int]:
        d = _get(f"{PUG}/assay/aid/{aid}/cids/JSON?cids_type={kind}")
        info = (d or {}).get("InformationList", {}).get("Information", [{}])
        return info[0].get("CID", []) if info else []

    def _smiles(self, cids: List[int]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for i in range(0, len(cids), 100):
            chunk = cids[i:i + 100]
            d = _get(f"{PUG}/compound/cid/{','.join(map(str, chunk))}/property/SMILES/JSON")
            for p in (d or {}).get("PropertyTable", {}).get("Properties", []):
                smi = p.get("SMILES") or p.get("ConnectivitySMILES")
                if smi:
                    out[p["CID"]] = smi
            time.sleep(0.25)
        return out

    def run(self) -> pd.DataFrame:
        dl = self.cfg["download"]
        rows: List[Dict] = []
        seen = set()
        n_org = int(dl.get("pubchem_max_organisms", len(self.cfg["target"]["organisms"])))
        for org in self.cfg["target"]["organisms"][:n_org]:
            for aid in self._aids(org, int(dl["pubchem_max_assays_per_organism"])):
                if aid in seen or len(rows) >= int(dl["pubchem_max_total"]):
                    continue
                seen.add(aid)
                act = self._cids(aid, "active")
                ina = self._cids(aid, "inactive")[: int(dl["pubchem_max_inactive_per_assay"])]
                pairs = [(c, 1) for c in act] + [(c, 0) for c in ina]
                smap = self._smiles([c for c, _ in pairs])
                for cid, lab in pairs:
                    if cid in smap:
                        rows.append({"smiles": smap[cid], "label": lab, "cid": cid,
                                     "aid": aid, "organism": org, "source": "PubChem"})
            self.log.info(f"PubChem {org}: total {len(rows)}")
        return pd.DataFrame(rows).drop_duplicates(subset=["smiles", "label"])


class DataDownloader:
    """Orchestrate ChEMBL + PubChem download with caching under data/raw."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = get_logger("download", cfg.dir_logs)

    def _cached_or(self, name: str, builder) -> pd.DataFrame:
        path = os.path.join(self.cfg.dir_raw, name)
        if self.cfg["download"].get("reuse_cache", True) and os.path.exists(path):
            self.log.info(f"reusing cached {name}")
            return pd.read_csv(path)
        self.log.info(f"downloading {name} from API ...")
        df = builder()
        df.to_csv(path, index=False)
        self.log.info(f"saved {name}: {len(df)} rows")
        return df

    def download(self) -> Dict[str, pd.DataFrame]:
        out = {}
        if "ChEMBL" in self.cfg["download"]["sources"]:
            out["ChEMBL"] = self._cached_or("chembl_raw.csv", lambda: ChEMBLDownloader(self.cfg, self.log).run())
        if "PubChem" in self.cfg["download"]["sources"]:
            out["PubChem"] = self._cached_or("pubchem_raw.csv", lambda: PubChemDownloader(self.cfg, self.log).run())
        return out
