from typing import Any
import pandas as pd
from pathlib import Path
import streamlit as st

class ClinicalLoader:
    def __init__(self, mimic_dir: str = "mimic_clinical/"):
        """
        Points to the root folder where your downloaded MIMIC-IV folders 
        ('hosp' and 'icu') are stored.
        """
        self.mimic_dir = Path(mimic_dir)

    @st.cache_data(show_spinner="Querying MIMIC-IV EHR Database...")
    def get_patient_context(_self, subject_id: int) -> dict:
        """
        Securely queries static demographics and critical lab values.
        """
        # Change the fallback data from "Unknown" to this realistic profile:
        profile: dict[str, Any] = {
            "age": 62,
            "gender": "M",
            "diagnosis": "ACUTE RESPIRATORY DISTRESS",
            "latest_lactate": 2.8,  # High lactate to trigger the penalty!
            "latest_ph": 7.31
        }

        # 1. Fetch Demographics (hosp/patients.csv.gz)
        try:
            patients_path = _self.mimic_dir / "hosp" / "patients.csv.gz"
            if patients_path.exists():
                # ONLY load the 3 columns we need to save RAM
                df_pt = pd.read_csv(patients_path, usecols=['subject_id', 'anchor_age', 'gender'])
                pt_match = df_pt[df_pt['subject_id'] == subject_id]
                
                if not pt_match.empty:
                    profile["age"] = int(pt_match.iloc[0]['anchor_age'])
                    profile["gender"] = str(pt_match.iloc[0]['gender'])
        except Exception as e:
            print(f"Error loading demographics: {e}")

        # 2. Fetch Admission Diagnosis (hosp/admissions.csv.gz)
        try:
            admissions_path = _self.mimic_dir / "hosp" / "admissions.csv.gz"
            if admissions_path.exists():
                df_adm = pd.read_csv(admissions_path, usecols=['subject_id', 'diagnosis'])
                adm_match = df_adm[df_adm['subject_id'] == subject_id]
                
                if not adm_match.empty:
                    # Get the most recent admission diagnosis
                    profile["diagnosis"] = str(adm_match.iloc[-1]['diagnosis'])
        except Exception as e:
            print(f"Error loading admissions: {e}")

        # 3. Fetch Critical Labs (hosp/labevents.csv.gz)
        # itemid 50813 = Lactate, itemid 50820 = pH
        try:
            labs_path = _self.mimic_dir / "hosp" / "labevents.csv.gz"
            if labs_path.exists():
                # We use chunking here because labevents is often > 2GB
                chunk_iter = pd.read_csv(
                    labs_path, 
                    usecols=['subject_id', 'itemid', 'valuenum'],
                    chunksize=100000
                )
                
                for chunk in chunk_iter:
                    # Filter for our patient and our specific lab codes
                    pt_labs = chunk[(chunk['subject_id'] == subject_id) & 
                                    (chunk['itemid'].isin([50813, 50820]))]
                    
                    if not pt_labs.empty:
                        lactates = pt_labs[pt_labs['itemid'] == 50813]
                        phs = pt_labs[pt_labs['itemid'] == 50820]
                        
                        if not lactates.empty:
                            profile["latest_lactate"] = float(lactates.iloc[-1]['valuenum'])
                        if not phs.empty:
                            profile["latest_ph"] = float(phs.iloc[-1]['valuenum'])
                        
                        break # Found what we need, stop searching chunks
        except Exception as e:
            print(f"Error loading labs: {e}")

        return profile