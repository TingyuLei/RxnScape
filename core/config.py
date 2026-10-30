import os


PERIODIC_TABLE_ELEMENTS = tuple("""
H He Li Be B C N O F Ne
Na Mg Al Si P S Cl Ar K Ca
Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr
Nb Mo Tc Ru Rh Pd Ag Cd In Sn
Sb Te I Xe Cs Ba La Ce Pr Nd
Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg
Tl Pb Bi Po At Rn Fr Ra Ac Th
Pa U Np Pu Am Cm Bk Cf Es Fm
Md No Lr Rf Db Sg Bh Hs Mt Ds
Rg Cn Nh Fl Mc Lv Ts Og
""".split())


class Config:
    """Global Configuration for RxnScape."""
    DIR_DATA = "00_Data"
    DIR_PLOTS = "01_Plots"
    DIR_STRUC = "02_Structures"

    # Every currently named IUPAC element is accepted.  The default stays C
    # for backward compatibility, while individual applications can select
    # any chemically relevant conserved element in rxn.yaml or on the CLI.
    SANKEY_ALLOWED_ELEMENTS = PERIODIC_TABLE_ELEMENTS
    SANKEY_DEFAULT_ELEMENTS = ("C",)

    OUTPUT_FILES = {
        "mol_csv": os.path.join(DIR_DATA, "molecule_counts.csv"),
        "bond_csv": os.path.join(DIR_DATA, "bond_counts.csv"),
        "ring_csv": os.path.join(DIR_DATA, "ring_counts.csv"),
        "raw_events": os.path.join(DIR_DATA, "reaction_events_raw.csv"),
        "clean_events": os.path.join(DIR_DATA, "reaction_events_clean.csv"),
        "pathway_events": os.path.join(DIR_DATA, "reaction_events_pathway.csv"),
        "unresolved_events": os.path.join(DIR_DATA, "reaction_events_unresolved.csv"),
        "stats_csv": os.path.join(DIR_DATA, "reaction_stats.csv"),
        "inter_csv": os.path.join(DIR_DATA, "species_report.csv"),
        
        "snapshots": os.path.join(DIR_STRUC, "reaction_snapshots"),
        "neb": os.path.join(DIR_STRUC, "auto_neb_inputs"),
        "intermediates": os.path.join(DIR_STRUC, "intermediate_structures")
    }

    @staticmethod
    def make_dirs():
        for d in [Config.DIR_DATA, Config.DIR_PLOTS, Config.DIR_STRUC]:
            if not os.path.exists(d): os.makedirs(d)
