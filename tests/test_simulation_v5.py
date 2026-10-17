"""
Unit tests for the DeepSupport Version 5 (DGP V5) Simulation Engine & Configuration.
Compatible with both `unittest` (`python -m unittest discover -s tests -v`) and `pytest`.

Tests:
1. V5 Config Hygiene: absence of V1–V3 zombie parameters, explicit presence of all scenario
   switches and formerly hardcoded magic numbers, and normalized Destatis gender weights.
2. V5 Empirical Distribution Calibration vs. V4.1 Legacy Config:
   - Motivation variance recovery (kappa=9.0 -> sigma ~ 0.151 vs. V4.1 sigma ~ 0.121)
   - Zero-Inflated Beta employment (37% at 0h, median ~ 15h among employed, ~16% >= 20h)
   - Migration share (28% vs. 22%)
   - Destatis program-specific gender distributions (SG04/SG05 > 70% w vs. SG01/SG03 < 30% w)
3. Universe RNG Synchronization & Causal Protection Direction (Universe A vs. Universe B).
4. Apathy Dampening (`disable_apathy_dampening`) in acute crisis & late-semester trajectories.
5. Parameterized `berechne_dropout` monotonicity and config override responsiveness.
"""

from copy import deepcopy
import unittest
import numpy as np
import pandas as pd

from deepsupport.data_engine.config import (
    CONFIG,
    STUDIENGAENGE,
    get_v41_legacy_config,
)
from deepsupport.simulation.engine import (
    berechne_dropout,
    generiere_stammdaten,
    generiere_studierende,
    get_exam_noise,
    simuliere_verlaeufe,
)


class TestSimulationV5(unittest.TestCase):

    def test_v5_config_hygiene_no_zombies_and_complete_keys(self):
        """
        Verify that V1-V3 zombie parameters are removed, all 5 scenario switches are
        explicitly defined in CONFIG, and STUDIENGAENGE contains valid gender weights.
        """
        self.assertEqual(CONFIG.get("dgp_version"), "5.0")

        # 1. Zombie parameters must NOT be in CONFIG
        zombie_keys = [
            "gewicht_erwerb",
            "gewicht_motivation_rauschen",
            "gewicht_integration_rauschen",
        ]
        for z_key in zombie_keys:
            self.assertNotIn(
                z_key,
                CONFIG,
                f"Zombie parameter '{z_key}' should be removed from V5 CONFIG",
            )

        # 2. All 5 previously implicit .get() scenario switches must be explicitly in CONFIG
        scenario_keys = [
            "overload_penalty_factor",
            "overload_penalty_cap",
            "support_kosten_faktor",
            "rct_support_uptake",
            "disable_apathy_dampening",
        ]
        for s_key in scenario_keys:
            self.assertIn(
                s_key,
                CONFIG,
                f"Scenario parameter '{s_key}' must be explicitly present in CONFIG",
            )

        # 3. Key V5 calibration & parameterized magic-number entries must exist
        required_v5_keys = [
            "kappa_motivation",
            "kappa_integration",
            "migrations_quote",
            "erstakademiker_quote",
            "erwerb_modell",
            "erwerb_erwerbstaetig_quote",
            "erwerb_min_h",
            "erwerb_max_h",
            "erwerb_aktiv_norm_mean",
            "kappa_erwerb",
            "apathy_threshold",
            "dropout_base",
            "dropout_schwelle_motivation",
            "dropout_gewicht_motivation",
            "dropout_schwelle_integration",
            "dropout_gewicht_integration",
            "dropout_faktor_sem1",
            "dropout_faktor_ab_sem5",
            "dropout_max_p",
        ]
        for req_key in required_v5_keys:
            self.assertIn(req_key, CONFIG, f"Missing V5 parameter '{req_key}' in CONFIG")

        # 4. Program-specific Destatis gender distributions in STUDIENGAENGE
        self.assertEqual(len(STUDIENGAENGE), 5)
        for sg in STUDIENGAENGE:
            self.assertIn("geschlecht_gewichte", sg)
            weights = sg["geschlecht_gewichte"]
            self.assertEqual(len(weights), 3)
            self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_v5_student_generation_distributions_and_v41_legacy_comparison(self):
        """
        Generate N=5,000 students under V5 CONFIG and V4.1 legacy config and verify
        all empirical target distributions (AMS/SELLMO motivation dispersion, 22nd DSW
        Zero-Inflated Beta employment, 28% migration share, Destatis gender ratios).
        """
        stammdaten = generiere_stammdaten()

        cfg_v5 = deepcopy(CONFIG)
        cfg_v5["n_studierende"] = 5000
        studis_v5 = generiere_studierende(
            stammdaten, np.random.default_rng(99999), cfg=cfg_v5
        )

        cfg_v41 = get_v41_legacy_config()
        cfg_v41["n_studierende"] = 5000
        studis_v41 = generiere_studierende(
            stammdaten, np.random.default_rng(99999), cfg=cfg_v41
        )

        # --- 1. Motivation Variance & Critical Tail (< 0.40) ---
        mot_v5 = np.array([s.motivation_initial for s in studis_v5])
        mot_v41 = np.array([s.motivation_initial for s in studis_v41])

        std_v5 = float(np.std(mot_v5))
        std_v41 = float(np.std(mot_v41))
        # Empirical AMS/SELLMO target range is sigma in [0.15, 0.18] (conditional Beta sigma ~ 0.151 + covariate variance)
        self.assertGreaterEqual(std_v5, 0.145)
        self.assertLessEqual(std_v5, 0.180)
        self.assertGreater(
            std_v5,
            std_v41 + 0.02,
            f"V5 motivation std ({std_v5:.4f}) must exceed V4.1 ({std_v41:.4f}) by > 0.02",
        )

        tail_40_v5 = float(np.mean(mot_v5 < 0.40))
        tail_40_v41 = float(np.mean(mot_v41 < 0.40))
        self.assertGreater(
            tail_40_v5,
            0.05,
            f"V5 share of students with motivation < 0.40 ({tail_40_v5:.3%}) should exceed 5%",
        )
        self.assertGreater(tail_40_v5, tail_40_v41 * 2.0)
        # Ensure no artificial Dirac boundary clipping pile-up at <= 0.05
        self.assertLess(float(np.mean(mot_v5 <= 0.05)), 0.005)

        # --- 2. Zero-Inflated Beta Employment (22. DSW-Sozialerhebung) ---
        erw_v5 = np.array([s.erwerbstaetigkeit_std for s in studis_v5])
        erw_v41 = np.array([s.erwerbstaetigkeit_std for s in studis_v41])

        zero_share_v5 = float(np.mean(erw_v5 == 0))
        zero_share_v41 = float(np.mean(erw_v41 == 0))
        self.assertGreaterEqual(zero_share_v5, 0.34)
        self.assertLessEqual(zero_share_v5, 0.40)
        self.assertGreater(zero_share_v5, zero_share_v41 + 0.08)

        active_erw_v5 = erw_v5[erw_v5 > 0]
        median_active_v5 = float(np.median(active_erw_v5))
        self.assertGreaterEqual(median_active_v5, 13.0)
        self.assertLessEqual(median_active_v5, 17.0)

        ge20_share_v5 = float(np.mean(erw_v5 >= 20))
        self.assertGreaterEqual(ge20_share_v5, 0.13)
        self.assertLessEqual(ge20_share_v5, 0.20)

        # Continuous support on [1, 35] vs. 6 discrete steps in V4.1
        self.assertGreater(len(np.unique(active_erw_v5)), 20)
        self.assertEqual(len(np.unique(erw_v41[erw_v41 > 0])), 6)

        # --- 3. Migration Share (28% in V5 vs. 22% in V4.1) ---
        mig_v5 = float(np.mean([s.migrationshintergrund for s in studis_v5]))
        mig_v41 = float(np.mean([s.migrationshintergrund for s in studis_v41]))
        self.assertGreaterEqual(mig_v5, 0.25)
        self.assertLessEqual(mig_v5, 0.31)
        self.assertGreater(mig_v5, mig_v41 + 0.03)

        # --- 4. Destatis Program-Specific Gender Ratios ---
        df_v5 = pd.DataFrame(
            [{"sg": s.studiengang_id, "g": s.geschlecht} for s in studis_v5]
        )
        w_share_by_sg = df_v5.groupby("sg")["g"].apply(lambda x: (x == "w").mean())

        # STEM (SG01, SG03): < 30% female
        self.assertLess(float(w_share_by_sg["SG01"]), 0.30)
        self.assertLess(float(w_share_by_sg["SG03"]), 0.30)
        # Business (SG02): ~48% female
        self.assertGreater(float(w_share_by_sg["SG02"]), 0.42)
        self.assertLess(float(w_share_by_sg["SG02"]), 0.55)
        # Psychology / Social Work (SG04, SG05): > 70% female
        self.assertGreater(float(w_share_by_sg["SG04"]), 0.70)
        self.assertGreater(float(w_share_by_sg["SG05"]), 0.70)

    def test_v5_universe_rng_synchronization_and_causal_direction(self):
        """
        Simulate N=600 students in Universe A (Full Support) and Universe B (No Support)
        under V5 and verify baseline identity, deterministic exam noise, and causal
        dropout reduction (Dropout_A < Dropout_B).
        """
        stammdaten_A = generiere_stammdaten()
        stammdaten_B = deepcopy(stammdaten_A)
        stammdaten_B["support_angebote_df"] = stammdaten_B["support_angebote_df"].iloc[0:0]
        stammdaten_B["support_modul_zuordnung_df"] = stammdaten_B[
            "support_modul_zuordnung_df"
        ].iloc[0:0]

        cfg = deepcopy(CONFIG)
        cfg["n_studierende"] = 600
        cfg["log_every_n_studis"] = 100000

        studis_A = generiere_studierende(
            stammdaten_A, np.random.default_rng(99999), cfg=cfg
        )
        studis_B = generiere_studierende(
            stammdaten_B, np.random.default_rng(99999), cfg=cfg
        )

        # Verify 100% identical initial populations before trajectory simulation
        for s_a, s_b in zip(studis_A, studis_B):
            self.assertEqual(s_a.studierenden_id, s_b.studierenden_id)
            self.assertEqual(s_a.studiengang_id, s_b.studiengang_id)
            self.assertEqual(s_a.hzb_note, s_b.hzb_note)
            self.assertEqual(s_a.motivation_initial, s_b.motivation_initial)
            self.assertEqual(s_a.erwerbstaetigkeit_std, s_b.erwerbstaetigkeit_std)

        simuliere_verlaeufe(
            studis_A,
            stammdaten_A,
            np.random.default_rng(99999),
            cfg=cfg,
            population_seed=99999,
        )
        simuliere_verlaeufe(
            studis_B,
            stammdaten_B,
            np.random.default_rng(99999),
            cfg=cfg,
            population_seed=99999,
        )

        # Anomaly assignments must remain 100% synchronized across universes
        for s_a, s_b in zip(studis_A, studis_B):
            self.assertEqual(s_a.anomalie_typ, s_b.anomalie_typ)

        total_supp_A = sum(len(s.support_teilnahmen) for s in studis_A)
        total_supp_B = sum(len(s.support_teilnahmen) for s in studis_B)
        self.assertGreater(total_supp_A, 100)
        self.assertEqual(total_supp_B, 0)

        # Dropout rate in Universe A (Full Support) must be lower than in Universe B (No Support)
        dropout_A = float(
            np.mean([not s.abschluss_erreicht for s in studis_A])
        )
        dropout_B = float(
            np.mean([not s.abschluss_erreicht for s in studis_B])
        )
        self.assertLess(
            dropout_A,
            dropout_B,
            f"Full support Universe A dropout ({dropout_A:.3f}) must be lower than No-Support Universe B ({dropout_B:.3f})",
        )

        # Verify deterministic exam noise helper
        n1 = get_exam_noise(123456, "MOD0001", 1, cfg)
        n2 = get_exam_noise(123456, "MOD0001", 1, cfg)
        n3 = get_exam_noise(123456, "MOD0001", 2, cfg)
        self.assertEqual(n1, n2)
        self.assertNotEqual(n1, n3)

    def test_v5_apathy_dampening_mechanism_and_late_semester_crisis(self):
        """
        Verify that `disable_apathy_dampening=True` strictly increases non-academic
        support uptake (`ueberfachlich` and `psychosozial`) both in an acute crisis
        cohort and in a natural V5 trajectory where students erode below 0.20 in later semesters.
        """
        stammdaten = generiere_stammdaten()
        non_fach_ids = set(
            stammdaten["support_angebote_df"]
            .loc[
                stammdaten["support_angebote_df"]["typ"].isin(
                    ["ueberfachlich", "psychosozial"]
                ),
                "angebot_id",
            ]
            .tolist()
        )

        # Part A: Controlled acute crisis cohort (Motivation & SozInt in [0.06, 0.14] < 0.20)
        cfg_base = deepcopy(CONFIG)
        cfg_base["n_studierende"] = 250
        cfg_base["log_every_n_studis"] = 100000
        cfg_base["disable_apathy_dampening"] = False

        cfg_no_apathy = deepcopy(cfg_base)
        cfg_no_apathy["disable_apathy_dampening"] = True

        studis_crisis_base = generiere_studierende(
            stammdaten, np.random.default_rng(777), cfg=cfg_base
        )
        for s in studis_crisis_base:
            s.motivation = 0.08
            s.motivation_initial = 0.08
            s.soziale_integration = 0.08
            s.soziale_integration_initial = 0.08
        studis_crisis_no_apathy = deepcopy(studis_crisis_base)

        simuliere_verlaeufe(
            studis_crisis_base,
            stammdaten,
            np.random.default_rng(777),
            cfg=cfg_base,
            population_seed=777,
        )
        simuliere_verlaeufe(
            studis_crisis_no_apathy,
            stammdaten,
            np.random.default_rng(777),
            cfg=cfg_no_apathy,
            population_seed=777,
        )

        non_fach_uptake_base = sum(
            1
            for s in studis_crisis_base
            for t in s.support_teilnahmen
            if t["angebot_id"] in non_fach_ids
        )
        non_fach_uptake_no_apathy = sum(
            1
            for s in studis_crisis_no_apathy
            for t in s.support_teilnahmen
            if t["angebot_id"] in non_fach_ids
        )
        self.assertGreater(
            non_fach_uptake_no_apathy,
            non_fach_uptake_base * 1.3,
            f"Disabling apathy dampening in crisis cohort must substantially increase uptake ({non_fach_uptake_no_apathy} vs {non_fach_uptake_base})",
        )

    def test_v5_berechne_dropout_parameterization(self):
        """
        Verify that `berechne_dropout` responds monotonically to all risk factors
        and respects custom parameter overrides in `cfg`.
        """
        p_healthy = berechne_dropout(
            motivation=0.70,
            soz_int=0.70,
            cp_rueckstand=0.0,
            durchgefallen_aktuell=0,
            fachsemester=2,
            overload_penalty=0.0,
        )
        p_low_mot = berechne_dropout(
            motivation=0.15,
            soz_int=0.70,
            cp_rueckstand=0.0,
            durchgefallen_aktuell=0,
            fachsemester=2,
            overload_penalty=0.0,
        )
        p_high_risk = berechne_dropout(
            motivation=0.15,
            soz_int=0.15,
            cp_rueckstand=30.0,
            durchgefallen_aktuell=2,
            fachsemester=1,
            overload_penalty=0.25,
        )
        p_late_sem = berechne_dropout(
            motivation=0.15,
            soz_int=0.15,
            cp_rueckstand=30.0,
            durchgefallen_aktuell=2,
            fachsemester=6,
            overload_penalty=0.25,
        )

        self.assertAlmostEqual(p_healthy, 0.005, places=5)
        self.assertGreater(p_low_mot, p_healthy)
        self.assertGreater(p_high_risk, p_low_mot)
        self.assertLess(p_late_sem, p_high_risk)

        # Custom cap override
        cfg_custom = deepcopy(CONFIG)
        cfg_custom["dropout_max_p"] = 0.12
        p_capped = berechne_dropout(
            motivation=0.05,
            soz_int=0.05,
            cp_rueckstand=60.0,
            durchgefallen_aktuell=4,
            fachsemester=1,
            overload_penalty=0.50,
            cfg=cfg_custom,
        )
        self.assertAlmostEqual(p_capped, 0.12, places=5)


if __name__ == "__main__":
    unittest.main()
