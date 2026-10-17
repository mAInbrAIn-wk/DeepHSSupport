"""
DeepSupport PyTorch & PyCox Survival Suite
==========================================
Enthält State-of-the-Art neuronale Survival-Architekturen:
1. PyTorchLogisticHazard: Exakte diskrete Intervall-Likelihood (Semester 1–16)
2. PyTorchDeepHit: Diskretes Ranking- + Likelihood-Modell (Lee et al., 2018)
3. PyTorchCoxPH: Extended DeepSurv mit Breslow Baseline Hazard Schätzung
4. PyTorchCoxTime: Nicht-proportionales Cox-Modell mit zeitvariierenden Effekten g(x, t)
5. PyTorchDeepHitCompetingRisks: Competing Risks Survival (Dropout vs. Abschluss)

Alle Modelle nutzen konsequent LayerNormalization (anstelle von BatchNorm),
um chargenunabhängige, stabile Inferenz ohne Batch-Order-Variance zu garantieren.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBackbone(nn.Module):
    """
    Konfigurierbarer MLP-Backbone mit LayerNormalization, GELU/ReLU und Dropout.
    Garantiert batchgrößenunabhängige Normalisierung.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        layers: List[nn.Module] = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            if activation.lower() == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.out_features = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PyTorchLogisticHazard(nn.Module):
    """
    Diskretes Intervall-Hazard-Modell (K=16 Zeitschritte für Bachelor-Semester 1 bis 16).
    
    Optimiert die exakte diskrete Bernoulli-Likelihood:
    h_k(x) = sigmoid(logits_k(x))
    S_k(x) = prod_{j=0}^k (1 - h_j(x))
    """
    def __init__(
        self,
        in_features: int,
        num_durations: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_durations = num_durations
        self.backbone = MLPBackbone(
            in_features=in_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation="gelu",
        )
        self.head = nn.Linear(self.backbone.out_features, num_durations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Liefert unbeschränkte Hazard-Logits (N, num_durations)."""
        feat = self.backbone(x)
        return self.head(feat)

    def predict_hazard(self, x: torch.Tensor) -> torch.Tensor:
        """Berechnet diskrete Hazard-Raten h_k(x) in (0, 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict_surv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die diskrete Überlebensfunktion S_k(x) = prod_{j=0}^k (1 - h_j(x)).
        Rückgabe: Tensor der Form (N, num_durations)
        """
        hazards = self.predict_hazard(x)
        surv = torch.cumprod(1.0 - hazards + 1e-7, dim=1)
        return surv

    def predict_step_hazard(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Gibt den momentanen Hazard h_t(x) für den Zeitschritt timestep zurück.
        Ermöglicht den direkten paritätischen Vergleich mit der gepoolten Keras-Regression.
        """
        hazards = self.predict_hazard(x)
        ts = torch.clamp(timestep, 0, self.num_durations - 1)
        return hazards.gather(1, ts.unsqueeze(1)).squeeze(1)

    def predict_risk(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gibt das relative Ausfallrisiko zurück (z. B. für ROC-AUC und C-Index).
        Falls timestep angegeben ist, wird die Ausfallwahrscheinlichkeit für diesen Zeitschritt genutzt,
        sonst das kumulative Risiko 1 - S(T_max).
        """
        surv = self.predict_surv(x)
        if timestep is not None:
            # timestep ist 0-basiert
            ts = torch.clamp(timestep, 0, self.num_durations - 1)
            risk = 1.0 - surv.gather(1, ts.unsqueeze(1)).squeeze(1)
        else:
            risk = 1.0 - surv[:, -1]
        return risk

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Berechnet den negativen Log-Likelihood-Verlust für diskrete Hazards.
        Unterstützt sowohl Panel-Intervalle (mit 't_start' in batch, sodass jede Zeile
        exakt auf ihrem Intervall [t_start, duration] evaluiert wird, ohne frühere Semester
        mehrfach zu gewichten) als auch statische Überlebensdaten (t_start = 0).
        """
        x = batch["x"]
        durations = batch["duration"]  # z. B. 1 bis 16
        events = batch["event"]        # 0 = censored, 1 = event

        logits = self.forward(x)  # (N, K)
        batch_size, num_bins = logits.shape

        # 0-basierter Index des Ziel-Intervalls (0 bis num_bins - 1)
        idx_dur = torch.clamp((durations - 1).long(), 0, num_bins - 1)
        if "t_start" in batch and batch["t_start"] is not None:
            idx_start = torch.clamp(batch["t_start"].long(), 0, num_bins - 1)
        else:
            idx_start = torch.zeros_like(idx_dur)

        # Erstelle Maske für aktive Intervalle [idx_start, idx_dur]
        # Bei Panel-Zeilen (t_start = k-1, duration = k) ist idx_start == idx_dur == k-1,
        # sodass jede Panel-Zeile exakt den bedingten Intervall-Hazard h_k(x_{i,k}) Beitrags-Loss liefert.
        grid = torch.arange(num_bins, device=x.device).unsqueeze(0)
        start_expanded = idx_start.unsqueeze(1)
        dur_expanded = idx_dur.unsqueeze(1)

        mask_active = (grid >= start_expanded) & (grid <= dur_expanded)

        # Targets: 0 für alle überlebten Intervalle (grid < dur_expanded),
        # und event (1 oder 0) genau im Ereignis-Intervall (grid == dur_expanded)
        targets = torch.zeros_like(logits)
        targets.scatter_(1, dur_expanded, events.unsqueeze(1))

        # Binary Cross Entropy with Logits auf den aktiven Intervallen
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (bce * mask_active.float()).sum() / torch.clamp(mask_active.float().sum(), min=1.0)
        return loss


class PyTorchDeepHit(nn.Module):
    """
    DeepHit Single-Event Modell (Lee et al., 2018).
    Kombiniert diskrete PMF-Log-Likelihood mit paarweisem Konkordanz-Ranking-Loss.
    """
    def __init__(
        self,
        in_features: int,
        num_durations: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        alpha_ranking: float = 0.5,
        sigma_ranking: float = 0.1,
    ):
        super().__init__()
        self.num_durations = num_durations
        self.alpha_ranking = alpha_ranking
        self.sigma_ranking = sigma_ranking

        self.backbone = MLPBackbone(
            in_features=in_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation="gelu",
        )
        # Softmax über num_durations diskrete Zeitschritte
        self.head = nn.Linear(self.backbone.out_features, num_durations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gibt PMF-Wahrscheinlichkeiten f_k(x) mit Softmax zurück (N, num_durations)."""
        feat = self.backbone(x)
        logits = self.head(feat)
        return F.softmax(logits, dim=1)

    def predict_surv(self, x: torch.Tensor) -> torch.Tensor:
        """Berechnet S_k(x) = 1 - sum_{j=0}^k f_j(x)."""
        pmf = self.forward(x)
        cif = torch.cumsum(pmf, dim=1)
        surv = torch.clamp(1.0 - cif, min=0.0, max=1.0)
        return surv

    def predict_risk(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        pmf = self.forward(x)
        cif = torch.cumsum(pmf, dim=1)
        if timestep is not None:
            ts = torch.clamp(timestep, 0, self.num_durations - 1)
            return cif.gather(1, ts.unsqueeze(1)).squeeze(1)
        return cif[:, -1]

    def predict_step_hazard(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Gibt den bedingten Intervall-Hazard h_t(x) = f_t(x) / S_{t-1}(x) zurück.
        """
        pmf = self.forward(x)
        cif = torch.cumsum(pmf, dim=1)
        surv = torch.clamp(1.0 - cif, min=0.0, max=1.0)
        ts = torch.clamp(timestep, 0, self.num_durations - 1)
        f_t = pmf.gather(1, ts.unsqueeze(1)).squeeze(1)
        s_t = surv.gather(1, ts.unsqueeze(1)).squeeze(1)
        return f_t / torch.clamp(s_t + f_t, min=1e-7)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        durations = batch["duration"]
        events = batch["event"]

        pmf = self.forward(x)  # (N, K)
        cif = torch.cumsum(pmf, dim=1)
        surv = torch.clamp(1.0 - cif, min=1e-7, max=1.0)

        idx_dur = torch.clamp((durations - 1).long(), 0, self.num_durations - 1)

        # 1. Log-Likelihood Verlust L1
        # Event: log(f(T))
        # Censored: log(S(T))
        p_event = pmf.gather(1, idx_dur.unsqueeze(1)).squeeze(1)
        p_surv = surv.gather(1, idx_dur.unsqueeze(1)).squeeze(1)

        log_lik = events * torch.log(p_event + 1e-7) + (1.0 - events) * torch.log(p_surv + 1e-7)
        loss_lik = -torch.mean(log_lik)

        # 2. Ranking Loss L2 (nur für Event-Paare mit T_i < T_j)
        # Wenn i ein Event bei T_i hatte und j bei T_i noch lebte, sollte F(T_i|x_i) > F(T_i|x_j) sein
        uncensored_mask = (events == 1.0)
        if uncensored_mask.sum() > 1 and self.alpha_ranking > 0.0:
            # Subsampling für Skalierbarkeit bei großen Batches
            max_pairs = min(500, len(durations))
            sub_idx = torch.randperm(len(durations))[:max_pairs]
            d_sub = durations[sub_idx]
            e_sub = events[sub_idx]
            cif_sub = cif[sub_idx]
            idx_sub = idx_dur[sub_idx]

            # Paarweise Matrix
            d_i = d_sub.unsqueeze(1)  # (M, 1)
            d_j = d_sub.unsqueeze(0)  # (1, M)
            e_i = e_sub.unsqueeze(1)  # (M, 1)

            # Relevante Paare: i hatte Event und T_i < T_j
            pair_mask = (e_i == 1.0) & (d_i < d_j)

            if pair_mask.sum() > 0:
                # F(T_i | x_i) und F(T_i | x_j)
                # CIF bei Zeitpunkt T_i:
                t_i_col = idx_sub.unsqueeze(1).expand(-1, max_pairs)  # (M, M)
                cif_i_at_ti = cif_sub.gather(1, idx_sub.unsqueeze(1)).expand(-1, max_pairs)
                cif_j_at_ti = cif_sub[torch.arange(max_pairs).unsqueeze(0).expand(max_pairs, -1), t_i_col]

                diff = cif_i_at_ti - cif_j_at_ti
                rank_penalty = torch.exp(-diff / self.sigma_ranking)
                loss_rank = (rank_penalty * pair_mask.float()).sum() / torch.clamp(pair_mask.float().sum(), min=1.0)
            else:
                loss_rank = torch.tensor(0.0, device=x.device)
        else:
            loss_rank = torch.tensor(0.0, device=x.device)

        return loss_lik + self.alpha_ranking * loss_rank


class PyTorchCoxPH(nn.Module):
    """
    Extended DeepSurv / Cox Proportional Hazards Netzwerk in reinem PyTorch.
    
    Verwendet negative Breslow Partial Likelihood mit vollständig
    vektorisierter Counting-Process-Summation über Risikosets R(t_i) = {j: t_start_j < t_i <= t_stop_j}
    und Schätzung der kumulativen Baseline-Hazard-Funktion H_0(t) nach Breslow.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = MLPBackbone(
            in_features=in_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation="gelu",
        )
        self.head = nn.Linear(self.backbone.out_features, 1)
        self.baseline_durations = None
        self.baseline_hazards = None
        self.cum_baseline_hazards = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Liefert skalaren Log-Risk r(x) = beta^T f(x)."""
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)

    def compute_baseline_hazard(
        self,
        x_train: Union[np.ndarray, torch.Tensor],
        durations_train: Union[np.ndarray, torch.Tensor],
        events_train: Union[np.ndarray, torch.Tensor],
        t_start_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Schätzt die kumulative Baseline-Hazard-Funktion H_0(t) nach Breslow
        unter Berücksichtigung des Counting-Process Risikosets (t_start < d <= duration):
        dh_0(t_i) = d_i / sum_{j in R(t_i)} exp(g(x_j))
        H_0(t) = sum_{t_i <= t} dh_0(t_i)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(x_train, np.ndarray):
                x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
            else:
                x_t = x_train.to(device)
            chunks = torch.split(x_t, 8192)
            log_risks = [self.forward(c).cpu().numpy() for c in chunks]
            risk = np.exp(np.clip(np.concatenate(log_risks), -20.0, 20.0))

        dur_np = durations_train if isinstance(durations_train, np.ndarray) else durations_train.cpu().numpy()
        ev_np = events_train if isinstance(events_train, np.ndarray) else events_train.cpu().numpy()
        if t_start_train is not None:
            t_start_np = t_start_train if isinstance(t_start_train, np.ndarray) else t_start_train.cpu().numpy()
        else:
            t_start_np = np.maximum(0.0, dur_np - 1.0)

        unique_durations = np.sort(np.unique(dur_np[ev_np == 1]))
        cum_haz = 0.0
        base_hazards = []
        cum_hazards = []

        for d in unique_durations:
            at_risk = (t_start_np < d) & (dur_np >= d)
            denom = np.sum(risk[at_risk])
            n_events = np.sum(ev_np[dur_np == d])
            dh0 = n_events / denom if denom > 0 else 0.0
            cum_haz += dh0
            base_hazards.append(dh0)
            cum_hazards.append(cum_haz)

        self.baseline_durations = unique_durations
        self.baseline_hazards = np.array(base_hazards)
        self.cum_baseline_hazards = np.array(cum_hazards)

    def predict_surv(
        self,
        x: Union[np.ndarray, torch.Tensor],
        durations: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Berechnet die Überlebenswahrscheinlichkeit S(t|x) = exp(-H_0(t) * exp(g(x))).
        Falls durations angegeben ist, wird S(t_i|x_i) berechnet, sonst für den maximalen Horizont.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_t = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x_t = x.to(device)
            chunks = torch.split(x_t, 8192)
            log_risks = [self.forward(c).cpu().numpy() for c in chunks]
            risk = np.exp(np.clip(np.concatenate(log_risks), -20.0, 20.0))

        if self.cum_baseline_hazards is None:
            raise RuntimeError("Baseline-Hazard wurde noch nicht geschätzt. Rufe zuerst compute_baseline_hazard() auf.")

        if durations is not None:
            dur_np = durations if isinstance(durations, np.ndarray) else durations.cpu().numpy()
            idx = np.searchsorted(self.baseline_durations, dur_np, side='right') - 1
            h0 = np.where(idx >= 0, self.cum_baseline_hazards[np.clip(idx, 0, len(self.cum_baseline_hazards) - 1)], 0.0)
        else:
            h0 = self.cum_baseline_hazards[-1]

        surv = np.exp(-h0 * risk)
        return surv

    def predict_step_hazard(
        self,
        x: Union[np.ndarray, torch.Tensor],
        durations: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Berechnet den bedingten Intervall-Hazard h_t(x) = 1 - exp(-dh_0(t) * exp(g(x)))
        für das spezifische Semester-Intervall `durations` (1-basiert oder 0-basiert).
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_t = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x_t = x.to(device)
            chunks = torch.split(x_t, 8192)
            log_risks = [self.forward(c).cpu().numpy() for c in chunks]
            risk = np.exp(np.clip(np.concatenate(log_risks), -20.0, 20.0))

        if self.baseline_hazards is None:
            return risk

        dur_np = durations if isinstance(durations, np.ndarray) else durations.cpu().numpy()
        # Falls 0-basierter step_test übergeben wurde (min == 0), auf 1-basiertes Fachsemester heben
        if len(dur_np) > 0 and np.min(dur_np) == 0:
            dur_np = dur_np + 1
        idx = np.searchsorted(self.baseline_durations, dur_np, side='right') - 1
        dh0 = np.where(idx >= 0, self.baseline_hazards[np.clip(idx, 0, len(self.baseline_hazards) - 1)], 0.0)
        return 1.0 - np.exp(-dh0 * risk)

    def predict_risk(
        self,
        x: Union[np.ndarray, torch.Tensor],
        durations: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Gibt die kalibrierte Ausfallwahrscheinlichkeit F(t|x) = 1 - S(t|x) zurück.
        Falls die Baseline-Hazard noch nicht geschätzt wurde, wird der unbeschränkte
        relative Risikoscore exp(g(x)) zurückgegeben.
        """
        if self.cum_baseline_hazards is not None:
            return 1.0 - self.predict_surv(x, durations)
        
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_t = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x_t = x.to(device)
            log_risk = self.forward(x_t)
            return torch.exp(log_risk).cpu().numpy()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Exakte Breslow Negative Partial Likelihood für Counting-Process Panel- und Survival-Daten.
        Vermeidet die Reihenfolge-Abhängigkeit von torch.cumsum bei Ties und nutzt
        R(t_i) = {j : t_start_j < t_stop_i <= t_stop_j}.
        """
        x = batch["x"]
        durations = batch["duration"]
        events = batch["event"]
        t_starts = batch.get("t_start", torch.zeros_like(durations))

        log_risk = self.forward(x)  # (B,)

        t_stop_i = durations.unsqueeze(1)   # (B, 1)
        t_start_j = t_starts.unsqueeze(0)   # (1, B)
        t_stop_j = durations.unsqueeze(0)   # (1, B)

        in_risk_set = (t_start_j < t_stop_i) & (t_stop_j >= t_stop_i)  # (B, B)

        risk_matrix = log_risk.unsqueeze(0).expand_as(in_risk_set)
        masked_risks = torch.where(in_risk_set, risk_matrix, torch.full_like(risk_matrix, -1e9))
        log_cum_risk = torch.logsumexp(masked_risks, dim=1)  # (B,)

        loss = -torch.sum((log_risk - log_cum_risk) * events)
        num_events = torch.sum(events) + 1e-7

        return loss / num_events


class PyTorchCoxTime(nn.Module):
    """
    CoxTime (Kvamme et al., 2019): Nicht-proportionales neuronales Cox-Modell.
    
    Verarbeitet Kovariaten x und skalierte Beobachtungszeit t (z. B. t / 16.0) als
    gemeinsame Eingabe [x, t] in einem Pre-LayerNorm MLP-Backbone, um zeitabhängige
    Interaktionen g(x, t) zu erfassen.
    
    Optimiert die exakte diskrete Breslow-Partial-Likelihood über alle Counting-Process Risikosets
    und schätzt die zeitvariierende Baseline-Hazard-Funktion:
    dh_0(t) = d(t) / sum_{j in R(t)} exp(g(x_j, t)).
    """
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        max_duration: float = 16.0,
    ):
        super().__init__()
        self.max_duration = max_duration
        self.backbone = MLPBackbone(
            in_features=in_features + 1,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation="gelu",
        )
        self.head = nn.Linear(self.backbone.out_features, 1)
        self.baseline_durations = None
        self.baseline_hazards = None
        self.cum_baseline_hazards = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        t: (B, 1) oder (B,)
        Liefert skalaren relativen Log-Hazard g(x, t).
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_norm = t / self.max_duration
        xt = torch.cat([x, t_norm], dim=1)
        feat = self.backbone(xt)
        return self.head(feat).squeeze(-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Breslow Negative Partial Likelihood für CoxTime mit Counting-Process Risikoset.
        """
        x = batch["x"]
        durations = batch["duration"]
        events = batch["event"]
        t_starts = batch.get("t_start", torch.zeros_like(durations))

        event_mask = (events == 1.0)
        if event_mask.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        event_durations = durations[event_mask]
        unique_durations = torch.unique(event_durations)

        total_loss = 0.0
        total_events = 0.0

        for d in unique_durations:
            cases_mask = (durations == d) & (events == 1.0)
            n_cases = cases_mask.sum()
            if n_cases == 0:
                continue

            risk_mask = (t_starts < d) & (durations >= d)
            x_risk = x[risk_mask]
            t_risk = torch.full((x_risk.size(0), 1), float(d), device=x.device)
            g_risk = self.forward(x_risk, t_risk)

            log_sum_exp_risk = torch.logsumexp(g_risk, dim=0)

            x_cases = x[cases_mask]
            t_cases = torch.full((x_cases.size(0), 1), float(d), device=x.device)
            g_cases = self.forward(x_cases, t_cases)

            step_loss = n_cases * log_sum_exp_risk - torch.sum(g_cases)
            total_loss = total_loss + step_loss
            total_events = total_events + n_cases.float()

        return total_loss / torch.clamp(total_events, min=1.0)

    def compute_baseline_hazard(
        self,
        x_train: Union[np.ndarray, torch.Tensor],
        durations_train: Union[np.ndarray, torch.Tensor],
        events_train: Union[np.ndarray, torch.Tensor],
        t_start_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Schätzt zeitvariierende Baseline-Hazard-Inkremente dh_0(t) nach Breslow
        unter Berücksichtigung des Counting-Process Risikosets (t_start < d <= duration):
        dh_0(t) = d(t) / sum_{j in R(t)} exp(g(x_j, t)).
        """
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(x_train, dtype=torch.float32, device=device) if isinstance(x_train, np.ndarray) else x_train.to(device)
        dur_np = durations_train if isinstance(durations_train, np.ndarray) else durations_train.cpu().numpy()
        ev_np = events_train if isinstance(events_train, np.ndarray) else events_train.cpu().numpy()
        if t_start_train is not None:
            t_start_np = t_start_train if isinstance(t_start_train, np.ndarray) else t_start_train.cpu().numpy()
        else:
            t_start_np = np.maximum(0.0, dur_np - 1.0)

        unique_durations = np.sort(np.unique(dur_np[ev_np == 1]))
        base_hazards = []
        cum_haz = 0.0
        cum_hazards = []

        with torch.no_grad():
            for d in unique_durations:
                at_risk = (t_start_np < d) & (dur_np >= d)
                n_events = np.sum(ev_np[dur_np == d])
                if n_events == 0:
                    base_hazards.append(0.0)
                    cum_hazards.append(cum_haz)
                    continue

                x_sub = x_t[at_risk]
                chunks = torch.split(x_sub, 8192)
                exp_g_sum = 0.0
                for c in chunks:
                    t_c = torch.full((c.size(0), 1), float(d), device=device)
                    g_c = self.forward(c, t_c)
                    exp_g_sum += torch.exp(g_c).sum().item()

                dh0 = n_events / exp_g_sum if exp_g_sum > 0 else 0.0
                cum_haz += dh0
                base_hazards.append(dh0)
                cum_hazards.append(cum_haz)

        self.baseline_durations = unique_durations
        self.baseline_hazards = np.array(base_hazards)
        self.cum_baseline_hazards = np.array(cum_hazards)

    def predict_surv(
        self,
        x: Union[np.ndarray, torch.Tensor],
        max_duration: Optional[int] = None,
    ) -> np.ndarray:
        """
        Berechnet zeitabhängige Überlebenskurven S(t|x) = exp(- sum_{u <= t} dh_0(u) exp(g(x, u))).
        Rückgabe: (N, len(baseline_durations))
        """
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device) if isinstance(x, np.ndarray) else x.to(device)
        n = len(x)
        num_d = len(self.baseline_durations)
        cum_h = np.zeros((n, num_d), dtype=np.float32)

        with torch.no_grad():
            running_h = np.zeros(n, dtype=np.float32)
            for idx, d in enumerate(self.baseline_durations):
                dh0 = self.baseline_hazards[idx]
                if dh0 > 0:
                    chunks = torch.split(x_t, 8192)
                    g_list = []
                    for c in chunks:
                        t_c = torch.full((c.size(0), 1), float(d), device=device)
                        g_list.append(self.forward(c, t_c).cpu().numpy())
                    g_d = np.concatenate(g_list)
                    running_h += dh0 * np.exp(g_d)
                cum_h[:, idx] = running_h

        surv = np.exp(-cum_h)
        return surv

    def predict_step_hazard(
        self,
        x: Union[np.ndarray, torch.Tensor],
        durations: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Berechnet den bedingten Intervall-Hazard h_t(x) = 1 - exp(-dh_0(t) * exp(g(x, t)))
        für das jeweilige Semester-Intervall `durations`.
        """
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device) if isinstance(x, np.ndarray) else x.to(device)
        dur_np = durations if isinstance(durations, np.ndarray) else durations.cpu().numpy()
        if len(dur_np) > 0 and np.min(dur_np) == 0:
            dur_np = dur_np + 1
        t_t = torch.tensor(dur_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            g_val = self.forward(x_t, t_t).cpu().numpy()
            risk = np.exp(np.clip(g_val, -20.0, 20.0))
        if self.baseline_hazards is None:
            return risk
        idx = np.searchsorted(self.baseline_durations, dur_np, side='right') - 1
        dh0 = np.where(idx >= 0, self.baseline_hazards[np.clip(idx, 0, len(self.baseline_hazards) - 1)], 0.0)
        return 1.0 - np.exp(-dh0 * risk)

    def predict_risk(
        self,
        x: Union[np.ndarray, torch.Tensor],
        timestep: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Berechnet die Ausfallwahrscheinlichkeit F(t|x) = 1 - S(t|x).
        """
        surv = self.predict_surv(x)
        if timestep is not None:
            ts_np = timestep if isinstance(timestep, np.ndarray) else timestep.cpu().numpy()
            ts = np.clip(ts_np, 0, surv.shape[1] - 1)
            risk = 1.0 - surv[np.arange(len(x)), ts]
        else:
            risk = 1.0 - surv[:, -1]
        return risk


class PyTorchDeepHitCompetingRisks(nn.Module):
    """
    DeepHit für Competing Risks (Lee et al., 2018).
    
    Modelliert die diskrete multivariate PMF f_k(t|x) über mehrere konkurrierende
    Ereignisse (z. B. k=1: Dropout, k=2: Abschluss) über K Zeitschritte (1..16 Semester).
    
    Kombiniert Multi-Task Likelihood mit paarweisem Cause-Specific Konkordanz-Ranking:
    Loss = (1 - alpha) * NLL + alpha * RankLoss(sigma).
    """
    def __init__(
        self,
        in_features: int,
        num_risks: int = 2,
        num_durations: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        alpha_ranking: float = 0.2,
        sigma_ranking: float = 0.1,
    ):
        super().__init__()
        self.num_risks = num_risks
        self.num_durations = num_durations
        self.alpha_ranking = alpha_ranking
        self.sigma_ranking = sigma_ranking

        self.backbone = MLPBackbone(
            in_features=in_features,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation="gelu",
        )
        self.head = nn.Linear(self.backbone.out_features, num_risks * num_durations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gibt die diskrete PMF f_k(t|x) mit Softmax zurück.
        Shape: (N, num_risks, num_durations)
        """
        feat = self.backbone(x)
        logits = self.head(feat)
        logits_padded = F.pad(logits, (0, 1), value=0.0)
        sm = F.softmax(logits_padded, dim=1)[:, :-1]
        pmf = sm.view(-1, self.num_risks, self.num_durations)
        return pmf

    def predict_pmf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def predict_cif(self, x: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die Cumulative Incidence Functions F_k(t|x) = sum_{u <= t} f_k(u|x).
        Shape: (N, num_risks, num_durations)
        """
        pmf = self.forward(x)
        cif = torch.cumsum(pmf, dim=2)
        return cif

    def predict_surv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Berechnet die Gesamtwahrscheinlichkeit, bisher KEINES der Ereignisse erlitten zu haben:
        S(t|x) = 1 - sum_{k} F_k(t|x).
        Shape: (N, num_durations)
        """
        cif = self.predict_cif(x)
        surv = torch.clamp(1.0 - cif.sum(dim=1), min=0.0, max=1.0)
        return surv

    def predict_risk(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        risk_idx: int = 0,
    ) -> torch.Tensor:
        """
        Gibt das kumulative Risiko für einen spezifischen Endpunkt zurück:
        risk_idx = 0: Dropout (k=1)
        risk_idx = 1: Abschluss (k=2)
        """
        cif = self.predict_cif(x)
        cif_k = cif[:, risk_idx, :]
        if timestep is not None:
            ts = torch.clamp(timestep, 0, self.num_durations - 1)
            return cif_k.gather(1, ts.unsqueeze(1)).squeeze(1)
        return cif_k[:, -1]

    def predict_step_hazard(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        risk_idx: int = 0,
    ) -> torch.Tensor:
        """
        Gibt den ursachenspezifischen bedingten Intervall-Hazard h_{k,t}(x) = f_{k,t}(x) / S_{t-1}(x) zurück.
        """
        pmf = self.forward(x)
        surv = self.predict_surv(x)
        ts = torch.clamp(timestep, 0, self.num_durations - 1)
        f_kt = pmf[:, risk_idx, :].gather(1, ts.unsqueeze(1)).squeeze(1)
        f_all_t = pmf.sum(dim=1).gather(1, ts.unsqueeze(1)).squeeze(1)
        s_t = surv.gather(1, ts.unsqueeze(1)).squeeze(1)
        return f_kt / torch.clamp(s_t + f_all_t, min=1e-7)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Multi-Task Likelihood + Ranking Loss für Competing Risks.
        """
        x = batch["x"]
        durations = batch["duration"]
        comp_events = batch.get("competing_event", batch["event"].long())

        pmf = self.forward(x)
        cif = torch.cumsum(pmf, dim=2)
        surv = torch.clamp(1.0 - cif.sum(dim=1), min=1e-7, max=1.0)

        idx_dur = torch.clamp((durations - 1).long(), 0, self.num_durations - 1)
        batch_size = x.size(0)

        loss_lik = torch.zeros(batch_size, device=x.device)

        censored_mask = (comp_events == 0)
        if censored_mask.sum() > 0:
            p_surv = surv[censored_mask, idx_dur[censored_mask]]
            loss_lik[censored_mask] = -torch.log(p_surv + 1e-7)

        for k in range(1, self.num_risks + 1):
            risk_mask = (comp_events == k)
            if risk_mask.sum() > 0:
                p_event = pmf[risk_mask, k - 1, idx_dur[risk_mask]]
                loss_lik[risk_mask] = -torch.log(p_event + 1e-7)

        loss_l1 = torch.mean(loss_lik)

        loss_rank = torch.tensor(0.0, device=x.device)
        if self.alpha_ranking > 0.0:
            max_pairs = min(500, batch_size)
            sub_idx = torch.randperm(batch_size)[:max_pairs]
            d_sub = durations[sub_idx]
            ce_sub = comp_events[sub_idx]
            cif_sub = cif[sub_idx]
            idx_sub = idx_dur[sub_idx]

            d_i = d_sub.unsqueeze(1)
            d_j = d_sub.unsqueeze(0)

            for k in range(1, self.num_risks + 1):
                e_i_k = (ce_sub == k).unsqueeze(1)
                pair_mask = e_i_k & (d_i < d_j)
                if pair_mask.sum() > 0:
                    cif_k = cif_sub[:, k - 1, :]
                    cif_i = cif_k.gather(1, idx_sub.unsqueeze(1)).expand(-1, max_pairs)
                    t_i_idx = idx_sub.unsqueeze(1).expand(-1, max_pairs)
                    cif_j = cif_k[torch.arange(max_pairs).unsqueeze(0).expand(max_pairs, -1), t_i_idx]

                    diff = cif_i - cif_j
                    rank_penalty = torch.exp(-diff / self.sigma_ranking)
                    loss_rank = loss_rank + (rank_penalty * pair_mask.float()).sum() / torch.clamp(pair_mask.float().sum(), min=1.0)

        return (1.0 - self.alpha_ranking) * loss_l1 + self.alpha_ranking * loss_rank
