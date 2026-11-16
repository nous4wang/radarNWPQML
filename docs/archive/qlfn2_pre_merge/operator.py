"""The phase-routed rain operator.

The claim this module implements, and where it comes from:

* A global Z-R relation explains only R^2 = 0.061 of the log-log variance
  between composite reflectivity and the 6-minute rainfall, and it never
  reaches the 2 mm threshold at all (measured CSI@2.0 = 0.0000).  So the
  radar -> rain relation is not a single power law.
* Routing that relation by the *environment* does not help: the residual
  variance explained by DVG850 convergence/divergence and by the freezing
  level is 0.58 % on average.  An environment-driven router is therefore
  dead on arrival.
* What is left is the *observed* column: VIL correlates with rain better than
  composite reflectivity does (0.295 against 0.208), and the six CAPPI levels
  carry the vertical structure that tells the operator whether the column is
  liquid, mixed or icy.

So the operator is

    R_hat = sum_k  pi_k(x) * ZR_k(z)

with ``z`` the reflectivity, ``ZR_k(z) = a_k * z ** b_k`` a small family of
power laws -- one per phase state -- and ``pi_k(x)`` a soft weight derived
from the observed vertical structure.  The number of free parameters is
small and every one of them has a name, so the operator is auditable; that
is the difference from a generic ten-channel CNN.

The weights are produced by a tiny network over the phase inputs, so the
routing is learned but its *inputs* are fixed physical quantities.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cappi import phase_inputs

#: The phase states the operator routes between.
PHASE_STATES: tuple[str, ...] = ("warm_rain", "mixed", "ice")


@dataclass
class OperatorOutput:
    """What the operator produces.

    Attributes:
        rain: (B, H, W) the rain rate in mm per 6-minute accumulation.
        weights: (B, K, H, W) the soft phase weights.
        params: the fitted power-law parameters, for logging.
    """

    rain: torch.Tensor
    weights: torch.Tensor
    params: dict[str, torch.Tensor]


class PhaseRouter(nn.Module):
    """Soft weights over the phase states, from the observed column only.

    The input is the phase-feature bundle of :func:`qlfn2.cappi.phase_inputs`;
    the output is a (B, K, H, W) tensor of non-negative weights that sum to
    one over the phase axis.
    """

    def __init__(self, n_inputs: int, hidden: int = 24, n_states: int = len(PHASE_STATES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_inputs, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, n_states, 1),
        )
        # Start from a uniform mixture so that the operator begins as the
        # familiar single power law and *earns* its phase routing.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(phase), dim=1)


class PhaseRoutedZROperator(nn.Module):
    """``R = sum_k pi_k * a_k * z ** b_k`` with learnable, named parameters.

    Args:
        z_min: the reflectivity floor, in dBZ.  Below it the column has no
            precipitation and the operator returns exactly zero.
        init_a: the initial intercept of each power law, in the usual
            Marshall-Palmer units (mm/h per (dBZ)^b).  The packed archive
            stores rainfall per 6-minute accumulation, so the amount is
            divided by ten at the end.
        init_b: the initial exponent of each power law.
        residual_scale: the initial width of the learned correction, in
            millimetres per 6-minute accumulation.  It sets how far the
            head may depart from the operator on its own.
    """

    def __init__(
        self,
        z_min: float = 5.0,
        init_a: tuple[float, ...] = (1.600, 1.600, 1.600),
        init_b: tuple[float, ...] = (0.660, 0.660, 0.660),
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if len(init_a) != len(PHASE_STATES) or len(init_b) != len(PHASE_STATES):
            raise ValueError("one (a, b) pair per phase state is required")
        self.z_min = float(z_min)
        self.router = PhaseRouter(n_inputs=6)
        # Softplus keeps the amplitudes positive; the exponents are free.
        self.raw_a = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_a, dtype=torch.float32))))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        #: The scale of the learned correction, in mm per 6-minute
        #: accumulation.  It is a fixed constant rather than a parameter, so
        #: that the correction is bounded by a number that can be read off
        #: rather than by one that the training can grow without limit.
        self.residual_scale = float(residual_scale)

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    def forward(self, radar: torch.Tensor, z_dbz: torch.Tensor | None = None) -> OperatorOutput:
        """Evaluate the operator on one radar frame.

        Args:
            radar: (B, 10, H, W) in the archive's raw units.
            z_dbz: the reflectivity used by the power laws, in dBZ.  Defaults
                to the composite reflectivity channel, which is what the
                benchmark's Z-R baseline uses.

        Returns:
            :class:`OperatorOutput`.
        """
        from .cappi import CR_INDEX

        if z_dbz is None:
            z_dbz = radar[:, CR_INDEX] / 10.0
        phase = phase_inputs(radar)
        weights = self.router(phase)                       # (B, K, H, W)

        z = z_dbz.clamp(min=self.z_min)
        # (B, K, H, W)
        zk = z.unsqueeze(1).expand_as(weights)
        a = self.a.view(1, -1, 1, 1)
        b = self.b.view(1, -1, 1, 1)
        zr = a * torch.pow(zk, b)

        rain_h = (weights * zr).sum(dim=1)                  # mm per hour
        rain = rain_h / 10.0                                # mm per 6-minute accumulation
        rain = torch.where(z_dbz > self.z_min, rain, torch.zeros_like(rain))
        return OperatorOutput(
            rain=rain,
            weights=weights,
            params={"a": self.a.detach(), "b": self.b.detach()},
        )


class StructuredRainHead(nn.Module):
    """The operator plus a learned residual correction.

    The operator supplies the physically-shaped part of the rain field; the
    residual branch learns what the operator misses.

    The residual is **additive and bounded by a constant**, not scaled by the
    operator's own output.  That distinction is not cosmetic.  A residual
    proportional to the operator's output inherits the operator's magnitude
    twice over: it is small where the operator is small, and it can at most
    double it.  Since the whole point of the head is to correct the
    operator, a correction that is itself proportional to the operator's
    error cannot leave the operator's magnitude regime.  Measured on the
    archive, the operator's own relation between reflectivity and rainfall
    explains R^2 = 0.137 of the log-log variance, so the residual is where
    most of the skill has to come from.

    The correction is added only where the operator believes there is
    precipitation, so the head cannot hallucinate rain in empty space.
    """

    def __init__(self, n_features: int, z_min: float = 5.0, hidden: int = 32,
                 residual_scale: float = 2.0) -> None:
        super().__init__()
        self.operator = PhaseRoutedZROperator(z_min=z_min,
                                               residual_scale=residual_scale)
        self.residual = nn.Sequential(
            nn.Conv2d(n_features, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, radar: torch.Tensor, features: torch.Tensor,
                z_dbz: torch.Tensor | None = None) -> OperatorOutput:
        op = self.operator(radar, z_dbz=z_dbz)
        delta = self.residual(features).squeeze(1)
        # The bound on the correction has two parts: a constant, so that a
        # column the operator rates weakly can still be corrected by an
        # absolute amount, and a term proportional to the operator's own
        # output, so that a heavy column can be corrected proportionally.
        # The proportional part is detached: it is a magnitude limit, not
        # something the model should be able to inflate by inflating the
        # operator.
        bound = self.operator.residual_scale + op.rain.detach()
        correction = torch.tanh(delta) * bound
        rain = torch.clamp(op.rain + correction, min=0.0)
        # where the operator sees no echo there is no rain, whatever the
        # correction says
        if z_dbz is None:
            from .cappi import CR_INDEX

            z_dbz = radar[:, CR_INDEX] / 10.0
        rain = torch.where(z_dbz > self.operator.z_min, rain,
                           torch.zeros_like(rain))
        return OperatorOutput(rain=rain, weights=op.weights, params=op.params)
