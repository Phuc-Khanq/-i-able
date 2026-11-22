"""S047b test helper: write simulator sessions to a temp store (it may import sim_*)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iable_eye import session_store as store  # noqa: E402
from iable_eye.paths import ROOT  # noqa: E402
from iable_eye.sim_protocol import PROTOCOL_DURATION_S  # noqa: E402
from iable_eye.sim_session import simulate_session  # noqa: E402
from tools.run_gaze_baseline import BASE_SEED  # noqa: E402


def write_sim_sessions(n, data_root, duration_s=PROTOCOL_DURATION_S, person_code="S001", root=ROOT) -> list[Path]:
    """simulate_session(k, BASE_SEED, person_code, duration_s) -> write_session, for k = 1..n."""
    paths = []
    for k in range(1, n + 1):
        s = simulate_session(k, BASE_SEED, person_code=person_code, duration_s=duration_s)
        paths.append(store.write_session(s.arrays, s.meta, s.channels, data_root=data_root, root=root))
    return paths
