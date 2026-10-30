import os
import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.io import read


class TrajectoryLoader:
    def __init__(self, filepath, cell=None, pbc=None):
        self.filepath = filepath
        self.cell = cell
        self.pbc = pbc

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"t", "true", "1", "yes", "y", "on"}:
            return True
        if s in {"f", "false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse PBC value: {value}")

    def _parse_pbc(self):
        if self.pbc is None:
            return None
        if isinstance(self.pbc, (bool, int, float)):
            b = self._as_bool(self.pbc)
            return [b, b, b]
        if isinstance(self.pbc, (list, tuple, np.ndarray)):
            vals = list(self.pbc)
            if len(vals) == 1:
                b = self._as_bool(vals[0])
                return [b, b, b]
            if len(vals) == 3:
                return [self._as_bool(x) for x in vals]
            raise ValueError("pbc must contain 1 or 3 values")

        s = str(self.pbc).strip()
        if not s or s.lower() == "null":
            return None
        compact = s.replace(" ", "").replace(",", "").upper()
        if compact in {"T", "TRUE", "1", "Y", "YES"}:
            return [True, True, True]
        if compact in {"F", "FALSE", "0", "N", "NO"}:
            return [False, False, False]
        if len(compact) == 3 and all(c in "TF10" for c in compact):
            return [c in "T1" for c in compact]
        parts = s.replace(",", " ").split()
        if len(parts) == 3:
            return [self._as_bool(x) for x in parts]
        raise ValueError(f"Cannot parse pbc='{self.pbc}'. Use e.g. 'T T T' or 'T T F'.")

    def _parse_numeric_cell(self, value):
        parts = str(value).replace(",", " ").split()
        nums = [float(x) for x in parts]
        if len(nums) == 3:
            return np.diag(nums)
        if len(nums) == 6:
            return Cell.fromcellpar(nums).array
        if len(nums) == 9:
            return np.array(nums, dtype=float).reshape(3, 3)
        raise ValueError(
            "cell must be 3, 6, or 9 numbers, or an AMS RKF source such as 'ams.rkf'/'auto'."
        )

    def _cell_to_3x3(self, cell):
        if cell is None or len(cell) == 0:
            return np.zeros((3, 3)), [False, False, False]
        arr = np.array(cell, dtype=float).reshape(-1, 3)
        nvec = min(len(arr), 3)
        box = np.zeros((3, 3))
        box[:nvec, :] = arr[:nvec, :]
        return box, [True] * nvec + [False] * (3 - nvec)

    def _has_valid_periodic_cell(self, atoms):
        try:
            pbc = np.asarray(atoms.get_pbc(), dtype=bool)
            cell = np.asarray(atoms.get_cell().array, dtype=float)
            if not np.any(pbc) or cell.shape != (3, 3):
                return False
            periodic_vectors = cell[pbc]
            return (
                np.all(np.isfinite(periodic_vectors))
                and np.all(np.linalg.norm(periodic_vectors, axis=1) > 1.0e-8)
                and np.linalg.matrix_rank(periodic_vectors, tol=1.0e-8)
                == len(periodic_vectors)
            )
        except Exception:
            return False

    def _get_rkf_class(self):
        try:
            from scm.plams import RKFTrajectoryFile
            return RKFTrajectoryFile
        except Exception as exc:
            raise ImportError(
                "AMS RKF support requires AMS/PLAMS. Run with $AMSBIN/amspython "
                "or load the AMS environment first."
            ) from exc

    def _load_rkf(self, filepath):
        RKFTrajectoryFile = self._get_rkf_class()
        rkf = RKFTrajectoryFile(filepath)
        elements = rkf.get_elements()
        traj = []
        for iframe in range(rkf.get_length()):
            coords, cell = rkf.read_frame(iframe)
            box, pbc = self._cell_to_3x3(cell)
            atoms = Atoms(
                symbols=elements,
                positions=np.asarray(coords, dtype=float).reshape(-1, 3),
                cell=box,
                pbc=pbc,
            )
            atoms.info["Frame"] = int(iframe)
            atoms.info["SourceFormat"] = "ams.rkf"
            traj.append(atoms)
        rkf.close()
        return traj

    def _read_rkf_cells(self, filepath):
        RKFTrajectoryFile = self._get_rkf_class()
        rkf = RKFTrajectoryFile(filepath)
        cells, pbcs = [], []
        for iframe in range(rkf.get_length()):
            _coords, cell = rkf.read_frame(iframe)
            box, pbc = self._cell_to_3x3(cell)
            cells.append(box)
            pbcs.append(pbc)
        rkf.close()
        return cells, pbcs

    def _resolve_cell_source(self):
        if self.cell is None:
            return None
        s = str(self.cell).strip()
        if s.lower() == "auto":
            path = os.path.join(os.path.dirname(os.path.abspath(self.filepath)), "ams.rkf")
            if not os.path.exists(path):
                raise FileNotFoundError("cell: auto requested, but no ams.rkf was found next to the trajectory")
            return path
        if s.lower().endswith(".rkf"):
            return s
        return None

    def _apply_cell_pbc(self, traj):
        pbc = self._parse_pbc()
        cell_source = self._resolve_cell_source()

        if cell_source is not None:
            if not os.path.exists(cell_source):
                raise FileNotFoundError(f"Cell source not found: {cell_source}")
            print(f"📦 Reading cell from: {cell_source}")
            cells, pbcs = self._read_rkf_cells(cell_source)
            if len(cells) not in {1, len(traj)}:
                raise ValueError(f"Frame mismatch: trajectory={len(traj)}, cell source={len(cells)}")
            for i, atoms in enumerate(traj):
                k = i if len(cells) > 1 else 0
                atoms.set_cell(cells[k])
                atoms.set_pbc(pbc if pbc is not None else pbcs[k])
                if any(atoms.get_pbc()) and not self._has_valid_periodic_cell(atoms):
                    raise ValueError(
                        f"Invalid periodic cell at frame {i} from {cell_source}"
                    )
            return traj

        if self.cell is not None:
            cell = self._parse_numeric_cell(self.cell)
            if pbc is None:
                pbc = [True, True, True]
            for i, atoms in enumerate(traj):
                atoms.set_cell(cell)
                atoms.set_pbc(pbc)
                if any(atoms.get_pbc()) and not self._has_valid_periodic_cell(atoms):
                    raise ValueError(
                        f"Invalid cell/PBC combination at frame {i}"
                    )
            return traj

        if pbc is not None:
            for atoms in traj:
                atoms.set_pbc(pbc)
                if any(pbc) and not self._has_valid_periodic_cell(atoms):
                    raise ValueError("pbc=True was requested, but no valid cell is available. Please also set cell.")
        return traj

    def load(self, remove_species=None):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"{self.filepath} not found")
        print(f"📂 Loading trajectory: {self.filepath} ...")

        if self.filepath.lower().endswith(".rkf"):
            traj = self._load_rkf(self.filepath)
            # Apply the same explicit cell/PBC policy as every other input
            # format.  Previously numeric/external cell overrides were silently
            # ignored for RKF trajectories.
            traj = self._apply_cell_pbc(traj)
        else:
            traj = read(self.filepath, index=':')
            traj = self._apply_cell_pbc(traj)

        if remove_species:
            print(f"🧹 Removing elements: {remove_species}")
            new_traj = []
            for atoms in traj:
                del atoms[[a.index for a in atoms if a.symbol in remove_species]]
                new_traj.append(atoms)
            traj = new_traj

        if not traj:
            raise RuntimeError("No frames were loaded from the trajectory")
        if len(traj[0]) == 0:
            raise ValueError(
                "The loaded trajectory contains zero atoms after element removal"
            )

        for frame_index, atoms in enumerate(traj):
            if any(atoms.get_pbc()) and not self._has_valid_periodic_cell(atoms):
                raise ValueError(
                    f"Frame {frame_index} has periodic boundary flags but no "
                    "valid linearly independent periodic cell vectors"
                )

        # Atom-index lineage is the central invariant of the reaction detector.
        # Fail before topology analysis if a variable-composition trajectory
        # would make an index refer to different atoms in different frames.
        reference_symbols = traj[0].get_chemical_symbols()
        for frame_index, atoms in enumerate(traj[1:], 1):
            symbols = atoms.get_chemical_symbols()
            if symbols != reference_symbols:
                raise ValueError(
                    "Trajectory atom identity/order changes at frame "
                    f"{frame_index}. RxnScape requires a fixed atom count and "
                    "element order for atom-mapped reaction lineage."
                )

        print(f"✅ Loaded: {len(traj)} frames, {len(traj[0])} atoms/frame")
        if self._has_valid_periodic_cell(traj[0]):
            print(f"📦 PBC-aware topology: cell={traj[0].cell.lengths()}, pbc={traj[0].pbc}")
        else:
            print("⚠️ Non-periodic topology: no valid cell+pbc detected")
        return traj
