from pathlib import Path
from smithlab import lammps


base_dir = Path(__file__).resolve().parent
lammps.setup_lammps(base_dir / "equilibrated.lmps", base_dir / "equilibrated_out.lmps", base_dir / "data.lmps")