from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from smithlab import lammps, trajectory, regression

def distributions():
    base_dir = Path(__file__).resolve().parent

    bond_file = base_dir / "bonds.dump"
    angle_file = base_dir / "angles.dump"
    dihedral_file = base_dir / "dihedrals.dump"
    types = [3, 10, 8]
    # bond type 3 = c5,c5
    # angle type 10 = ca,c5,ca
    # dihedral type 8 = c5,ca,ca,ca

    for type, dist_path in zip(types, [bond_file, angle_file, dihedral_file]):
        lammps.parse_local_dump(filename=dist_path,
                                start_timestep=7000000,
                                selected_types=[type],
                                outfile_prefix=base_dir / f"{dist_path.stem}"
                                )
        


def c2():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "equilibrated_ua.lmps"
    dump_file = base_dir / "compress.lammpstrj"
    timestep_fs = 0.1
    max_lag_ps = 2000
    stride = 1
    backbone_selection = "type 2 3" # ca-ca, ca-c5, and c5-c5
    origin_stride_ps = 10
    output_file = base_dir / "c2_analysis.csv"

    trajectory.c2_bondvec(data_file = str(data_file),
                          dump_file = str(dump_file),
                          timestep_fs=timestep_fs,
                          max_lag_ps=max_lag_ps,
                          stride=stride,
                          backbone_selection=backbone_selection,
                          origin_stride_ps=origin_stride_ps,
                          output_file=output_file)


def fs():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "equilibrated_ua.lmps"
    dump_file = base_dir / "compress.lammpstrj"
    timestep_fs = 0.1
    max_lag_ps = 2000
    stride = 1
    backbone_selection = "type 2 3" # ca-ca, ca-c5, and c5-c5
    origin_stride_ps = 10
    output_file = base_dir / "fs_analysis.csv"
    output_file_sq = base_dir / "sq_analysis.csv"

    q, Sq, q_star = trajectory.sq(data_file=str(data_file),
                                  dump_file=str(dump_file),
                                  stride=stride,
                                  num_proc=48,
                                  output_file=str(output_file_sq),
                                   )
    if np.isnan(q_star):
        q_Ainv = 1
    else:
        q_Ainv = q_star

    trajectory.fs_self(data_file = str(data_file),
                          dump_file = str(dump_file),
                          q_Ainv=q_Ainv,
                          timestep_fs=timestep_fs,
                          max_lag_ps=max_lag_ps,
                          stride=stride,
                          selection=backbone_selection,
                          origin_stride_ps=origin_stride_ps,
                          output_file=output_file)


def c2_all():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "equilibrated_ua.lmps"
    dump_file = base_dir / "compress.lammpstrj"
    timestep_fs = 0.1
    max_lag_ps = 2000
    stride = 1
    backbone_selection = None
    origin_stride_ps = 10
    output_file = base_dir / "c2_analysis_all.csv"

    trajectory.c2_bondvec(data_file = str(data_file),
                          dump_file = str(dump_file),
                          timestep_fs=timestep_fs,
                          max_lag_ps=max_lag_ps,
                          stride=stride,
                          backbone_selection=backbone_selection,
                          origin_stride_ps=origin_stride_ps,
                          output_file=output_file)


def fs_all():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "equilibrated_ua.lmps"
    dump_file = base_dir / "compress.lammpstrj"
    timestep_fs = 0.1
    max_lag_ps = 2000
    stride = 1
    backbone_selection = None
    origin_stride_ps = 10
    output_file = base_dir / "fs_analysis_all.csv"
    output_file_sq = base_dir / "sq_analysis_all.csv"

    q, Sq, q_star = trajectory.sq(data_file=str(data_file),
                                  dump_file=str(dump_file),
                                  stride=stride,
                                  num_proc=48,
                                  output_file=str(output_file_sq),
                                   )
    if np.isnan(q_star):
        q_Ainv = 1
    else:
        q_Ainv = q_star

    trajectory.fs_self(data_file = str(data_file),
                          dump_file = str(dump_file),
                          q_Ainv=q_Ainv,
                          timestep_fs=timestep_fs,
                          max_lag_ps=max_lag_ps,
                          stride=stride,
                          selection=backbone_selection,
                          origin_stride_ps=origin_stride_ps,
                          output_file=output_file)



def kww():
    base_dir = Path(__file__).resolve().parent
    for file in ["c2_analysis.csv", "c2_analysis_all.csv"]:
        output_file = base_dir / file
        time, c2, _ = np.genfromtxt(output_file, delimiter=",", skip_header=1, unpack=True)
        c2_normalize = regression.normalize_decay(time, c2, plateau_frac=0.01)
        time_lin, c2_lin = regression.linearized_kww(time, c2_normalize, cut_head=0.3, cut_tail = 0.3)
        (beta, intercept), cov = np.polyfit(time_lin, c2_lin, 1, cov=True)
        beta_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])
        cov_beta_intercept = cov[0, 1]
        tau = np.exp(-intercept / beta)
        # error propagation
        d_tau_d_beta = tau * intercept / beta**2
        d_tau_d_intercept = -tau / beta
        tau_var = (
            d_tau_d_beta**2 * cov[0, 0]
            + d_tau_d_intercept**2 * cov[1, 1]
            + 2 * d_tau_d_beta * d_tau_d_intercept * cov[0, 1]
        )
        tau_err = np.sqrt(tau_var)
        print(f"beta = {beta:.4f} ± {beta_err:.4f}")
        print(f"tau  = {tau:.4e} ± {tau_err:.4e}")
        plt.plot(time_lin, c2_lin)
        plt.show()
        
def kww_fs():
    print("====fs======")
    base_dir = Path(__file__).resolve().parent
    for file in ["fs_analysis.csv", "fs_analysis_all.csv"]:
        output_file = base_dir / file
        time, c2, _ = np.genfromtxt(output_file, delimiter=",", skip_header=1, unpack=True)
        c2_normalize = regression.normalize_decay(time, c2, plateau_frac=0.01)
        time_lin, c2_lin = regression.linearized_kww(time, c2_normalize, cut_head=0.3, cut_tail = 0.3)
        (beta, intercept), cov = np.polyfit(time_lin, c2_lin, 1, cov=True)
        beta_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])
        cov_beta_intercept = cov[0, 1]
        tau = np.exp(-intercept / beta)
        # error propagation
        d_tau_d_beta = tau * intercept / beta**2
        d_tau_d_intercept = -tau / beta
        tau_var = (
            d_tau_d_beta**2 * cov[0, 0]
            + d_tau_d_intercept**2 * cov[1, 1]
            + 2 * d_tau_d_beta * d_tau_d_intercept * cov[0, 1]
        )
        tau_err = np.sqrt(tau_var)
        print(f"beta = {beta:.4f} ± {beta_err:.4f}")
        print(f"tau  = {tau:.4e} ± {tau_err:.4e}")
        plt.plot(time_lin, c2_lin)
        plt.show()


#distributions()
#c2()
#fs()
#c2_all()
#fs_all()
#kww()
kww_fs()



