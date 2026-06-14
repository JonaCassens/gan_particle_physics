import glob
import uproot
import numpy as np
import awkward as ak
import pandas as pd
import vector
import pyarrow.parquet as pq
from typing import Optional, cast

vector.register_awkward()

# Path to the COMET Phase-I geometry file (used for detector-origin centering)
_GEOMETRY_FILE = "/vols/comet/data/phaseIgeom.root"
_GEOMETRY_KEY  = "COMETGeometry-991d653d-29fe3ca5-487c39de-61ad2e64-d67e8aca"

# Volume path whose global translation gives the CDC detector origin
# CylindricalDetector is the central CDC+CTH assembly at (674.0, 0.0, 765.0) mm
_CDC_PATH = "/comet_1/DetectorSolenoid_0/CylindricalDetector_0"

# Verified midstream monitor-plane origin (MonitorID=4) from geometry checks.
# x=3259 mm is the monitor plane location, y=0 is detector axis, z from validated dataset center.
_MIDSTREAM_MONITOR4_ORIGIN = (3259.0, 0.0, 7655.529)

# Module-level cache so the file is only opened once per process
_detector_origin_cache = None


def load_detector_origin() -> tuple[float, float, float]:
    """Return the CDC detector origin (x0, y0, z0) in mm from the geometry file.

    Uses ROOT's TGeoManager via subprocess so that PyROOT is not required in the
    Python environment.  The result is cached after the first call.

    Returns
    -------
    (x0, y0, z0) : floats, mm, in the COMET world-frame convention
        x0 ~ 674 mm  (along beam / detector axis)
        y0 ~   0 mm  (vertical)
        z0 ~ 765 mm  (horizontal transverse)
    """
    global _detector_origin_cache
    if _detector_origin_cache is not None:
        return _detector_origin_cache

    import os
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".C", delete=False) as tmp:
            macro_path = tmp.name
            macro_func = os.path.splitext(os.path.basename(macro_path))[0]
            macro = rf"""
void {macro_func}() {{
    TFile *f = TFile::Open("{_GEOMETRY_FILE}");
    if (!f || f->IsZombie()) {{ printf("GEOM_ERR:cannot_open\n"); return; }}
    TGeoManager *g = (TGeoManager*)f->Get("{_GEOMETRY_KEY}");
    if (!g) {{ printf("GEOM_ERR:no_geomanager\n"); return; }}
    if (!g->cd("{_CDC_PATH}")) {{ printf("GEOM_ERR:path_not_found\n"); return; }}
    const Double_t *t = g->GetCurrentMatrix()->GetTranslation();
    printf("GEOM_OK:%.6f:%.6f:%.6f\n", t[0], t[1], t[2]);
}}
"""
            tmp.write(macro)

        result = subprocess.run(
            ["root", "-b", "-q", f"{macro_path}()"],
            capture_output=True, text=True, timeout=60
        )

        for line in result.stdout.splitlines():
            if line.startswith("GEOM_OK:"):
                parts = line.split(":")[1:]
                x0, y0, z0 = float(parts[0]), float(parts[1]), float(parts[2])
                _detector_origin_cache = (x0, y0, z0)
                print(f"✅ Geometry origin loaded: x0={x0:.3f}, y0={y0:.3f}, z0={z0:.3f} mm")
                return _detector_origin_cache
            if line.startswith("GEOM_ERR:"):
                raise RuntimeError(f"Geometry macro reported error: {line}")
    except FileNotFoundError:
        raise RuntimeError(
            "'root' binary not found on PATH. Cannot read geometry origin."
        )
    finally:
        try:
            import os
            if 'macro_path' in locals() and os.path.exists(macro_path):
                os.remove(macro_path)
        except Exception:
            pass

    raise RuntimeError(
        f"Could not parse geometry origin from ROOT output.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def load_centering_origin(monitor_id_value: int) -> tuple[float, float, float]:
    """Return centering origin for the requested monitor.

    For midstream MonitorID=4, use the verified monitor-plane origin.
    For other monitor IDs, fall back to CDC origin from geometry.
    """
    # if int(monitor_id_value) == 4:
    return _MIDSTREAM_MONITOR4_ORIGIN
    # return load_detector_origin()


def _normalize_pdg_allowlist(pdg_allowlist) -> Optional[list[int]]:
    if pdg_allowlist is None:
        return None
    if isinstance(pdg_allowlist, str):
        tokens = [tok.strip() for tok in pdg_allowlist.split(",") if tok.strip()]
    else:
        try:
            tokens = list(pdg_allowlist)
        except TypeError as exc:
            raise ValueError("pdg_allowlist must be a comma-separated string or an iterable of ints") from exc

    if not tokens:
        return []

    parsed = []
    for token in tokens:
        try:
            parsed.append(int(token))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid PDG code in pdg_allowlist: {token}") from exc
    return sorted(set(parsed))


def _transform_preprocessed_batch(
    df_batch: pd.DataFrame,
    pdg_code=None,
    pdg_allowlist: Optional[list[int]] = None,
    keep_pdg: bool = False,
) -> pd.DataFrame:
    """Apply filtering and feature transforms to one preprocessed parquet batch."""
    if "pdg" in df_batch.columns:
        if pdg_allowlist is not None:
            pdg_mask = df_batch["pdg"].isin(pdg_allowlist)
            df_batch = cast(pd.DataFrame, df_batch.loc[pdg_mask].copy())
        if pdg_code is not None:
            df_batch = cast(pd.DataFrame, df_batch.loc[df_batch["pdg"] == pdg_code].copy())
        if (pdg_code is not None or pdg_allowlist is not None) and not keep_pdg:
            df_batch = df_batch.drop(columns=["pdg"])

    if len(df_batch) == 0:
        return df_batch

    if "p_mag" in df_batch.columns:
        df_batch["log1p_p_mag"] = np.log1p(np.clip(df_batch["p_mag"].to_numpy(dtype=np.float64), 0.0, None))
        df_batch = df_batch.drop(columns=["p_mag"])

    # Clip to center circle
    if "r" in df_batch.columns:
        df_batch = cast(pd.DataFrame, df_batch.loc[df_batch["r"] < 350].copy())

    if "r" in df_batch.columns:
        df_batch["log1p_r"] = np.log1p(np.clip(df_batch["r"].to_numpy(dtype=np.float64), 0.0, None))
        df_batch = df_batch.drop(columns=["r"])

    # Comment out usually, leave in for debugging
    if "x" in df_batch.columns:
        df_batch = cast(pd.DataFrame, df_batch.loc[np.isclose(df_batch["x"], -0.03999999999859938, atol=1e-9)].copy())
    #     df_batch = df_batch.drop(columns=["x"])  # Remove x feature

    return df_batch


def iter_preprocessed_data(
    parquet_file,
    pdg_code=None,
    pdg_allowlist=None,
    entries=None,
    test_entries=None,
    batch_size=65536,
    keep_pdg: bool = False,
):
    """Yield transformed parquet data in batches to avoid full-file reads.

    Parameters
    ----------
    parquet_file : str
        Path to preprocessed parquet file.
    pdg_code : int | None
        Optional PDG filter. If provided, keeps only matching rows.
    pdg_allowlist : str | Iterable[int] | None
        Optional multi-class PDG filter. Accepts comma-separated string or iterable.
    entries : int | None
        Optional base number of rows to yield after filtering.
    test_entries : int | None
        Optional extra rows to include on top of ``entries``.
        If both are provided, total yielded rows are ``entries + test_entries``.
    batch_size : int
        Parquet read batch size.
    keep_pdg : bool
        If True, preserve the `pdg` column after filtering.
    """
    parquet = pq.ParquetFile(parquet_file)
    normalized_allowlist = _normalize_pdg_allowlist(pdg_allowlist)

    entries_value = None if entries is None else int(entries)
    test_entries_value = None if test_entries is None else int(test_entries)

    if entries_value is None and test_entries_value is None:
        max_entries = None
    elif entries_value is None:
        max_entries = test_entries_value
    elif test_entries_value is None:
        max_entries = entries_value
    else:
        max_entries = entries_value + test_entries_value

    if max_entries is not None and max_entries <= 0:
        return

    yielded = 0
    for record_batch in parquet.iter_batches(batch_size=int(batch_size)):
        df_batch = record_batch.to_pandas()
        df_batch = _transform_preprocessed_batch(
            df_batch,
            pdg_code=pdg_code,
            pdg_allowlist=normalized_allowlist,
            keep_pdg=keep_pdg,
        )

        if len(df_batch) == 0:
            continue

        if max_entries is not None:
            remaining = max_entries - yielded
            if remaining <= 0:
                break
            if len(df_batch) > remaining:
                df_batch = df_batch.iloc[:remaining].copy()

        if len(df_batch) == 0:
            continue

        yielded += len(df_batch)
        yield df_batch

        if max_entries is not None and yielded >= max_entries:
            break

def load_preprocessed_data(
    parquet_file,
    pdg_code=None,
    pdg_allowlist=None,
    entries=None,
    test_entries=None,
    keep_pdg: bool = False,
):
    """Load preprocessed Parquet file, apply timing split, and convert to cylindrical coordinates."""

    print(f"Loading preprocessed data from {parquet_file} (streaming batches)")

    chunks = []
    loaded = 0
    for chunk_df in iter_preprocessed_data(
        parquet_file,
        pdg_code=pdg_code,
        pdg_allowlist=pdg_allowlist,
        entries=entries,
        test_entries=test_entries,
        keep_pdg=keep_pdg,
    ):
        chunks.append(chunk_df)
        loaded += len(chunk_df)

    if not chunks:
        print("✅ Loaded 0 entries")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(df)} entries")

    if entries is not None or test_entries is not None:
        entries_value = 0 if entries is None else int(entries)
        test_entries_value = 0 if test_entries is None else int(test_entries)
        requested_total = entries_value + test_entries_value
        print(
            f"Requested streamed rows: entries={entries_value}, "
            f"test_entries={test_entries_value}, total={requested_total}"
        )

    if pdg_code is not None:
        print(f"Filtered to PDG={pdg_code}, {len(df)} entries remaining")
    if pdg_allowlist is not None:
        normalized_allowlist = _normalize_pdg_allowlist(pdg_allowlist)
        print(f"Filtered to PDG allowlist={normalized_allowlist}, {len(df)} entries remaining")

    return df

def load_root_files(file_pattern, max_files=10, pdg_code=None, monitor_id_value=None):
    """Load ROOT files and convert to DataFrame with cylindrical position and spherical momentum."""
    file_list = sorted(glob.glob(file_pattern))[:max_files]
    
    if not file_list:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(file_list)} files")
    
    dfs = []
    for file_path in file_list:
        try:
            with uproot.open(file_path) as root_file:
                tree = root_file["RooTrackerTree"]
                
                arrays = tree.arrays(
                    ["StdHepPdg", "StdHepP4", "StdHepX4", "MonitorID"],
                    library="ak"
                )
                
                pdg = ak.flatten(arrays["StdHepPdg"])
                p4 = ak.flatten(arrays["StdHepP4"])
                x4 = ak.flatten(arrays["StdHepX4"])
                monitor_id = ak.flatten(arrays["MonitorID"])
                
                particle_mask = ak.ones_like(pdg, dtype=bool)
                if pdg_code is not None:
                    particle_mask = particle_mask & (pdg == pdg_code)
                if monitor_id_value is not None:
                    particle_mask = particle_mask & (monitor_id == monitor_id_value)
                
                pdg_filtered = pdg[particle_mask]
                p4_filtered = p4[particle_mask]
                x4_filtered = x4[particle_mask]

                # Keep awkward arrays until final conversion
                x_cart = x4_filtered[:, 0]
                y_cart = x4_filtered[:, 1]
                z_cart = x4_filtered[:, 2]
                
                # Centre using verified monitor-aware geometry origin
                geom_x0, geom_y0, geom_z0 = load_centering_origin(monitor_id_value)
                x_cart = x_cart - geom_x0
                y_cart = y_cart - geom_y0
                z_cart = z_cart - geom_z0

                # Spatial cylindrical around detector x-axis:
                # x = longitudinal (beam direction)
                # r = sqrt(y^2 + z^2) = radial distance from x-axis
                # phi_s = atan2(z, y) = azimuthal angle in y-z plane
                x = ak.to_numpy(x_cart)
                y = ak.to_numpy(y_cart)
                z = ak.to_numpy(z_cart)
                r = np.sqrt(y**2 + z**2)
                phi_spatial = np.arctan2(z, y)
                sin_phi_s = np.sin(phi_spatial)
                cos_phi_s = np.cos(phi_spatial)

                # Momentum extraction (spherical coordinates):
                pz = ak.to_numpy(p4_filtered[:, 0])
                py = ak.to_numpy(p4_filtered[:, 1])
                px = ak.to_numpy(p4_filtered[:, 2])
                t = ak.to_numpy(p4_filtered[:, 3])

                # Momentum spherical:
                # r = |p| = sqrt(px^2 + py^2 + pz^2)
                # theta = polar angle from x-axis = acos(px / |p|)
                # phi_p = azimuthal angle in y-z plane = atan2(pz, py)
                p_mag = np.sqrt(px**2 + py**2 + pz**2)
                cos_theta = np.divide(px, p_mag, where=(p_mag > 0), out=np.ones_like(px))
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                sin_theta = np.sin(theta)
                phi_p = np.arctan2(pz, py)

                log_t = np.log(t + 1e-10)
                
                file_df = pd.DataFrame({
                    "pdg": ak.to_numpy(pdg_filtered),  # ADD THIS LINE
                    "log_t": log_t,
                    "x": x,
                    "r": r,
                    "sin_phi_s": sin_phi_s,
                    "cos_phi_s": cos_phi_s,
                    "p_mag": p_mag,
                    "sin_theta": sin_theta,
                    "cos_theta": cos_theta,
                    "phi_p": phi_p
                })
                
                dfs.append(file_df)
                print(f"Loaded: {file_path.split('/')[-1]} ({len(file_df)} entries)")
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()