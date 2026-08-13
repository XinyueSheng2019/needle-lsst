import json
import os
import sys
import lasair
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from kn_model import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
LSST_BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

class KNLinear:
    """Per-object killnova feature extraction and classification."""

    def __init__(self, object_info: dict, model_bundle=None):
        self.object_info = object_info
        self.object_id = str(
            object_info.get("diaObjectId", object_info.get("objectId", ""))
        )
        self.model_bundle = model_bundle
        self.lasair_data = object_info.get("lasairData", {})
        self.sherlock_data = self.lasair_data.get("sherlock", {})
        self.photo_df = self.get_photo_df(unify_lc=True)
        self.get_peak()
        self.meta = self.get_meta()
        

    @classmethod
    def from_object_id(cls, object_id: str, model_bundle=None):
        path = os.path.join(EXAMPLES_DIR, f"{object_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Example {object_id} not found at {path}")
        with open(path) as f:
            return cls(json.load(f), model_bundle=model_bundle)

    @staticmethod
    def _nearest_band(target_band, detected_bands):
        """Pick the detected band closest in wavelength to the target band."""
        target_idx = LSST_BANDS.index(target_band)
        return min(detected_bands, key=lambda band: abs(LSST_BANDS.index(band) - target_idx))

    def band_padding(self):
        """Pad missing bands during the fade phase using the KN colour model."""
        if self.photo_df is None or self.photo_df.empty:
            return None

        photo_df = self.photo_df.copy()
        
        fade_df = photo_df[photo_df["time"] >= self.peak_time]
        rows = []

        for detection_time, detection_df in fade_df.groupby("time", sort=True):
            detected_bands = set(detection_df["band"])
            delta_t = detection_time - self.peak_time

            for band in LSST_BANDS:
                if band in detected_bands:
                    rows.append(detection_df.loc[detection_df["band"] == band].iloc[0])
                    continue

                ref_band = self._nearest_band(band, detected_bands)
                ref_row = detection_df.loc[detection_df["band"] == ref_band].iloc[0]
                colour_offset, colour_err = fading_colour(
                    delta_t+1e-3, band, ref_band, convert_to_mag=False
                )
                rows.append({
                    "time": detection_time,
                    "band": band,
                    "flux": ref_row["flux"] + colour_offset,
                    "flux_err": np.hypot(ref_row["flux_err"], colour_err),
                })

        if not rows:
            return photo_df

        padded_fade_df = pd.DataFrame(rows)
        rise_df = photo_df[photo_df["time"] < self.peak_time]
        if rise_df.empty:
            result = padded_fade_df
        else:
            result = pd.concat([rise_df, padded_fade_df], ignore_index=True)

        result["mag"], result["mag_err"] = convert_flux_to_mag(result["flux"], result["flux_err"])
        result["mag"] = result["mag"] - result["mag"].min()
        return result.sort_values("time").reset_index(drop=True)

 
    def get_peak(self, photo_df = None):
        photo_df = self.photo_df if photo_df is None else photo_df
        self.peak_time = None
        self.peak_flux = None
        self.peak_flux_err = None
        self.peak_mag = None
        self.peak_mag_err = None
        if photo_df is not None:   
            peak_idx = photo_df["flux"].idxmax()
            peak_row = photo_df.loc[peak_idx]
            self.peak_time = float(peak_row["time"])
            self.peak_flux = float(peak_row["flux"])
            self.peak_flux_err = float(peak_row["flux_err"])
            self.peak_mag = convert_flux_to_mag(self.peak_flux, self.peak_flux_err)[0]
            self.peak_mag_err = convert_flux_to_mag(self.peak_flux, self.peak_flux_err)[1]


    def _flux_to_normalized_mag(self, photo_df, flux_values):
        """Convert flux values to the same normalized mag scale as photo_df."""
        raw_mags, _ = convert_flux_to_mag(photo_df["flux"].values, photo_df["flux_err"].values)
        mag_ref = float(np.min(raw_mags))
        pred_mags, mag_errs = convert_flux_to_mag(
            np.asarray(flux_values, dtype=float),
            np.full_like(flux_values, 1e-9, dtype=float),
        )
        self.mag_ref = mag_ref
        return pred_mags - mag_ref, mag_errs
    

    def check_rise(self, photo_df = None):

        photo_df = self.photo_df if photo_df is None else photo_df
        if photo_df is None:
            raise ValueError("photo_df is None")

        rise_df = photo_df[photo_df["time"] <= self.peak_time].copy()
        rise_df["delta_t"] = rise_df["time"] - self.peak_time

        if len(rise_df) <= 1:
            return True

        if abs(rise_df["delta_t"].min()) <= 2.0: # 100 eJy/day, fade_df["flux_err"]/
            return True
        else:
            return False
            

    def check_fade(self):
        decay_by_band = (
            self.meta.get("decay_by_band_padded")
            or self.meta.get("decay_by_band")
            or {}
        )
        chi_square = []
        for band, fit in decay_by_band.items():
            if fit is None or fit.get("decay_beta") is None:
                continue
            chi_square.append(
                (fit["decay_beta"] - decay_beta[band]) ** 2 / decay_beta_err[band] ** 2
            )
        return float(np.mean(chi_square)) if chi_square else None

    def _fit_band_fade(self, band_df):
        """Fit mag = intercept + beta * delta_t after the universal peak."""
        if band_df is None or band_df.empty or self.peak_time is None:
            return None

        fade_df = band_df[band_df["time"] >= self.peak_time].copy()
        fade_df["delta_t"] = fade_df["time"] - self.peak_time
        if len(fade_df) < 2:
            return None

        mag_beta, mag_intercept = stats.linregress(fade_df["delta_t"], fade_df["mag"])[0:2]

        if mag_beta is None or mag_intercept is None:
            return None

        return {
            "decay_mag_beta": mag_beta,
            "decay_mag_intercept": mag_intercept,
            "peak_time": self.peak_time,
            "decay_time": float(fade_df["delta_t"].max()),
            "n_points": len(fade_df),
        }

    def fit_fade(self, photo_df=None, padding=True):
        """Fit per-band power-law fade: |delta_flux| = amplitude * delta_t^beta."""
        if photo_df is None:
            photo_df = self.band_padding() if padding else self.photo_df
        if photo_df is None or photo_df.empty:
            return None

        decay_fits = {}
        for band in photo_df["band"].unique():
            band_fit = self._fit_band_fade(photo_df[photo_df["band"] == band])
            if band_fit is not None:
                decay_fits[band] = band_fit
        return decay_fits if decay_fits else None

    def fit_rise(self, photo_df=None):
        """Linear rise fit before the band peak."""
        photo_df = self.photo_df if photo_df is None else photo_df
        if photo_df is None:
            return None

        rise_df = photo_df[photo_df["time"] <= self.peak_time]
        if len(rise_df) < 2:
            return None

        flux_slope, flux_intercept = stats.linregress(rise_df["time"], rise_df["flux"])[0:2]
        mag_slope, mag_intercept = stats.linregress(rise_df["time"], rise_df["mag"])[0:2]


        if flux_slope is None:
            return None

        return {
            "rise_mag_slope": float(mag_slope),
            "rise_mag_intercept": float(mag_intercept),
            "rise_flux_slope": float(flux_slope) if flux_slope is not None else None,
            "rise_flux_intercept": float(flux_intercept) if flux_intercept is not None else None,
            "rise_time": float(self.peak_time - rise_df["time"].min()),
        }

    def _aggregate_fade_fits(self, fade_by_band):
        if not fade_by_band:
            return None, None, None
        betas = [fit["decay_mag_beta"] for fit in fade_by_band.values()]
        weights = [fit["n_points"] for fit in fade_by_band.values()]
        decay_beta = float(np.average(betas, weights=weights))
        decay_intercept = float(np.mean([fit["decay_mag_intercept"] for fit in fade_by_band.values()]))
        decay_time = max(fit["decay_time"] for fit in fade_by_band.values())
        return decay_beta, decay_intercept, decay_time

    def get_meta(self):
        r_latest = self.lasair_data.get("r_latestMJD", self.lasair_data.get("g_latestMJD"))
        r_first = self.lasair_data.get("r_firstMJD", self.lasair_data.get("g_firstMJD"))
        r_last = self.lasair_data.get("r_lastMJD", self.lasair_data.get("g_lastMJD"))

        rise_fit = self.fit_rise()
        fade_by_band = self.fit_fade(padding=False)
        fade_by_band_padded = self.fit_fade(padding=True)
        valid_fade = fade_by_band or {}
        valid_fade_padded = fade_by_band_padded or {}

        decay_beta, decay_intercept, decay_time = self._aggregate_fade_fits(valid_fade_padded)
        if decay_beta is None:
            decay_beta, decay_intercept, decay_time = self._aggregate_fade_fits(valid_fade)
        if decay_time is None:
            decay_time = (
                (r_latest - r_last) if r_latest is not None and r_last is not None else None
            )

        return {
            "rise_time": rise_fit["rise_time"] if rise_fit else (
                (r_latest - r_first) if r_latest is not None and r_first is not None else None
            ),
            "rise_mag_slope": rise_fit["rise_mag_slope"] if rise_fit else None,
            "rise_mag_intercept": rise_fit["rise_mag_intercept"] if rise_fit else None,
            "rise_flux_slope": rise_fit["rise_flux_slope"] if rise_fit else None,
            "rise_flux_intercept": rise_fit["rise_flux_intercept"] if rise_fit else None,
            "decay_time": decay_time,
            "decay_mag_beta": decay_beta,
            "decay_mag_intercept": decay_intercept,
            "decay_by_band": valid_fade if valid_fade else None,
            "decay_by_band_padded": valid_fade_padded if valid_fade_padded else None,
            "fade_chi_square": self._fade_chi_square(valid_fade_padded or valid_fade),
            "peak_time": self.peak_time,
            "peak_mag": self.peak_mag,
            "peak_mag_err": self.peak_mag_err,
        }

    def _fade_chi_square(self, decay_by_band):
        chi_square = []
        for band, fit in (decay_by_band or {}).items():
            if fit is None or fit.get("decay_beta") is None:
                continue
            chi_square.append(
                (fit["decay_beta"] - decay_beta[band]) ** 2 / decay_beta_err[band] ** 2
            )
        return float(np.mean(chi_square)) if chi_square else None




    def uniform_light_curve(self, photo_df=None, window_size = 0.5):
        '''
        merge detection within the window size.
        '''
        photo_df = self.photo_df if photo_df is None else photo_df
        # Remove any duplicate rows based on all columns
        if photo_df is None:
            return None
        photo_df = photo_df.drop_duplicates(subset=['time', 'band', 'flux', 'flux_err'], keep='first')
        
        new_photo_df = pd.DataFrame()
        for band in LSST_BANDS:
            band_data = photo_df[photo_df['band'] == band].reset_index(drop=True)
            band_data = band_data.sort_values("time")
            groups = (band_data["time"].diff().ge(window_size)).cumsum()
            merged = band_data.groupby(groups).agg({
                "time": "first",
                "flux": "mean",
                "flux_err": lambda x: np.sqrt((x**2).sum()) / len(x),
                "band": "first"
            })

            new_photo_df = pd.concat([new_photo_df, merged]).reset_index(drop=True)
        
        return new_photo_df
       

    def get_photo_df(self, min_detection = 2, photo_df = None, unify_lc = True):

        if self.lasair_data.get("nSourcesGood", 0) >= min_detection:

            photometric_data = self.object_info["diaSourcesList"]
            photo_df = pd.DataFrame()
            photo_df["time"] = [
                data["midpointMjdTai"] - photometric_data[-1]["midpointMjdTai"]
                for data in photometric_data
            ]
            
            photo_df["flux"] = [data["psfFlux"] for data in photometric_data]
            photo_df["flux_err"] = [data["psfFluxErr"] for data in photometric_data]
            photo_df['band'] = [data["band"] for data in photometric_data]
            photo_df['reliability'] = [data["reliability"] for data in photometric_data]
            photo_df['time'] = photo_df['time'].round(1)
   
            photo_df = photo_df[photo_df['flux']/photo_df['flux_err'] >= 3.0]

            if unify_lc:
                photo_df = self.uniform_light_curve(photo_df=photo_df, window_size=0.5)

            photo_df['mag'], photo_df['mag_err'] = self._flux_to_normalized_mag(photo_df, photo_df["flux"])
            
            return photo_df
        else:
            return None

    def predict(self) -> dict:
        """Predict the probability of the object being a Kilonova using kn model."""
        if self.photo_df is None:
            return {"KN": 0.0, "non-KN": 1.0}
        if not self.check_rise():
            return {"KN": 0.0, "non-KN": 1.0}

        chi_square = self.meta.get("fade_chi_square")
        if chi_square is None:
            chi_square = self.check_fade()
        if chi_square is None:
            return {"KN": 0.0, "non-KN": 1.0}

        prob_KN = 1 / (1 + np.exp(chi_square))
        return {"KN": prob_KN, "non-KN": 1.0 - prob_KN}

    def annotate(self) -> dict:
        """Return an annotation payload for Lasair."""
        return {
            "classification": "kilonova_candidate",
            "explanation": (
                f"KN linear filter: rise_time={self.meta['rise_time']}, "
                f"decay_beta={self.meta['decay_beta']}"
            ),
            "classdict": self.meta,
            "version": "0.1",
            "url": "",
        }

    def plot_object(self, photo_df = None, unit = "mag", show_fit = False, padding = True, predict_result = None):
        band_colors = {'u': 'red', 'g': 'green', 'r': 'blue', 'i': 'purple', 'z': 'orange', 'y': 'brown'}
        y_col = "mag" if unit == "mag" else "flux"
        y_err_col = "mag_err" if unit == "mag" else "flux_err"

        def _plot_df(df, fmt = "o"):
            for band in df["band"].unique():
                band_df = df[df["band"] == band]
                plt.errorbar(
                    band_df["time"],
                    band_df[y_col],
                    yerr=band_df[y_err_col],
                    color=band_colors[band],
                    fmt=fmt,
                    label=band if fmt == "o" else band + "_pad",
                )

        photo_df = self.photo_df if photo_df is None else photo_df
        if photo_df is None:
            print("No photometry data available.")
            return

    
        _plot_df(photo_df)
        if padding:
            padded_photo_df = self.band_padding()
     
            # Get only the padded data by removing rows where (time, band) exists in the original photo_df ("real" data points).
            real_keys = set(zip(photo_df["time"], photo_df["band"]))
            padded_photo_df = padded_photo_df[
                ~padded_photo_df.apply(lambda row: (row["time"], row["band"]) in real_keys, axis=1)
            ]

            _plot_df(padded_photo_df, fmt="s")
        
        plt.vlines(x=self.peak_time, ymin=np.min(photo_df[y_col]) - np.max(photo_df[y_err_col]), ymax=np.max(photo_df[y_col]) + np.max(photo_df[y_err_col]), color="red", linestyle="--")


        if show_fit:
            fit_times = np.linspace(photo_df["time"].min(), photo_df["time"].max(), 200)
            plt.vlines(
                x=self.peak_time,
                ymin=np.min(photo_df[y_col]) - np.max(photo_df[y_err_col]),
                ymax=np.max(photo_df[y_col]) + np.max(photo_df[y_err_col]),
                color="red",
                linestyle="--",
            )

            if self.meta.get("rise_mag_slope") is not None and self.meta.get("rise_mag_intercept") is not None:
                rise_mask = fit_times <= self.peak_time
                if unit == "mag":
                    rise_y = (
                        self.meta["rise_mag_slope"] * fit_times[rise_mask]
                        + self.meta["rise_mag_intercept"]
                    )
                else:
                    rise_y = (
                        self.meta.get("rise_flux_slope")
                        * fit_times[rise_mask]
                        + self.meta.get("rise_flux_intercept")
                    )
                plt.plot(fit_times[rise_mask], rise_y, "k--", label=f"rise fit (averaged slope: {self.meta['rise_mag_slope']:.2f})")

            decay_key = "decay_by_band_padded" if padding else "decay_by_band"
            decay_by_band = self.meta.get(decay_key) or self.meta.get("decay_by_band")
            if decay_by_band:
         
                for band, band_fit in decay_by_band.items():
    
                    band_peak_t = band_fit.get("peak_time")
                    mag_beta = band_fit.get("decay_mag_beta") 
                    mag_intercept = band_fit.get("decay_mag_intercept")
                    if band_peak_t is None or mag_beta is None or mag_intercept is None:
                        continue

                    fade_mask = fit_times > band_peak_t
                    delta_t = fit_times[fade_mask] - band_peak_t
                    valid = delta_t > 0
                    if not np.any(valid):
                        continue

                    mag_pred = mag_intercept + mag_beta * delta_t[valid]
               
                    if unit == "flux":
                        fade_y = 10**((mag_pred + self.mag_ref)/-2.5)
                    else:
                        fade_y = mag_pred

                    plt.plot(
                        fit_times[fade_mask][valid],
                        fade_y,
                        color=band_colors[band],
                        linestyle="--",
                        label=f"fade fit ({band}) beta: {mag_beta:.2f}",
                    )
        
         
        plt.legend(loc='upper right')
        plt.xlabel("Time (days)")
        plt.ylabel(f"{unit}")
        # if unit == "flux": plt.yscale("log")
        if unit == "mag": plt.gca().invert_yaxis()
        plt.title(
            f"{self.object_info['diaObjectId']}, KN: {predict_result['KN']:.3f}, non-KN: {predict_result['non-KN']:.3f}"
            if predict_result is not None
            else str(self.object_info["diaObjectId"])
        )
        plt.show()
        


def classify_object_info(object_info: dict) -> dict:
    """Module-level worker for parallel classification."""
    return KNLinear(object_info).predict()


def fetch_examples_from_lasair(example_table="example_table.dat"):
    example_ids = []
    table_path = os.path.join(os.path.dirname(__file__), example_table)
    with open(table_path) as f:
        for line in f:
            object_id = line.split()[0]
            example_ids.append(object_id)

    client = lasair.lasair_client(config.LASAIR_API_TOKEN)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for object_id in example_ids:
        object_info = client.object(object_id, lasair_added=True, lite=False)
        out_path = os.path.join(EXAMPLES_DIR, f"{object_id}.json")
        with open(out_path, "w") as f:
            json.dump(object_info, f, indent=4)
            print(f"Saved example {object_id} to {out_path}")


def main(object_id = str | None):
    if object_id is None:
        random_example = np.random.choice(os.listdir(EXAMPLES_DIR))
        object_id = random_example.split(".")[0]
    kn_linear = KNLinear.from_object_id(object_id)
    print("random_example:", object_id)
    # print(kn_linear.meta)
    kn_linear.plot_object(photo_df=kn_linear.photo_df, unit="mag", show_fit=True, padding=False, predict_result=kn_linear.predict())




if __name__ == "__main__":
    import random

    # Loop over example files, pick random objects until dataframe is not None
    files = os.listdir(EXAMPLES_DIR)
    random.shuffle(files)
    for file in files:
        object_id = file.split(".")[0]
        try:
            kn_linear = KNLinear.from_object_id(object_id)
            if kn_linear.photo_df is not None and not kn_linear.photo_df.empty:
                print(f"Selected object with non-empty dataframe: {object_id}")
                break
        except Exception as e:
            print(f"Skipping {object_id}, error: {e}")
    else:
        print("No object found with a non-empty dataframe.")
        kn_linear = None

    main(object_id) 
    # '170019696272736349'
