import os
import sys

sys.path.append('.')
from tasks import mimic_utils
import pandas as pd

def los_mimic(mimic_dir: str, save_dir: str, seed: int, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ADMITTIME" and "DISCHTIME" from ADMISSIONS.csv.
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "LOS_WEEKS"
    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # Calculating the Length of Stay in days per admission
    mimic_admissions['ADMITTIME'] = pd.to_datetime(mimic_admissions['ADMITTIME'])
    mimic_admissions['DISCHTIME'] = pd.to_datetime(mimic_admissions['DISCHTIME'])

    mimic_admissions['LOS_days'] = round(
        (mimic_admissions['DISCHTIME'] - mimic_admissions['ADMITTIME']).dt.total_seconds() / (24 * 60 * 60), 1)

    mimic_admissions = mimic_admissions[["ROW_ID", "SUBJECT_ID", "HADM_ID", "LOS_days", "HOSPITAL_EXPIRE_FLAG"]]

    # Creation of Label
    """
        <= 3: 0
        > 3 & <= 7: 1
        > 7 & <= 14: 2
        >14 & <= 21: 3
        >21: 4
    """
    mimic_admissions.loc[mimic_admissions['LOS_days'] <= 3, 'LOS_label'] = 0
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 3) & (
            mimic_admissions['LOS_days'] <= 7), 'LOS_label'] = 1
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 7) & (
            mimic_admissions['LOS_days'] <= 14), 'LOS_label'] = 2
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 14) & (
            mimic_admissions['LOS_days'] <= 21), 'LOS_label'] = 3
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 14), 'LOS_label'] = 4
    mimic_admissions.LOS_label = mimic_admissions.LOS_label.astype(int)

    """
    New version
        <= 1: 0
        > 1 <= 2: 1
        > 2 <= 3: 2
        > 3 & <= 4: 3
        > 4 & <= 5: 4
        > 5 & <= 7: 5
        > 7 & <= 10: 6
        > 10 & <= 14: 7
        > 14 & <= 21: 8
        > 21: 9
    mimic_admissions.loc[mimic_admissions['LOS_days'] <= 1, 'LOS_label'] = 0
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 1) & (
            mimic_admissions['LOS_days'] <= 2), 'LOS_label'] = 1
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 2) & (
            mimic_admissions['LOS_days'] <= 3), 'LOS_label'] = 2
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 3) & (
            mimic_admissions['LOS_days'] <= 4), 'LOS_label'] = 3
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 4) & (
            mimic_admissions['LOS_days'] <= 5), 'LOS_label'] = 4
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 5) & (
            mimic_admissions['LOS_days'] <= 7), 'LOS_label'] = 5
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 7) & (
            mimic_admissions['LOS_days'] <= 10), 'LOS_label'] = 6
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 10) & (
            mimic_admissions['LOS_days'] <= 14), 'LOS_label'] = 7
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 14) & (
            mimic_admissions['LOS_days'] <= 21), 'LOS_label'] = 8
    mimic_admissions.loc[(mimic_admissions['LOS_days'] > 21), 'LOS_label'] = 9
    mimic_admissions.LOS_label = mimic_admissions.LOS_label.astype(int)
    """

    # Keeping the required variables
    mimic_admissions = mimic_admissions[["HADM_ID", "LOS_label", "HOSPITAL_EXPIRE_FLAG"]]
    mimic_notes = mimic_notes[['HADM_ID', 'TEXT', "ROW_ID", "SUBJECT_ID"]]

    # Merging Mimic Notes data with Admissions data
    notes_adm_df = pd.merge(mimic_notes, mimic_admissions, how="left", on="HADM_ID")

    # Removing records where the patient died within a given hospitalization
    notes_adm_df = notes_adm_df[notes_adm_df['HOSPITAL_EXPIRE_FLAG'] == 0]
    notes_adm_df = notes_adm_df[["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT", "LOS_label"]]

    print(f'LOS_label: {mimic_admissions.LOS_label.value_counts()}')
    mimic_utils.save_mimic_split_patient_wise(notes_adm_df,
                                              label_column='LOS_label',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    los_mimic(args.mimic_dir, args.save_dir, args.seed, args.admission_only)
