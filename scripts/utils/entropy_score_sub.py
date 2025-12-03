import os
import sys
import csv
import numpy as np
import nibabel as nib


# --------------------
#  ENTROPY FUNCTION
# --------------------
def compute_entropy_from_volume(seg_mean, lower=30, upper=255, max_value=255.0):
    """Intensity-distribution entropy (현재 사용 중인 방식)"""
    seg_mean = seg_mean / max_value

    # intensity 범위 기반 ROI 선택
    mask = (seg_mean >= lower / max_value) & (seg_mean <= upper / max_value)
    roi_values = seg_mean[mask]

    total = np.sum(roi_values)
    if total == 0:
        return 0.0

    probs = roi_values / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def load_nii(path):
    return nib.load(path).get_fdata()


# --------------------
#  MAIN
# --------------------
def main(src_base):

    # 12개 태그 정의
    tags_L = ["LVS","LAC","LPC","LAP","LPP","LVP"]
    tags_R = ["RVS","RAC","RPC","RAP","RPP","RVP"]
    tags_all = tags_L + tags_R   # 12개

    base_path = os.path.join(os.getcwd(), src_base)
    pid_list = sorted(os.listdir(base_path))

    # CSV 출력 파일명
    out_csv = os.path.join(os.getcwd(), "entropy_table.csv")

    # CSV Header
    header = ["PID"] + [f"ENT_{t}" for t in tags_all]

    rows = []

    print(f"=== Computing ENTROPY for {len(pid_list)} subjects ===")

    for pid in pid_list:
        pid_dir = os.path.join(base_path, pid)

        row = [pid]   # 첫 컬럼 = PID

        # -------------------------
        # 12개 ROI 각각 ENT 계산
        # -------------------------
        for tag in tags_all:
            nii_path = os.path.join(pid_dir, f"{pid}_{tag}_prb.nii.gz")

            if os.path.exists(nii_path):
                vol = load_nii(nii_path)
                ent = compute_entropy_from_volume(vol, lower=30, upper=200)
            else:
                ent = None

            row.append(ent)

        rows.append(row)

    # -------------------------
    # CSV 저장
    # -------------------------
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nCSV 저장 완료 → {out_csv}")


# --------------------
# ENTRY POINT
# --------------------
if __name__ == '__main__':
    if len(sys.argv) == 2:
        src_base = os.path.normpath(str(sys.argv[1]))
        main(src_base)
    else:
        print("Usage: python script.py <src_base_path>")
