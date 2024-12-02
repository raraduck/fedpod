import os
import shutil
import nibabel as nib
import sys
import numpy as np
import glob

def main(src_base, postfix):
    base = os.path.dirname(src_base)
    pid = os.path.basename(src_base)
    dst_folder = os.path.join(base, pid)
    assert not os.path.exists(os.path.join(dst_folder, f'{pid}_sub_{postfix}.nii.gz')), f"{pid}_sub_{postfix}.nii.gz already exists."
    pattern = os.path.join(src_base, '[0-3].nii.gz')
    items = glob.glob(pattern)
    assert len(items) == 4, f"[FAILED] four files (0~3.nii.gz) are not detected on the given path."
    for el in items:
        print(el)
    sub_list = ['0.nii.gz', '3.nii.gz', '2.nii.gz', '1.nii.gz']
    label_list = [3, 4, 2, 1]
    final_data = None
    for subregion, label_postfix in zip(sub_list, label_list):
        src_file = os.path.join(base, pid, subregion)
        # pet_file = os.path.join(src_path, src_folder, 'realigned_pet.nii.gz')
        proxy = nib.load(src_file)
        data = proxy.get_fdata()
        affine = proxy.affine
        active_mask = np.isin(data, [11, 12, 26, 50, 51, 58])
        active_data = np.where(active_mask, data, 0)
        active_data[active_data == 11] = 11 * 10 + label_postfix  # L Caudate
        active_data[active_data == 12] = 12 * 10 + label_postfix  # L Putamen
        active_data[active_data == 26] = 26 * 10 + 1  # L Accumbens
        active_data[active_data == 50] = 50 * 10 + label_postfix  # R Caudate
        active_data[active_data == 51] = 51 * 10 + label_postfix  # R Putamen
        active_data[active_data == 58] = 58 * 10 + 1  # R Accumbens
        # 최종 데이터 배열 초기화
        if final_data is None:
            final_data = np.zeros_like(active_data)

        # 0이 아닌 값을 가진 픽셀만 덮어쓰기
        nonzero_mask = active_data != 0
        final_data[nonzero_mask] = active_data[nonzero_mask]

    # # 최종 데이터를 NIfTI 이미지로 저장
    # final_img = nib.Nifti1Image(final_data, affine)
    # nib.save(final_img, os.path.join(dst_folder, f'{pid}_sub_unknown.nii.gz'))

    active_mask = final_data > 0
    active_data = np.where(active_mask, final_data, 0)
    active_data[np.isin(active_data, [111, 121, 261])] = 1  # LVS,
    active_data[np.isin(active_data, [112])] = 2  # LAC,
    active_data[np.isin(active_data, [113])] = 3  # LAP,
    active_data[np.isin(active_data, [122])] = 4  # LAP,
    active_data[np.isin(active_data, [123])] = 5  # LPP,
    active_data[np.isin(active_data, [124])] = 6  # LVP,
    active_data[np.isin(active_data, [501, 511, 581])] = 7  # RVS,
    active_data[np.isin(active_data, [502])] = 8  # RAC,
    active_data[np.isin(active_data, [503])] = 9  # RAP,
    active_data[np.isin(active_data, [512])] = 10  # RAP,
    active_data[np.isin(active_data, [513])] = 11  # RPP,
    active_data[np.isin(active_data, [514])] = 12  # RVP,

    nonzero_mask = active_data != 0
    final_data[nonzero_mask] = active_data[nonzero_mask]
    # 최종 데이터를 NIfTI 이미지로 저장
    # shutil.copy2(mri_file, os.path.join(dst_folder, 'brain.nii.gz'))
    # shutil.copy2(pet_file, os.path.join(dst_folder, 'realigned_pet.nii.gz'))
    final_img = nib.Nifti1Image(final_data, affine)
    nib.save(final_img, os.path.join(dst_folder, f'{pid}_sub_{postfix}.nii.gz'))
    print(f"created {pid}_sub_{postfix}.nii.gz")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        src_base = str(sys.argv[1])
        postfix = str(sys.argv[2])
        main(src_base, postfix)
    else:
        print("Usage: python script.py <src_base_path> <postfix>")