# build dockerfile


# fsl flirt and fnirt

### flirt
```
docker run \
-v /home/cmc_admin/workspace/fedpod/data256_cc359ppmicmc_newseg/MNI152:/src \
-v /home/cmc_admin/workspace/fedpod/data256_cc359ppmicmc_newseg/inst_05:/ref \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt:/dst \
dwnusa/myfsl:v1.0 /bin/bash -c \
'
for i in {1..359}; \
do \
SUBJ=$(printf "cc%04d" $i); \
echo ${SUBJ}; \
mkdir -p /dst/${SUBJ}; \
flirt \
-in /src/mni152_t1.nii.gz \
-ref /ref/${SUBJ}/${SUBJ}_t1.nii.gz \
-out /dst/${SUBJ}/${SUBJ}_mni_to_t1_flirt.nii.gz \
-omat /dst/${SUBJ}/mni_to_t1_flirt.mat \
-cost corratio \
-dof 6 \
-interp trilinear; \
done \
'
```
### flirt apply to mask
```
docker run \
-v /home/cmc_admin/workspace/fedpod/data256_cc359ppmicmc_newseg/MNI152:/src \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/ref \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt:/mat \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt_sub:/dst \
dwnusa/myfsl:v1.0 /bin/bash -c \
'
for i in {1..359}; \
do \
SUBJ=$(printf "cc%04d" $i); \
echo ${SUBJ}; \
mkdir -p /dst/${SUBJ}; \
flirt \
-in /src/mni152_sub.nii.gz \
-ref /ref/${SUBJ}/${SUBJ}_sub.nii.gz \
-applyxfm \
-init /mat/${SUBJ}/mni_to_t1_flirt.mat \
-out /dst/${SUBJ}/${SUBJ}_flirt_sub.nii.gz \
-interp nearestneighbour; \
done \
'
```

### fnirt
```
docker run \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/src \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/ref \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_fnirt:/dst \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt:/mat \
dwnusa/myfsl:v1.0 /bin/bash -c \
'
for i in {1..359}; \
do \
SUBJ=$(printf "cc%04d" $i); \
echo ${SUBJ}; \
mkdir -p /dst/${SUBJ}; \
fnirt \
--in=/src/mni152_t1.nii.gz \
--ref=/ref/${SUBJ}/${SUBJ}_t1.nii.gz \
--cout=/dst/${SUBJ}/mni_to_t1_fnirt.nii.gz \
--aff=/mat/${SUBJ}/mni_to_t1_flirt.mat \
--iout=/dst/${SUBJ}/${SUBJ}_mni_to_t1_fnirt.nii.gz \
--warpres=4,4,4; \
done \
'
```


##### fnirt with inmask and refmask
```
docker run \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/src \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/ref \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_fnirt:/dst \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt:/mat \
dwnusa/myfsl:v1.0 /bin/bash -c \
'
for i in 1; \
do \
SUBJ=$(printf "cc%04d" $i); \
echo ${SUBJ}; \
mkdir -p /dst/${SUBJ}; \
fslmaths /src/mni152_sub.nii.gz -bin /src/mni152_mask.nii.gz -odt char; \
done \
'
```
```
docker run \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/src \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/ref \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_fnirt:/dst \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt:/mat \
dwnusa/myfsl:v1.0 /bin/bash -c \
'
for i in 1; \
do \
SUBJ=$(printf "cc%04d" $i); \
echo ${SUBJ}; \
mkdir -p /dst/${SUBJ}; \
fnirt \
--in=/src/mni152_t1.nii.gz \
--ref=/ref/${SUBJ}/${SUBJ}_t1.nii.gz \
--cout=/dst/${SUBJ}/mni_to_t1_fnirt.nii.gz \
--aff=/mat/${SUBJ}/mni_to_t1_flirt.mat \
--iout=/dst/${SUBJ}/${SUBJ}_mni_to_t1_fnirt.nii.gz \
--warpres=2,2,2 \
--inmask=/src/mni152_mask.nii.gz \
--refmask=/ref/${SUBJ}/${SUBJ}_mask.nii.gz; \
done \
'
```


### fnirt apply to mask
```
docker run \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/src \
-v /home/cmc_admin/workspace/fedpod/data256_cc359:/ref \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_flirt:/flirt \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_fnirt:/fnirt \
-v /home/cmc_admin/workspace/fedpod/data256_cc359_fnirt_sub:/dst \
dwnusa/myfsl:v1.0 /bin/bash -c \
'
for i in {1..359}; \
do \
SUBJ=$(printf "cc%04d" $i); \
echo ${SUBJ}; \
mkdir -p /dst/${SUBJ}; \
applywarp \
--in=/src/mni152_sub.nii.gz \
--ref=/ref/${SUBJ}/${SUBJ}_t1.nii.gz \
--warp=/fnirt/${SUBJ}/mni_to_t1_fnirt.nii.gz \
--out=/dst/${SUBJ}/${SUBJ}_fnirt_sub.nii.gz \
--interp=nn; \
done \
' 
```