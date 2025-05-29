import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# 파일 경로 및 시트 이름
file_path = "data/PET Quant Results_2_SD of difference.xlsx"
sheet_name = "SD of Difference"

# 데이터 불러오기 및 전처리
df = pd.read_excel(file_path, sheet_name=sheet_name)
row_labels = df["index"]
df_values_only = df.drop(columns=["index"])
df_transposed = df_values_only.transpose()
df_transposed.columns = row_labels

# 히트맵 그리기
plt.figure(figsize=(8, 8))
ax = sns.heatmap(
    df_transposed,
    xticklabels=True,
    yticklabels=True,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.0,
    center=0.0,
    vmax=1.0,
    linewidths=0.8,
    linecolor='white'
)

# 제목 및 축 라벨 회전
plt.title("Heatmap of SBR Standard Deviation")
plt.yticks(rotation=90, ha='center')
plt.xticks(rotation=0, ha='center')

# ✅ x축 이름 제거
ax.set_xlabel("")  # "index" 제거

# ✅ 컬럼 라벨을 굵게
for label in ax.get_xticklabels():
    label.set_fontweight('bold')

# ✅ colorbar 숫자를 소수점 첫째 자리로 포맷팅
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
plt.show()