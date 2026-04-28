import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Load reaction-time CSV
df = pd.read_csv(
    r"C:\Users\User\Desktop\דברים\לימודים\שנה ג\מבוא ללמידה עמוקה\Assignment_1\cifar-10h-master\data\cifar10h-raw\cifar10h-raw.csv"
)
df = df[df["reaction_time"] <= 40000]

# Load model probabilities file
model_probs = np.load("model_true_label_probs.npy")

# Compute average reaction time per image
avg_reaction_time = df.groupby("cifar10_test_test_idx")["reaction_time"].mean()

# Keep only CIFAR-10 test images 0–9999 and sort by image index
avg_reaction_time = avg_reaction_time.loc[0:9999].sort_index()
reaction_times = avg_reaction_time.values

print("Number of reaction times:", len(reaction_times))
print("Number of model probabilities:", len(model_probs))

# Correlation metrics
pearson_corr, pearson_p = pearsonr(model_probs, reaction_times)
spearman_corr, spearman_p = spearmanr(model_probs, reaction_times)

print("\n===== Correlation Results =====")
print(f"Pearson correlation:  {pearson_corr:.4f}")
print(f"Pearson p-value:      {pearson_p:.4e}")

print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Spearman p-value:     {spearman_p:.4e}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(model_probs, reaction_times, alpha=0.3)
plt.xlabel("Model probability for true label")
plt.ylabel("Average human reaction time")
plt.title("Model Confidence vs. Human Reaction Time")
plt.grid(True)


# --- הוספה לקוד הקיים שלך לפני plt.show() ---

# 1. הגדרת ה"סלים" (Bins) על ציר ה-X
# מחלקים את הטווח [0, 1] ל-20 חלקים שווים
n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2 # מרכזי הסלים עבור הציור

# 2. חישוב ממוצע Y לכל סל
bin_means = []
for i in range(n_bins):
    # מוצאים את כל האינדקסים שנופלים בטווח של הסל הנוכחי
    mask = (model_probs >= bins[i]) & (model_probs < bins[i+1])
    if mask.any():
        bin_means.append(reaction_times[mask].mean())
    else:
        bin_means.append(np.nan) # אם אין נתונים בסל נשים NaN

# 3. הוספת הגרף הממוצע על גבי ה-Scatter
plt.plot(bin_centers, bin_means, color='red', linewidth=3, label='Average Trend')
plt.legend() # הוספת מקרא כדי שנזהה את השורה
# -------------------------------------------

plt.show()