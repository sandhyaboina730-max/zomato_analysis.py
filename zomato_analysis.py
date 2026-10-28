# ============================================================
# ZOMATO RESTAURANT DATA ANALYSIS PROJECT
# ============================================================

# Install required libraries if needed
!pip install -q pandas numpy matplotlib seaborn wordcloud

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from IPython.display import display

sns.set_theme(style="whitegrid")

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("/kaggle/input/datasets/bhanupratapbiswas/zomato/zomato.csv", encoding="latin-1")

print("=" * 60)
print("ZOMATO RESTAURANT DATA ANALYSIS")
print("=" * 60)

print("\nOriginal Dataset Shape:", df.shape)
print("\nFirst Five Rows:")
display(df.head())

# Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove duplicate records
df = df.drop_duplicates().copy()

# Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# Replace common missing-value labels
df.replace(
    ["", "null", "NULL", "None", "nan", "N/A", "NaN"],
    np.nan,
    inplace=True
)

# Convert relevant numeric columns
numeric_columns = [
    "aggregate_rating",
    "average_cost_for_two",
    "votes",
    "price_range"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert rating zero to missing if zero means "not rated"
if "aggregate_rating" in df.columns:
    df.loc[
        ~df["aggregate_rating"].between(0, 5),
        "aggregate_rating"
    ] = np.nan

    df.loc[df["aggregate_rating"] == 0, "aggregate_rating"] = np.nan

# Remove records without restaurant names when available
if "restaurant_name" in df.columns:
    df = df[
        df["restaurant_name"].notna()
        & (df["restaurant_name"] != "")
    ].copy()

# Fill missing categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# Save cleaned data
df.to_csv("zomato_cleaned.csv", index=False)
print("\nCleaned dataset saved as zomato_cleaned.csv")

# ============================================================
# 4. RESTAURANT RATING ANALYSIS
# ============================================================

if "aggregate_rating" in df.columns:
    ratings = df["aggregate_rating"].dropna()

    print("\n--- RESTAURANT RATING ANALYSIS ---")
    print("Average Rating:", round(ratings.mean(), 2))
    print("Highest Rating:", ratings.max())
    print("Lowest Rating:", ratings.min())
    print("Total Rated Restaurants:", len(ratings))

    plt.figure(figsize=(9, 5))
    sns.histplot(ratings, bins=20, kde=True)
    plt.title("Distribution of Restaurant Ratings")
    plt.xlabel("Restaurant Rating")
    plt.ylabel("Number of Restaurants")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=ratings)
    plt.title("Restaurant Rating Distribution")
    plt.xlabel("Rating")
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. POPULAR CUISINE ANALYSIS
# ============================================================

if "cuisines" in df.columns:
    cuisine_data = df.loc[
        df["cuisines"].notna()
        & (df["cuisines"] != "Unknown"),
        "cuisines"
    ]

    cuisine_series = (
        cuisine_data.str.split(",")
        .explode()
        .str.strip()
    )

    cuisine_series = cuisine_series[
        (cuisine_series != "")
        & (cuisine_series.str.lower() != "unknown")
    ]

    top_cuisines = cuisine_series.value_counts().head(10)

    print("\n--- TOP 10 POPULAR CUISINES ---")
    print(top_cuisines)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_cuisines.values,
        y=top_cuisines.index
    )
    plt.title("Top 10 Popular Cuisines")
    plt.xlabel("Number of Restaurant Listings")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.show()

# ============================================================
# 6. LOCATION HOTSPOT ANALYSIS
# ============================================================

location_col = next(
    (col for col in ["city", "locality"]
     if col in df.columns),
    None
)

if location_col:
    location_data = df[
        df[location_col].notna()
        & (df[location_col] != "Unknown")
    ]

    top_locations = location_data[location_col].value_counts().head(10)

    print("\n--- TOP 10 RESTAURANT LOCATIONS ---")
    print(top_locations)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_locations.values,
        y=top_locations.index
    )
    plt.title("Top 10 Restaurant Locations")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("Location")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo city or locality column found.")

# ============================================================
# 7. PRICE VS RATING ANALYSIS
# ============================================================

if {"average_cost_for_two", "aggregate_rating"}.issubset(df.columns):

    price_rating = df[
        ["average_cost_for_two", "aggregate_rating"]
    ].dropna().copy()

    price_rating = price_rating[
        (price_rating["average_cost_for_two"] >= 0)
        & (price_rating["aggregate_rating"] > 0)
    ]

    print("\n--- PRICE VS RATING ANALYSIS ---")

    if len(price_rating) > 1:
        correlation = price_rating[
            "average_cost_for_two"
        ].corr(price_rating["aggregate_rating"])

        print("Price-Rating Correlation:", round(correlation, 3))

        plt.figure(figsize=(9, 5))
        sns.scatterplot(
            data=price_rating,
            x="average_cost_for_two",
            y="aggregate_rating",
            alpha=0.5
        )
        plt.title("Average Cost for Two vs Restaurant Rating")
        plt.xlabel("Average Cost for Two")
        plt.ylabel("Aggregate Rating")
        plt.tight_layout()
        plt.show()

        if correlation > 0.3:
            print("Interpretation: Moderate or strong positive relationship.")
        elif correlation < -0.3:
            print("Interpretation: Moderate or strong negative relationship.")
        else:
            print("Interpretation: Weak linear relationship.")
    else:
        print("Not enough valid price and rating records.")
else:
    print("\nRequired price or rating columns are unavailable.")

# ============================================================
# 8. CUISINE VS RATING ANALYSIS
# ============================================================

if {"cuisines", "aggregate_rating"}.issubset(df.columns):

    cuisine_ratings = df[
        (df["cuisines"] != "Unknown")
        & (df["aggregate_rating"].notna())
        & (df["aggregate_rating"] > 0)
    ].copy()

    cuisine_ratings["cuisine"] = (
        cuisine_ratings["cuisines"].str.split(",")
    )

    cuisine_ratings = cuisine_ratings.explode("cuisine")
    cuisine_ratings["cuisine"] = cuisine_ratings["cuisine"].str.strip()

    cuisine_summary = (
        cuisine_ratings.groupby("cuisine")
        .agg(
            average_rating=("aggregate_rating", "mean"),
            restaurant_count=("aggregate_rating", "size")
        )
    )

    cuisine_summary = cuisine_summary[
        cuisine_summary["restaurant_count"] >= 10
    ].sort_values("average_rating", ascending=False)

    print("\n--- TOP RATED CUISINES (MINIMUM 10 LISTINGS) ---")
    display(cuisine_summary.head(10).round(2))

    if not cuisine_summary.empty:
        top_rated_cuisines = cuisine_summary.head(10).reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_rated_cuisines,
            x="average_rating",
            y="cuisine"
        )
        plt.title("Top Rated Cuisines")
        plt.xlabel("Average Rating")
        plt.ylabel("Cuisine")
        plt.tight_layout()
        plt.show()

# ============================================================
# 9. RATING HEATMAP BY CITY
# ============================================================

if {"city", "aggregate_rating"}.issubset(df.columns):

    heatmap_data = df[
        (df["city"] != "Unknown")
        & (df["aggregate_rating"].notna())
        & (df["aggregate_rating"] > 0)
    ].copy()

    top_cities = heatmap_data["city"].value_counts().head(10).index
    heatmap_data = heatmap_data[
        heatmap_data["city"].isin(top_cities)
    ].copy()

    if not heatmap_data.empty:
        heatmap_data["rating_category"] = pd.cut(
            heatmap_data["aggregate_rating"],
            bins=[0, 2, 3, 4, 5],
            labels=[
                "Low (0-2)",
                "Average (2-3)",
                "Good (3-4)",
                "Excellent (4-5)"
            ],
            include_lowest=True
        )

        rating_heatmap = pd.crosstab(
            heatmap_data["city"],
            heatmap_data["rating_category"]
        )

        print("\n--- CITY VS RATING HEATMAP DATA ---")
        display(rating_heatmap)

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            rating_heatmap,
            annot=True,
            fmt="d",
            cmap="YlGnBu"
        )
        plt.title("Restaurant Rating Categories by City")
        plt.xlabel("Rating Category")
        plt.ylabel("City")
        plt.tight_layout()
        plt.show()

# ============================================================
# 10. CUSTOMER REVIEW WORD CLOUD
# ============================================================

review_col = next(
    (col for col in ["reviews_list", "review", "reviews"]
     if col in df.columns),
    None
)

if review_col:

    review_text = " ".join(
        df[review_col]
        .dropna()
        .astype(str)
        .tolist()
    )

    review_text = re.sub(r"[^A-Za-z\s]", " ", review_text)
    review_text = re.sub(r"\s+", " ", review_text).strip()

    print("\n--- CUSTOMER REVIEW ANALYSIS ---")

    if review_text:
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=150,
            collocations=False
        ).generate(review_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Restaurant Reviews")
        plt.show()

        print("Word cloud generated successfully.")
    else:
        print("Review column has no usable text.")
else:
    print("\nNo customer review text column found.")
    print("A word cloud cannot be generated without review text.")

# ============================================================
# 11. FIVE BUSINESS RECOMMENDATIONS
# ============================================================

recommendations = [
    "1. Improve restaurant discovery using cuisine, location, "
    "price and rating filters.",

    "2. Promote highly rated restaurants through featured listings "
    "and personalized recommendations.",

    "3. Help restaurant partners understand location-based competition "
    "and identify underserved areas.",

    "4. Analyze customer review keywords to identify complaints "
    "and improve service quality.",

    "5. Build partnerships with popular cuisines and restaurants "
    "that maintain good customer ratings."
]

print("\n" + "=" * 60)
print("BUSINESS RECOMMENDATIONS")
print("=" * 60)

for item in recommendations:
    print(item)

# ============================================================
# 12. GENERATE SUMMARY REPORT
# ============================================================

report_lines = []

report_lines.append("ZOMATO RESTAURANT DATA ANALYSIS REPORT")
report_lines.append("=" * 50)
report_lines.append(f"Total records after cleaning: {len(df)}")
report_lines.append(f"Total columns: {len(df.columns)}")

if "aggregate_rating" in df.columns:
    report_lines.append(
        "Average restaurant rating: "
        + str(round(df["aggregate_rating"].mean(), 2))
    )
    report_lines.append(
        "Restaurants with valid ratings: "
        + str(df["aggregate_rating"].notna().sum())
    )

if "cuisines" in df.columns:
    report_lines.append("\nTop 5 cuisines:")

    report_lines.extend(
        cuisine_series.value_counts().head(5).to_string().split("\n")
        if "cuisine_series" in globals()
        else ["Cuisine data unavailable"]
    )

if location_col:
    report_lines.append("\nTop 5 locations:")
    report_lines.extend(
        df.loc[df[location_col] != "Unknown", location_col]
        .value_counts()
        .head(5)
        .to_string()
        .split("\n")
    )

report_lines.append("\nBusiness Recommendations:")
report_lines.extend(recommendations)

with open(
    "zomato_analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(report_lines))

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Files generated:")
print("1. zomato_cleaned.csv")
print("2. zomato_analysis_report.txt")
print("\nSummary Report:\n")
print("\n".join(report_lines))
