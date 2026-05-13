import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap
from IPython.display import IFrame, display
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("dft-road-casualty-statistics-collision-2024.csv", low_memory=False)
# print(df.info())
# print(df.head(5))
print("\nColumn Names:")
print(df.columns.tolist())

# print(df.isnull().sum())
df['severity_name'] = df['collision_severity'].map({1: 'Fatal', 2: 'Serious', 3: 'Slight'})


cols_to_drop = [
 "collision_index",
 "collision_ref_no",
 "location_easting_osgr",
 "location_northing_osgr",
 "police_force",
 "local_authority_district",
 "local_authority_ons_district",
 "local_authority_highway",
 "local_authority_highway_current",
 "first_road_class",
 "first_road_number",
 "junction_detail_historic",
 "junction_control",
 "second_road_class",
 "second_road_number",
 "pedestrian_crossing_human_control_historic",
 "pedestrian_crossing_physical_facilities_historic",
 "pedestrian_crossing",
 "carriageway_hazards_historic",
 "carriageway_hazards",
 "did_police_officer_attend_scene_of_accident",
 "trunk_road_flag",
 "lsoa_of_accident_location",
 "enhanced_severity_collision",
 "collision_injury_based",
 "collision_adjusted_severity_serious",
 "collision_adjusted_severity_slight"
]

df = df.drop(columns=cols_to_drop, errors="ignore")

print("Cleaned Column Count:", len(df.columns))
print("Remaining Columns:")
print(df.columns)

df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['month'] = df['date'].dt.month

df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce')
df['hour'] = df['time'].dt.hour

# sns.countplot(x='collision_severity', data=df)
sns.countplot(x='severity_name', data=df, order=['Fatal', 'Serious', 'Slight'])
plt.title("Accident Severity Distribution")
plt.savefig("severity_distribution.png")
plt.show()

sns.countplot(x='month', data=df)
plt.title("Accidents Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.savefig("accidents_per_month.png")
plt.show()

# sns.countplot(x='weather_conditions', hue='collision_severity', data=df)
sns.countplot(x='weather_conditions', hue='severity_name', data=df, hue_order=['Fatal', 'Serious', 'Slight'])
plt.title("Weather Conditions vs Collision Severity")
plt.xlabel("Weather Conditions")
plt.ylabel("Accident Count")
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8],
           labels=['Fine', 'Raining', 'Snowing', 'Fine+Winds', 'Rain+Winds',
                   'Snow+Winds', 'Fog/Mist', 'Other', 'Unknown'],
           rotation=45, ha='right')

plt.tight_layout()
plt.savefig("weather_vs_severity.png")
plt.show()

numeric_cols = [
    'collision_severity',
    'month',
    'hour',
    'speed_limit',
    'road_type',
    'light_conditions',
    'weather_conditions',
    'road_surface_conditions'
]

# sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.savefig("correlation_heatmap.png")
# plt.show()


X = df[
    [
        "weather_conditions",
        "light_conditions",
        "road_type",
        "speed_limit",
        "road_surface_conditions",
        "hour",
        "month",
        "urban_or_rural_area",
        "number_of_vehicles",
        "number_of_casualties",
    ]
]
y = df["collision_severity"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Fatal', 'Serious', 'Slight'],
            yticklabels=['Fatal', 'Serious', 'Slight'])
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Fatal', 'Serious', 'Slight']))


importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.plot(kind="bar")
plt.title("Decision Tree - Feature Importance")
plt.ylabel("Importance")
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(18, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    fontsize=7
)
plt.title("Decision Tree Structure")
plt.savefig("decision_tree_plot.png", bbox_inches="tight")
plt.show()




df_hot = df[["date", "latitude", "longitude"]].dropna().copy()
df_hot["date"] = pd.to_datetime(df_hot["date"], dayfirst=True, errors="coerce")
df_hot = df_hot.dropna(subset=["date"])


df_hot["month_period"] = df_hot["date"].dt.to_period("M")


GRID = 0.02
df_hot["lat_zone"] = (df_hot["latitude"] / GRID).round().astype(int)
df_hot["lon_zone"] = (df_hot["longitude"] / GRID).round().astype(int)
df_hot["zone_id"] = df_hot["lat_zone"].astype(str) + "_" + df_hot["lon_zone"].astype(str)


zone_month = (
    df_hot.groupby(["zone_id", "month_period"])
          .size()
          .reset_index(name="acc_count")
          .sort_values(["zone_id", "month_period"])
)


zone_month["lag_1"] = zone_month.groupby("zone_id")["acc_count"].shift(1)

zone_month["roll3_mean"] = (
    zone_month.groupby("zone_id")["acc_count"]
              .shift(1)
              .rolling(3)
              .mean()
              .reset_index(level=0, drop=True)
)

zone_month[["lag_1", "roll3_mean"]] = zone_month[["lag_1", "roll3_mean"]].fillna(0)


zone_month["month_num"] = zone_month["month_period"].dt.month


n_test_months = 6

all_months = sorted(zone_month["month_period"].unique())


if len(all_months) <= n_test_months + 1:
    raise ValueError(
        f"Not enough months ({len(all_months)}) for {n_test_months}-month forecasting. "
        "Use more data or reduce n_test_months."
    )

train_months = all_months[:-n_test_months]
test_months  = all_months[-n_test_months:]


train_df = zone_month[zone_month["month_period"].isin(train_months)].copy()
test_df  = zone_month[zone_month["month_period"].isin(test_months)].copy()

X_train = train_df[["lag_1", "roll3_mean", "month_num"]]
y_train = train_df["acc_count"]

X_test = test_df[["lag_1", "roll3_mean", "month_num"]]
y_test = test_df["acc_count"]


reg = DecisionTreeRegressor(max_depth=6, random_state=42)
reg.fit(X_train, y_train)

test_df["pred_count"] = reg.predict(X_test)


target_month = test_months[-1]
pred_month_df = test_df[test_df["month_period"] == target_month].copy()


quantile_cut = 0.95
threshold = pred_month_df["pred_count"].quantile(quantile_cut)
hotspots = pred_month_df[pred_month_df["pred_count"] >= threshold].copy()


parts = hotspots["zone_id"].str.split("_", expand=True).astype(int)
hotspots["lat_center"] = parts[0] * GRID
hotspots["lon_center"] = parts[1] * GRID




pred_month_df = pred_month_df.copy()
pred_month_df["error"] = pred_month_df["pred_count"] - pred_month_df["acc_count"]
pred_month_df["abs_error"] = pred_month_df["error"].abs()


MAE = pred_month_df["abs_error"].mean()
RMSE = np.sqrt((pred_month_df["error"] ** 2).mean())

print(f"Forecast month: {target_month}")
print(f"Zone-level MAE : {MAE:.3f}")
print(f"Zone-level RMSE: {RMSE:.3f}")


topN = 10
top_zones = pred_month_df.sort_values("pred_count", ascending=False).head(topN).copy()


top_zones = top_zones.reset_index(drop=True)
top_zones["zone_rank"] = [f"Zone {i+1}" for i in range(len(top_zones))]


x = np.arange(len(top_zones))
width = 0.38

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, top_zones["acc_count"], width, label="Actual")
plt.bar(x + width/2, top_zones["pred_count"], width, label="Predicted")
plt.xticks(x, top_zones["zone_rank"])
plt.title(f"Actual vs Predicted Accident Counts (Top {topN} Zones) — {target_month}")
plt.xlabel("Zone Rank (Zone 1 = highest predicted risk)")
plt.ylabel("Accident Count")
plt.legend()
plt.tight_layout()
plt.savefig(f"actual_vs_predicted_top{topN}_{str(target_month)}.png", dpi=300)
plt.show()


plt.figure(figsize=(12, 5))
plt.bar(top_zones["zone_rank"], top_zones["abs_error"])
plt.title(f"Prediction Error (Absolute Error) — Top {topN} Zones — {target_month}")
plt.xlabel("Zone Rank (Zone 1 = highest predicted risk)")
plt.ylabel("Absolute Error |Predicted − Actual|")
plt.tight_layout()
plt.savefig(f"error_chart_top{topN}_{str(target_month)}.png", dpi=300)
plt.show()


print("\nTop zones (ranked):")
print(top_zones[["zone_rank", "zone_id", "acc_count", "pred_count", "abs_error"]].to_string(index=False))
print("\n  Generating Hotspot Map...")


pred_map = folium.Map(location=[54.5, -2.0], zoom_start=6, tiles="OpenStreetMap")


HeatMap(
    hotspots[["lat_center", "lon_center", "pred_count"]].values.tolist(),
    radius=12,
    blur=18,
    max_zoom=1
).add_to(pred_map)


from IPython.display import display

display(pred_map)
print(" Map displayed inline in notebook")


# GRAPH 3: Actual vs Predicted totals
month_compare = (
    test_df.groupby("month_period")[["acc_count", "pred_count"]]
          .sum()
          .reset_index()
          .sort_values("month_period")
)


month_compare["error"] = month_compare["pred_count"] - month_compare["acc_count"]

plt.figure(figsize=(9, 5))
plt.plot(month_compare["month_period"].astype(str), month_compare["acc_count"], marker="o", label="Actual Total")
plt.plot(month_compare["month_period"].astype(str), month_compare["pred_count"], marker="o", label="Predicted Total")
plt.title("Actual vs Predicted Total Accidents (Test Months)")
plt.xlabel("Month")
plt.ylabel("Total Accidents (All Zones Combined)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)


for i, row in month_compare.iterrows():
    month = str(row["month_period"])
    err = row["error"]
    plt.text(i, row["pred_count"], f"{err:+.0f}", fontsize=9, ha="center", va="bottom")

plt.tight_layout()
plt.savefig("actual_vs_predicted_totals_test_months.png", dpi=300)
plt.show()
print("\n[Graph 3 Month-by-month totals]")
print(month_compare[["month_period", "acc_count", "pred_count", "error"]])


max_v = max(pred_month_df["acc_count"].max(), pred_month_df["pred_count"].max())
plt.plot([0, max_v], [0, max_v], linestyle="--")
plt.tight_layout()
plt.savefig(f"pred_vs_actual_{str(target_month)}.png", dpi=300)
plt.show()



geo_uk = df[["latitude", "longitude"]].dropna().copy()
geo_uk = geo_uk.sample(n=min(20000, len(geo_uk)), random_state=42)
kmeans_uk = KMeans(n_clusters=5, random_state=42, n_init=10)
geo_uk["uk_cluster"] = kmeans_uk.fit_predict(geo_uk[["latitude", "longitude"]])


# UK scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x="longitude", y="latitude",
    hue="uk_cluster", data=geo_uk,
    palette="tab10", s=10
)
plt.title("UK Road Accident Hotspot Clusters (K-Means, Sample)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster ID")
plt.savefig("uk_kmeans_clusters.png", bbox_inches="tight")
plt.show()


# UK heatmap
uk_map = folium.Map(tiles="OpenStreetMap")
uk_map.fit_bounds([[geo_uk["latitude"].min(), geo_uk["longitude"].min()],
                   [geo_uk["latitude"].max(), geo_uk["longitude"].max()]])
HeatMap(geo_uk[["latitude","longitude"]].values.tolist(), radius=6).add_to(uk_map)
uk_map.save("uk_heatmap.html")
print("Saved: uk_heatmap.html")







