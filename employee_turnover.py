#!/usr/bin/env python
# coding: utf-8

# # Programming Environment 

# In[829]:


import sys, platform, pandas as pd, numpy as np
print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("OS:", platform.platform())


# ## A1. Number of Records and Variables

# In[830]:


# initial imports and setup
import pandas as pd
import numpy as np

df = pd.read_csv('employee_turnover_dataset.csv')


# In[831]:


# data shape
rows, cols = df.shape
print(f"Rows: {rows} | Columns: {cols}")


# ## A2. List Variables, Data Types, and Subtypes

# In[832]:


# list all variables
df.info()


# In[833]:


# list data types
df.dtypes


# In[834]:


# check subtypes
subtypes = {
    "EmployeeNumber": "identifier",
    "Age": "numeric-int",
    "Tenure": "numeric-int (continuous years)",
    "Turnover": "binary categorical",
    "HourlyRate": "numeric-float (currency)",
    "HoursWeekly": "numeric-int",
    "CompensationType": "categorical (nominal)",
    "AnnualSalary": "numeric-float (currency, derived)",
    "DrivingCommuterDistance": "numeric-int",
    "JobRoleArea": "categorical (nominal)",
    "Gender": "categorical (nominal)",
    "MaritalStatus": "categorical (nominal)",
    "NumCompaniesPreviouslyWorked": "numeric-int (count, nullable)",
    "AnnualProfessionalDevHrs": "numeric-int (count, nullable)",
    "PaycheckMethod": "categorical (nominal)",
    "TextMessageOptIn": "binary categorical (nullable)"
}

a2 = (
    pd.DataFrame({"variable": df.columns, "dtype": df.dtypes.astype(str)})
    .assign(subtype=lambda x: x["variable"].map(subtypes).fillna("review"))
)
a2


# ## A3. Sample of Observable Values

# In[835]:


df.head()
df.sample(5)


# ## B1. Data Quality Inspection Methods 

# ### Duplicate Entries

# In[836]:


# count the total of duplicate entries
df.duplicated().sum()


# In[837]:


# primary key duplicate check 
pk = "EmployeeNumber"

# quick preliminary check on primary key
if df[pk].is_unique:
    print("Primary key unique: no key-duplicate analysis needed.")
else:
    print("Repeated keys:", df[pk].duplicated().sum())


# In[838]:


# exact whole-row duplicates (beyond the first occurrence)
exact_dupes = df[df.duplicated()]
print(f"Exact row duplicates: {len(exact_dupes)}")
exact_dupes


# In[839]:


# double check the primary key repeates
pk = "EmployeeNumber"
dupe_keys = df[pk][df[pk].duplicated()]
print(f"Repeated keys: {dupe_keys.shape[0]}")
df[df[pk].isin(dupe_keys)].sort_values(pk)


# ### Removing Duplicates

# In[840]:


# set pk once
pk = "EmployeeNumber"

# copy of exact_dupes for audit
exact_dupes = df[df.duplicated()]
print(f"Exact row duplicates: {len(exact_dupes)}")

# Deduplicate
n_before = len(df)

# remove exact whole-row duplicates, keep the first copy
df_dedup = df.drop_duplicates(keep="first").copy()

n_after = len(df_dedup)
removed = n_before - n_after

print(f"Rows after drop_duplicates: {n_after}")
print(f"Removed: {removed} rows")
print(f"Overcount removed: {removed/n_before:.2%}")

# Verify PK uniqueness post-clean
print("PK unique after clean:", df_dedup[pk].is_unique)

# Checkpoint: there should be 0 exact dupes now
print("Exact dupes after clean:", df_dedup.duplicated().sum())

# Save the cleaned file
df_dedup.to_csv("employee_dedup_only.csv", index=False)


# In[841]:


# Is primary key unique?
df_dedup = df_dedup.drop_duplicates(subset=[pk], keep="first")
print("PK unique?", df_dedup[pk].is_unique)


# The file after removing duplicates is df_dedup. The next step will copy df_dedup and work from there on the next steps.

# ### Missing Values

# Make a copy of df_dedup and store it as df_missing. For this stage I work with df_missing data frame.

# In[842]:


# Start with a copy of deduplicated df_dedup
df_missing = df_dedup.copy()


# In[843]:


# missing by column
print(df_missing.isna().sum().sort_values(ascending=False))


# In[844]:


# missing by rows
rows_with_missing = df_missing[df_missing.isna().any(axis=1)]
print(rows_with_missing.shape)


# ### Handling Missing Values in TextMessagesOptIn

# In[887]:


text_optin_col = "TextMessageOptIn"

# 1) Normalize values to valid categories
s = df_missing[text_optin_col].astype("string").str.strip().str.title()
s = s.where(s.isin(["Yes", "No"]))   # invalid/other -> NaN

# 2) Assign back to the main column
df_missing[text_optin_col] = s

# 3) Add missingness flag
df_missing["TextMessageOptInMissingFlag"] = s.isna().astype("int8")

# 4) Numeric view
df_missing["TextMessageOptIn_bool"] = s.map({"Yes": 1, "No": 0}).astype("Int8")

# 5) Derived filled column (for charts/exports only)
df_missing["TextMessageOptIn_filled"] = s.fillna("Missing").astype("category")

# 6) Check to make sure counts line up
assert df_missing["TextMessageOptInMissingFlag"].sum() == int(s.isna().sum())


# In[846]:


cols_to_show = ["EmployeeNumber", "TextMessageOptIn", "TextMessageOptIn_filled", "TextMessageOptInMissingFlag"]
df_missing[cols_to_show].head(10)


# In[847]:


# Count total values in TextMessagesOptIn column
df_missing["TextMessageOptIn"].value_counts()
print("Total rows:", len(df_missing))


# ### Handling Missing Values in AnnualProfessionalDevHours 

# In[848]:


# AnnualProfessionalDevHrs: create a stable flag and fill missing hours

#OPTION 1 (impute structural zero)

col  = "AnnualProfessionalDevHrs"
flag = "DevHoursMissingFlag"

# 1) Mark which rows are empty before changing anything
na_mask = df_missing[col].isna()

# 2) Add a flag column once: 1 = was empty originally, 0 = had a value

if flag not in df_missing.columns:
    df_missing[flag] = na_mask.astype(int)

# 3) Fill only those originally empty rows with 0 hours

df_missing.loc[na_mask, col] = 0.0

# 4) Store as a float so partial hours like 1.5 are allowed
df_missing[col] = df_missing[col].astype("Float64")

# 5) Quick sanity print
print({
    "flag_ones": int(df_missing[flag].sum()),
    "remaining_nans": int(df_missing[col].isna().sum())
})


# In[849]:


# OPTION 2: make a median-imputed copy for sensitivity analysis

imputed_col = f"{col}_imputed_median"

# median of the ORIGINAL observed values 
med = df_missing.loc[~na_mask, col].median()

# start from current col, which has 0.0 for originally-missing rows per Option 1
df_missing[imputed_col] = df_missing[col].copy()

# replace only the rows that were originally missing with the median
df_missing.loc[na_mask, imputed_col] = med
df_missing[imputed_col] = df_missing[imputed_col].astype("Float64")

print({
    "option2_column": imputed_col,
    "median_used": float(med) if med == med else None,  # simple NaN-safe cast
    "option2_remaining_nans": int(df_missing[imputed_col].isna().sum())
})


# ### Missing Values in NumCompaniesPreviouslyWorked

# In[888]:


# NumCompaniesPreviouslyWorked (stable flag and median fill)

col  = "NumCompaniesPreviouslyWorked"
flag = "NumCompaniesMissingFlag"

# Snapshot which rows are empty now
na_mask = df_missing[col].isna()

# Flag once: 1 = was empty, 0 = had a value
if flag not in df_missing.columns:
    df_missing[flag] = na_mask.astype(int)

# Fill only the originally empty rows with the median (robust to outliers)
med = df_missing[col].median()
df_missing.loc[na_mask, col] = med

# Store as whole-number count (nullable int)
df_missing[col] = df_missing[col].round().astype("Int64")

# Quick print
print({
    "flag_ones": int(df_missing[flag].sum()),
    "remaining_nans": int(df_missing[col].isna().sum()),
    "median_used": float(med)
})


# ### Inconsitent Entries and Formatting Errors

# In[851]:


# # Safety guard: ensure df_missing exists
try:
    df_missing
except NameError:
    import pandas as pd
    df_dedup = pd.read_csv("employee_dedup_only.csv")
    df_missing = df_dedup.copy()


# #### Whitespaces
# Earlier sampling of the data revealed different ways it stores Mail_Check, Mail Check and other payment types. The code here removes whitespaces and replaces underscores with spaces.

# In[852]:


import re
import pandas as pd

# Normalize whistespaces

def normalize_spaces(x):
    if pd.isna(x):
        return x
    x = re.sub(r"\s+", " ", str(x))
    x = x.replace("_", " ")
    return x.strip()

cat_cols = [
    "CompensationType","JobRoleArea","Gender","MaritalStatus",
    "PaycheckMethod","TextMessageOptIn","Turnover"
]

for c in cat_cols:
    if c in df_missing.columns:
        s = df_missing[c].astype("string")
        s = s.apply(normalize_spaces)
        s = s.replace(r"^\s*$", pd.NA, regex=True)
        df_missing[c] = s.fillna("Unknown").astype("category")


# #### Whitespace inspection after cleaning spaces and underscores

# In[853]:


# INSPECTION: inventory of current categories
print_cols = [
    "TextMessageOptIn","Turnover","Gender","MaritalStatus",
    "CompensationType","PaycheckMethod","JobRoleArea"
]

for c in print_cols:
    if c in df_missing.columns:
        print(f"\n=== {c} ===")
        print(df_missing[c].value_counts(dropna=False))


# #### Since payment categories still show multiple payment types methods (same semantics), the mapping step will fix that issue

# In[854]:


# Canonicalize inconsistent categories within Cell Values

# 1) Define mappings only for values observed
maps = {
    "PaycheckMethod": {
        "DirectDeposit": "Direct Deposit",
        "MailedCheck": "Mailed Check",
        "Mail Check": "Mailed Check",
        # keep "Direct Deposit" and "Mailed Check" as the final forms
    },
    "JobRoleArea": {
        "InformationTechnology": "Information Technology",
        "HumanResources": "Human Resources",
    }
}

for col, mapping in maps.items():
    if col in df_missing.columns:
        df_missing[col] = df_missing[col].replace(mapping)
        df_missing[col] = df_missing[col].astype("category")  # tidy dtype

# 2) Quick checks
check_cols = ["PaycheckMethod","JobRoleArea","CompensationType"]
for c in check_cols:
    if c in df_missing.columns:
        print(f"\n{c} after mapping:")
        print(df_missing[c].value_counts(dropna=False))


# In[855]:


# Cast all cleaned categoricals
for c in cat_cols:
    if c in df_missing.columns:
        df_missing[c] = df_missing[c].astype("category")


# In[856]:


# Allowed sets for key columns
allowed = {
    "TextMessageOptIn": {"Yes","No","Unknown"},
    "Turnover": {"Yes","No"},
    "CompensationType": {"Salary"},
    "PaycheckMethod": {"Direct Deposit","Mailed Check"},
}

for col, ok in allowed.items():
    if col in df_missing.columns:
        current = set(df_missing[col].dropna().unique())
        extras = current - ok
        if extras:
            print(f"[WARN] Unexpected values in {col}: {extras}")


# #### Formatting Fix

# In[857]:


# Strip whitespaces from all column names
df_missing.columns = df_missing.columns.str.strip()


# In[858]:


# Preview column names again 
print(df_missing.columns.tolist())


# In[859]:


# Preview column names to make sure there are no whitespaces
print(df_missing.columns.tolist())


# In[860]:


# Clean $ string from HourlyRate column values
s = df_missing["HourlyRate"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
df_missing["HourlyRate"] = pd.to_numeric(s, errors="coerce").astype("Float64")


# In[861]:


# Save the file "employee_clean_after_inconsistencies"
df_missing.to_csv("employee_clean_after_inconsistencies.csv", index=False)


# #### Check values in the AnnualSalary derived column

# In[862]:


# AnnualSalary must equal HourlyRate * HoursWeekly * 52

HR, HW, SAL = "HourlyRate", "HoursWeekly", "AnnualSalary"
TOL = 1.00  # allow small rounding wiggle room in dollars

# 1) Ensure numerics (in case symbols slipped in)
df_missing[HR] = pd.to_numeric(df_missing[HR], errors="coerce").astype("Float64")
df_missing[HW] = pd.to_numeric(df_missing[HW], errors="coerce").astype("Float64")
df_missing[SAL] = pd.to_numeric(df_missing[SAL], errors="coerce").astype("Float64")

# 2) Compute expected salary from the dictionary definition
expected = (df_missing[HR] * df_missing[HW] * 52).round(2)

# 3) Flag rows that don’t match (or are missing)
mismatch = df_missing[SAL].isna() | (df_missing[SAL] - expected).abs().gt(TOL)

# 3.5) Preview mismatches before fixing (for audit/report)
mismatch_cols = [HR, HW, SAL]
mismatch_preview = (
    df_missing.loc[mismatch, mismatch_cols]
      .assign(expected=expected[mismatch])
      .assign(diff=lambda x: (x[SAL] - x["expected"]).round(2))
)

print("Mismatched salary rows (before fix):", mismatch_preview.shape[0])
print(mismatch_preview.head(10))  # sample for preview

# Diff summary BEFORE correction
print("Diff summary (SAL - expected):")
print(mismatch_preview["diff"].describe())

# 4) Salary audit flag (1 = corrected/replaced)
if "AnnualSalaryCorrected" not in df_missing.columns:
    df_missing["AnnualSalaryCorrected"] = 0
df_missing.loc[mismatch, "AnnualSalaryCorrected"] = 1

# 5) Replace only those mismatches (idempotent)
df_missing.loc[mismatch, SAL] = expected[mismatch]

# 6) Quick sanity summary (read-only)
print({
    "rows": len(df_missing),
    "corrected": int(mismatch.sum()),
    "remaining_mismatches": int((df_missing[SAL] - expected).abs().gt(TOL).sum())
})


# In[863]:


# No negatives or zeros in key fields?
print("HourlyRate <= 0:", int((df_missing["HourlyRate"] <= 0).sum()))
print("HoursWeekly <= 0:", int((df_missing["HoursWeekly"] <= 0).sum()))

# Spot-check a few corrected rows
df_missing.loc[df_missing["AnnualSalaryCorrected"] == 1, 
               ["HourlyRate", "HoursWeekly", "AnnualSalary"]].head()


# In[864]:


# Save to csv
df_missing.to_csv("employee_clean_after_salary_fix.csv", index=False)
print("Saved employee_clean_after_salary_fix.csv")


# After fixing missing values, inconsistent entries, and formatting issues the data frame is "employee_clean_after_salary_fix.csv"

# ### Outliers

# For cleaning outliers in this stage I take "employee_clean_after_salary_fix.csv" and create a new dataframe df_outliers.

# In[865]:


# New Outliers variable
df_outliers = pd.read_csv("employee_clean_after_salary_fix.csv")

# Identify numeric columns
num_cols = df_outliers.select_dtypes(include=["float64", "int64"]).columns
num_cols


# In[866]:


# Get descriptive stats
df_outliers[num_cols].describe().T


# Outliers per variable
# - exclude EmployeeNumber column (identifier - no math needed)
# - exclude flag columns (binary derived columns not necessary here)
# - calculate IQR, LowerFence, Upper Fence, TotalOutliers count per variable
# - add column that adds LowOutliers and HighOutliers that checks whether the total count > 0 for True, and = 0 for False

# In[867]:


# Calculate IQR, LowerFence, UpperFence, TotalOutliers, and OutliersPresent (True, False) column
exclude = {
    "EmployeeNumber",
    "TextMessageOptInMissingFlag",
    "TextMessageOptIn_bool",
    "DevHoursMissingFlag",
    "NumCompaniesMissingFlag",
    "AnnualSalaryCorrected",
}
use_cols = [c for c in df_outliers.columns if c in num_cols and c not in exclude]

desc = df_outliers[use_cols].describe(percentiles=[.25, .5, .75]).T
desc["IQR"] = desc["75%"] - desc["25%"]
desc["LowerFence"] = desc["25%"] - 1.5 * desc["IQR"]
desc["UpperFence"] = desc["75%"] + 1.5 * desc["IQR"]

low_out  = df_outliers[use_cols].lt(desc["LowerFence"], axis=1).sum()
high_out = df_outliers[use_cols].gt(desc["UpperFence"], axis=1).sum()

desc["LowOutliers"] = low_out
desc["HighOutliers"] = high_out
desc["TotalOutliers"] = desc["LowOutliers"] + desc["HighOutliers"]
desc["OutliersPresent"] = desc["TotalOutliers"].gt(0)

desc.loc[:, ["IQR","LowerFence","UpperFence","TotalOutliers","OutliersPresent"]]


# In[868]:


# Print rows with negative commuter distance
df_outliers[df_outliers["DrivingCommuterDistance"] < 0]


# In[869]:


# Coerce to numeric 
driv_com_col = "DrivingCommuterDistance"
df_outliers[driv_com_col] = pd.to_numeric(df_outliers[driv_com_col], errors="coerce")


# In[870]:


# Fix NaN and Negative Values

col = "DrivingCommuterDistance"

# 1) Convert absurd values (negatives) to NaN
neg_mask = df_outliers[col] < 0
df_outliers.loc[neg_mask, col] = pd.NA

# 2) Compute ONE fill value from valid data (median of nonnegative distances)
median_dist = df_outliers.loc[df_outliers[col].ge(0), col].median()

# 3) Impute all NaNs (includes prior negatives) with that median
to_impute = df_outliers[col].isna()
df_outliers[col + "ImputedFlag"] = to_impute.astype(int)
df_outliers.loc[to_impute, col] = median_dist

# 4) Treat true outliers AFTER ] (cap at 99th percentile)
upper_cap = df_outliers[col].quantile(0.99)
out_mask = df_outliers[col] > upper_cap
df_outliers[col + "OutlierFlag"] = out_mask.astype(int)
df_outliers.loc[out_mask, col] = upper_cap

# 5) Quick summary
print({
    "median_used": float(median_dist),
    "negatives_were": int(neg_mask.sum()),
    "imputed_count": int(to_impute.sum()),
    "capped_outliers": int(out_mask.sum()),
})


# In[871]:


# Save the cleaned dataset
df_outliers.to_csv("employee_clean_after_outlier_fix.csv", index=False)

print("Saved employee_clean_after_outlier_fix.csv")


# The file containing outlier fixes is employee_clean_after_outlier_fix.csv. I save it as df_final for next steps - final checks.

# In[872]:


# Save final, cleaned data to csv
df_final = pd.read_csv("employee_clean_after_outlier_fix.csv")
df_final.info()


# # FINAL CHECKS

# ## 1. Final Missing Values Count

# In[873]:


# Check final count of missing values
df_final.isna().sum()


# NOTE: ***TextMessageOptIn_bool*** - only used for analysis/modeling

# In[874]:


# Example use of TextMessageOptIn_bool in Analysis (gives the opt-in rate)
opt_in_rate = df_final["TextMessageOptIn_bool"].mean()   # uses only Yes/No
rate_by_role = df_final.groupby("JobRoleArea")["TextMessageOptIn_bool"].mean()
print(f"Overall opt-in rate: {opt_in_rate:.3f}")
print(rate_by_role.sort_values(ascending=False))


# ## 2. Final Duplicate Check

# In[875]:


# Duplicate check
df_final.duplicated().sum()


# ## 3. Final Statistical Check

# In[876]:


# Statistical check
df_final.describe().T


# ## 4. Final Validity Check for Distances

# In[877]:


# Verify all commuter distances are valid
(df_final["DrivingCommuterDistance"] < 0).sum()


# ## 5. Final Verification of Flagged Rows

# In[878]:


# Verify flagged rows 
df_final["DrivingCommuterDistanceImputedFlag"].value_counts()


# ## 6. Final Checks for Imputed Distance Rows

# In[879]:


# How many rows were imputed and what values they have now
imputed_mask = df_outliers[col + "ImputedFlag"] == 1
df_final.loc[imputed_mask, col].describe()
df_final.loc[imputed_mask, col].value_counts().head()

# How many rows had the median imputed for distance
print("Median used for imputation:", median_dist)

# Confirm all imputed rows equal the median
all_equal_median = (df_final.loc[imputed_mask, col] == median_dist).all()
print("All imputed rows equal median?", bool(all_equal_median))


# ## 7. Final Count for Rows in DrivingCommuterDistance that have 0 value

# In[880]:


(df_final["DrivingCommuterDistanceImputedFlag"] == 0).sum()


# ## 8. Final Confirmation That Code did not Immpute 0

# In[881]:


imputed_zero_ct = (
    (df_final["DrivingCommuterDistance"] == 0)
    & (df_final["DrivingCommuterDistanceImputedFlag"] == 1)
).sum()

print("Imputed zeros:", imputed_zero_ct)


# ## 9. How many NaN values did the original data have?

# In[882]:


df.isna().sum()


# ## 10. How many NaN values does the final data have?

# In[883]:


df_final.isna().sum()


# # 11. Original Data Summary

# In[884]:


df[sorted(df.columns)].info()


# # 12. Final Clean Data Structural Summary

# In[885]:


exclude = {
    "TextMessageOptInMissingFlag",
    "TextMessageOptIn_bool",
    "TextMessageOptIn_filled",
    "DevHoursMissingFlag",
    "NumCompaniesMissingFlag",
    "AnnualSalaryCorrected",
    "DrivingCommuterDistanceImputedFlag",
    "DrivingCommuterDistanceOutlierFlag",
}
df_final[df_final.columns.difference(exclude)].info()

