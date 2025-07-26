
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from shapely.geometry import Point
import geopandas as gpd
from sklearn.preprocessing import MultiLabelBinarizer

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


# Create output directory if it doesn't exist
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# # Load data
# sold01 = pd.read_csv('./dataset/CRMLSSold202412.csv')
# sold02 = pd.read_csv('./dataset/CRMLSSold202501_filled.csv')
# sold03 = pd.read_csv('./dataset/CRMLSSold202502.csv')
# sold04 = pd.read_csv('./dataset/CRMLSSold202503.csv')
# sold05 = pd.read_csv('./dataset/CRMLSSold202504.csv')
# sold06 = pd.read_csv('./dataset/CRMLSSold202505.csv')

# # Combine all datasets
# all_data = pd.concat([sold01, sold02, sold03, sold04, sold05, sold06], ignore_index=True)

# Load test dataset
all_data = pd.read_csv('./dataset/CRMLSSold202506.csv')

# Reorder columns so first column is ClosePrice
cols_first = ['ClosePrice']
all_columns = cols_first + [col for col in all_data.columns if col not in cols_first]
all_data = all_data[all_columns]

# Filter for target property type based on instruction and remove rows with missing ClosePrice
all_data = all_data[
    (all_data['PropertyType'] == 'Residential') &
    (all_data['PropertySubType'] == 'SingleFamilyResidence') &
    (all_data['ClosePrice'].notna())
]

# Remove possible duplicates - Jun
all_data = all_data.drop_duplicates()

# Drop irrelevant columns
features_to_drop = [
    # Agent/listing-related
    'ListingKey', 'ListingKeyNumeric', 'ListingId', 'MlsStatus',
    'OriginalListPrice', 'DaysOnMarket', 'ListAgentFullName',
    'ListAgentFirstName', 'ListAgentLastName', 'ContractStatusChangeDate',
    'PurchaseContractDate', 'ListingContractDate', 'BusinessType',
    'ListAgentEmail', 'ListPrice', 'ListAgentAOR', 'BuyerAgentFirstName',
    'BuyerAgentLastName', 'BuyerAgentMlsId', 'BuyerAgentAOR', 'BuyerOfficeAOR',
    'BuyerOfficeName', 'CoListOfficeName', 'CoListAgentFirstName',
    'CoListAgentLastName', 'CoBuyerAgentFirstName', 'ListOfficeName',

    # Unused/spatial/derived
    'latfilled', 'lonfilled', 'LotSizeAcres', 'LotSizeArea',
    'LotSizeDimensions', 'AssociationFeeFrequency', 'TaxAnnualAmount',
    'AboveGradeFinishedArea', 'BelowGradeFinishedArea', 'MainLevelBedrooms', 'BuilderName',
    'SubdivisionName', 'TaxYear', 'AssociationFee', 'MLSAreaMajor',
    'WaterfrontYN', 'BasementYN', 'FireplacesTotal', 'BuildingAreaTotal', 'CoveredSpaces',
    'ElementarySchool', 'ElementarySchoolDistrict', 'MiddleOrJuniorSchool', 'MiddleOrJuniorSchoolDistrict','HighSchool', 'HighSchoolDistrict', 'Levels'
]

# Drop columns from the DataFrame
all_data = all_data.drop(columns=[col for col in features_to_drop if col in all_data.columns])

# Processing rows with missing zip, Latitude, and Longitude - Hazel & Jun
missing_loc_data = all_data[(all_data['Latitude'].isna()) | 
                            (all_data['Longitude'].isna()) |
                            (all_data['PostalCode'].str.len() != 5)]

addresses = missing_loc_data.loc[:, ['UnparsedAddress', 'City', 'StateOrProvince']]
addresses['FullAddress'] = addresses[['UnparsedAddress', 'City', 'StateOrProvince']] \
    .apply(lambda row: ' '.join([str(x).strip() for x in row if pd.notna(x)]), axis=1)

# Geocoding to fill missing 
geolocator = Nominatim(user_agent="idx_cleaning_script", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5, max_retries=1, error_wait_seconds=0.5)

lat = {}
lon = {}
zip_code = {}
city = {}

for i in addresses['FullAddress']:
    try:
        location = geocode(i)
        if location is not None:
            lat[i] = location.latitude
            lon[i] = location.longitude

            raw_data = location.raw
            if 'address' in raw_data:
                zip_code[i] = raw_data['address'].get('postcode', '')
                city[i] = raw_data['address'].get('city', '')
            else:
                zip_code[i] = ''
                city[i] = ''
        else:
            print(f"Address not found: {i}")
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding timed out for: {i} – {e}")
        lat[i] = ''
        lon[i] = ''
        zip_code[i] = ''
        city[i] = ''

all_data['Latitude'] = all_data['Latitude'].fillna(all_data['UnparsedAddress'].map(lat))
all_data['Longitude'] = all_data['Longitude'].fillna(all_data['UnparsedAddress'].map(lon))
all_data['PostalCode'] = all_data['PostalCode'].fillna(all_data['UnparsedAddress'].map(zip_code))
all_data['City'] = all_data['City'].fillna(all_data['UnparsedAddress'].map(city))

# Drop rows where geocoding failed
all_data = all_data.dropna(subset=['Latitude', 'Longitude'])
all_data = all_data.dropna(subset=['PostalCode', 'City'])

# Remove outliers in isolated areas of California - Jun
all_data = all_data[(all_data['Latitude']>30)&(all_data['Longitude']<-50)]

# Macro Features - Tara
# including: Unemployment Rate, Inflation Rate, Interest Rate, CA Sales tax, Mortgage Rate 30 fixed
data_unemploy = pd.read_csv("./dataset/UNRATE.csv")
data_cpi = pd.read_csv("./dataset/CPIAUCNS.csv")
data_mortgage30us = pd.read_csv("./dataset/MORTGAGE30US.csv")
data_fedfunds = pd.read_csv("./dataset/FEDFUNDS.csv")
data_salestax = pd.read_csv("./dataset/SalesTaxRates.csv")

# CA Sales Tax
city_rate_map = data_salestax[data_salestax['Type'] == 'City'].set_index('Location')['Rate'].to_dict()
county_rate_map = data_salestax[data_salestax['Type'] == 'County'].set_index('County')['Rate'].to_dict()

def get_sales_tax_rate(row):
    city = row['City']
    county = row['CountyOrParish']
    # city match first
    if city in city_rate_map:
        return city_rate_map[city]
    # then county match
    elif county in county_rate_map:
        return county_rate_map[county]
    else:
        return 0.0725  # CA statewide tax rate
all_data['SalesTaxRate'] = all_data.apply(get_sales_tax_rate, axis=1)

all_data['CloseDate_Parsed'] = pd.to_datetime(all_data['CloseDate'], errors='coerce')
all_data['CloseDate_YearMonth'] = all_data['CloseDate_Parsed'].dt.strftime('%Y-%m')

# FedInterestRate
data_fedfunds['observation_date'] = pd.to_datetime(data_fedfunds['observation_date'])
data_fedfunds['CloseDate_YearMonth'] = data_fedfunds['observation_date'].dt.strftime('%Y-%m')
data_fedfunds = data_fedfunds.rename(columns={'FEDFUNDS': 'FedInterestRate'})

# Unemployment
data_unemploy['observation_date'] = pd.to_datetime(data_unemploy['observation_date'])
data_unemploy['CloseDate_YearMonth'] = data_unemploy['observation_date'].dt.strftime('%Y-%m')
data_unemploy = data_unemploy.rename(columns={'UNRATE': 'UnemploymentRate'})

# CPI
data_cpi['observation_date'] = pd.to_datetime(data_cpi['observation_date'])
data_cpi['CloseDate_YearMonth'] = data_cpi['observation_date'].dt.strftime('%Y-%m')
data_cpi = data_cpi.rename(columns={'CPIAUCNS': 'CPI'})

# Mortgage rate
data_mortgage30us['observation_date'] = pd.to_datetime(data_mortgage30us['observation_date'])
data_mortgage30us['CloseDate_YearMonth'] = data_mortgage30us['observation_date'].dt.strftime('%Y-%m')
data_mortgage30us = data_mortgage30us.rename(columns={'MORTGAGE30US': 'MortgageRate30Fixed'})

# Merge all macro data on CloseDate_YearMonth
all_data = all_data.merge(data_unemploy[['CloseDate_YearMonth', 'UnemploymentRate']], on='CloseDate_YearMonth', how='left')
all_data = all_data.merge(data_mortgage30us[['CloseDate_YearMonth', 'MortgageRate30Fixed']], on='CloseDate_YearMonth', how='left')
all_data = all_data.merge(data_fedfunds[['CloseDate_YearMonth', 'FedInterestRate']], on='CloseDate_YearMonth', how='left')
all_data = all_data.merge(data_cpi[['CloseDate_YearMonth', 'CPI']], on='CloseDate_YearMonth', how='left')
print(all_data[['CloseDate_YearMonth', 'UnemploymentRate', 'MortgageRate30Fixed', 'FedInterestRate', 'CPI']].drop_duplicates().sort_values('CloseDate_YearMonth'))

# Fill missing macro values with the median
macro_cols = ['UnemploymentRate', 'MortgageRate30Fixed', 'FedInterestRate', 'CPI']
for col in macro_cols:
    median_val = all_data[col].median()
    all_data[col] = all_data[col].fillna(median_val)

# Encode properties outside of top 20 cities as Other city - Jun
city_count = all_data["City"].value_counts()
other_cities = city_count[city_count<=450].index.tolist()
all_data['City']=all_data['City'].replace(other_cities, 'Others')

# Encode properties in unpopular counties as Other county
county_count = all_data["CountyOrParish"].value_counts()
other_counties= county_count[county_count<=265].index.tolist()
all_data['CountyOrParish'] = all_data['CountyOrParish'].replace(other_counties, 'Others')

# Remove outliers from ClosePrice
lower = all_data['ClosePrice'].quantile(0.01)
upper = all_data['ClosePrice'].quantile(0.99)
all_data = all_data[(all_data['ClosePrice'] >= lower) & (all_data['ClosePrice'] <= upper)]

# Remove outliers from property features - Tara
property_features = [
    'LivingArea', 'BathroomsTotalInteger','BedroomsTotal', 'Stories',
    'GarageSpaces', 'LotSizeSquareFeet'
]

for col in property_features:
    lower = all_data[col].quantile(0.01)
    upper = all_data[col].quantile(0.99)
    all_data = all_data[(all_data[col] >= lower) & (all_data[col] <= upper)]

# Calculate House Age - Tommy
current_year = datetime.now().year
all_data['Age'] = current_year - all_data['YearBuilt']

# Feature Engineering lot density
all_data['LotDensity'] = all_data['LotSizeSquareFeet'] / all_data['LivingArea']
all_data['LotDensity'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values (Median imputation for missing values, since the data is right-skewed) - Tara
all_data['LivingArea'] = all_data['LivingArea'].fillna(all_data['LivingArea'].median())
all_data['Stories'] = all_data['Stories'].fillna(all_data['Stories'].median())
all_data['GarageSpaces'] = all_data['GarageSpaces'].fillna(all_data['GarageSpaces'].median())
all_data['LotSizeSquareFeet'] = all_data['LotSizeSquareFeet'].fillna(all_data['LotSizeSquareFeet'].median())
all_data['LotDensity'].fillna(all_data['LotDensity'].median(), inplace=True)

# Handle boolean facilities - Abhishek
facilities_features_bool = [
    'ViewYN','PoolPrivateYN', 'AttachedGarageYN',
    'ParkingTotal', 'FireplaceYN', 'NewConstructionYN'
]
# Hazel Note: filled with 0 instead of -1 as per discussed in the meeting
all_data[facilities_features_bool] = all_data[facilities_features_bool].fillna(0).astype(int)

# Flooring: one-hot encode multi-label values
all_data['Flooring'] = all_data['Flooring'].fillna('Unknown')
all_data['Flooring'] = all_data['Flooring'].str.split(',')
mlb = MultiLabelBinarizer()
flooring_dummies = pd.DataFrame(
    mlb.fit_transform(all_data['Flooring']),
    columns=['Flooring_' + f for f in mlb.classes_],
    index=all_data.index
)
all_data = pd.concat([all_data, flooring_dummies], axis=1)
all_data.drop(columns='Flooring', inplace=True)

# Dropping properties outside of California - Jun
all_data = all_data[all_data["StateOrProvince"]=="CA"]

# Create GeoDataFrame from coordinates - Hazel
geometry = [Point(xy) for xy in zip(all_data['Longitude'], all_data['Latitude'])]
homes_gdf = gpd.GeoDataFrame(all_data, geometry=geometry, crs="EPSG:4326")

# Load school districts shapefile (shp file needs shx and dbf files in the same directory)
districts_gdf = gpd.read_file('./dataset/tl_2023_06_unsd.shp')

# Ensure CRS match
districts_gdf = districts_gdf.to_crs(homes_gdf.crs)

# Join and clean columns
homes_with_district = gpd.sjoin(homes_gdf, districts_gdf, how='left', predicate='within')

district_name_col = 'NAME'
district_id_col = 'GEOID'

extra_cols = set(homes_with_district.columns) - set(homes_gdf.columns) - {'index_right'}
to_drop = extra_cols - {district_name_col, district_id_col}
homes_with_district = homes_with_district.drop(columns=to_drop)

# Rename for clarity
homes_with_district = homes_with_district.rename(columns={
    district_name_col: 'SchoolDistrictName',
    district_id_col: 'SchoolDistrictID'
})

# One-hot encode school district name
homes_with_district['SchoolDistrictName'] = homes_with_district['SchoolDistrictName'].fillna('Unknown')
district_dummies = pd.get_dummies(homes_with_district['SchoolDistrictName'], prefix='District_')
all_data = pd.concat([homes_with_district.drop(columns=['SchoolDistrictName']), district_dummies], axis=1)

# Drop temporary columns created during processing
all_data = all_data.drop(columns=[
    'StreetNumberNumeric', 'YearBuilt', 'CountyOrParish', 'CloseDate', 'CloseDate_Parsed', 'CloseDate_YearMonth', 
    'index_right', 'SchoolDistrictID', 'UnparsedAddress', 'City', 'StateOrProvince', 'PropertyType', 'PropertySubType',
    'PostalCode', 'geometry', 
    ])

# # Save to CSV
# all_data.to_csv(os.path.join(output_dir, 'complete_cleaned_data.csv'), index=False)

# print("✅ Pipeline completed. Cleaned data saved to ./output/complete_cleaned_data.csv")

# Save test set to CSV
all_data.to_csv(os.path.join(output_dir, 'test_cleaned_data.csv'), index=False)

print("✅ Pipeline completed. Cleaned data saved to ./output/test_cleaned_data.csv")