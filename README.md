Description of the files given to students

README.md - this file

data/kernel/receivals.csv - training data
data/kernel/purchase_orders.csv - imporant data that contains whats ordered  

data/extended/materials.csv - optional data related to the different materials
data/extended/transportaton.csv - optional data related to transporation

data/prediction_mapping.csv - mapping used to generate submissions
data/sample_submission.csv - demo submission file for Kaggle, predicting all zeros

Dataset definitions and explanation.docx - a documents that gives more details about the dataset and column names  
Machine learning task for TDT4173.docx - brief introduction to the task
kaggle_metric.ipynb - the score function we use in the Kaggle competition

## ML Model Features

### Recommended Columns for Training an ML Model

#### **Main Features (Core Variables):**

##### **From purchase_orders.csv:**

- `quantity` - Order quantity (important for weight prediction)
- `delivery_date` - Delivery date (temporal features)
- `product_id` - Product identification
- `product_version` - Product version
- `created_date_time` - Creation timestamp
- `status_id`/`status` - Order status

##### **From receivals.csv:**

- `product_id` - Product identification (for joining)
- `supplier_id` - Supplier identification
- `date_arrival` - Arrival date
- `receival_status` - Receival status

##### **From materials.csv:**

- `raw_material_alloy` - Material alloy
- `raw_material_format_type` - Material format
- `stock_location` - Storage location

##### **From transportation.csv (very important for weight prediction):**

- `transporter_name` - Transporter
- `vehicle_no` - Vehicle identification
- `gross_weight` - Gross weight
- `tare_weight` - Tare weight
- `wood`, `ironbands`, `plastic`, `water`, `ice`, `other`, `chips`, `packaging`, `cardboard` - Additional weights

#### **Engineered Features (Derived Features):**

```python
# Time-based features
df['delivery_month'] = pd.to_datetime(df['delivery_date']).dt.month
df['delivery_weekday'] = pd.to_datetime(df['delivery_date']).dt.dayofweek
df['days_between_order_delivery'] = (delivery_date - created_date_time).dt.days

# Weight features
df['total_additional_weight'] = wood + ironbands + plastic + water + ice + other + chips + packaging + cardboard
df['net_to_gross_ratio'] = net_weight / gross_weight
```

#### **Target Variable:**

- `net_weight` (from receivals.csv or transportation.csv)
