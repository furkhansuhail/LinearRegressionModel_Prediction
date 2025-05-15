# Importing necessary modules
from ConfigurationSetup import *
from DataCleanupModule import *

# ✅ Update the config class to include all needed fields
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list  # Can be removed if not used


# ✅ Properly create the config object
config = DataIngestionConfig(
    root_dir=Path("Dataset"),  # <-- Folder where file is saved
    source_URL="https://github.com/furkhansuhail/ProjectData/raw/refs/heads/main/LinearRegressionDataset/used_cars_data.csv",
    local_data_file=Path("Dataset/used_cars_data.csv"),  # <-- CSV file name
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]  # Can be removed or filled if necessary
)


def download_csv_file(source_URL, local_data_file):
    if local_data_file.exists():
        print(f"File already exists at: {local_data_file}")
    else:
        print(f"⬇Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(
            url=source_URL,
            filename=local_data_file
        )
        print(f"File downloaded and saved to: {file_path}")


class LinearRegression_UsedCarPrediction:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        os.makedirs(self.config.root_dir, exist_ok=True)
        download_csv_file(self.config.source_URL, self.config.local_data_file)
        print("Data downloaded from GITHUB and saved to: ", self.config.local_data_file)
        self.Dataset = pd.read_csv(self.config.local_data_file)
        self.CleanedDataset = pd.DataFrame()
        self.ModelDriver()

    def ModelDriver(self):

        # Supressing print Statements Please uncomment the code to restore print statement before running the code
        # Save the original stdout to restore it later
        original_stdout = sys.stdout
        # Redirect stdout to devnull to suppress print statements
        sys.stdout = open(os.devnull, 'w')


        # DataCleanupModule(self.Dataset)
        data_cleanup = DataCleanupModule(self.Dataset)
        data_cleanup.DataPreprocessing()
        data_cleanup.ProcessingSeats()
        data_cleanup.EDA()
        data_cleanup.HandlingMissingValues()
        self.CleanedDataset = data_cleanup.Bivariate_Multivariate_Analysis()

        # Restore stdout to original so print statements will show again
        sys.stdout = original_stdout

        self.TrainModel()

    def TrainModel(self):
        # Step 1: Split features and target
        X = self.CleanedDataset.drop(["Price", "Price_log"], axis=1)
        y = self.CleanedDataset[["Price_log", "Price"]]

        # Step 2: Clean & encode categorical features
        def encode_cat_vars(x):
            if "Unnamed: 0" in x.columns:
                x = x.drop(columns=["Unnamed: 0"])
            return pd.get_dummies(
                x,
                columns=x.select_dtypes(include=["object", "category"]).columns.tolist(),
                drop_first=True
            )

        X = encode_cat_vars(X)

        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Step 4: Add constant and ensure numeric
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        X_train = X_train.apply(pd.to_numeric, errors='coerce').astype(float)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').astype(float)

        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        y_train["Price_log"] = y_train["Price_log"].astype(float)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        print("X_train:", X_train.shape)
        print("X_test:", X_test.shape)

        # Step 5: Build OLS model
        def build_ols_model(train):
            olsmodel = sm.OLS(y_train["Price_log"], train)
            return olsmodel.fit()

        olsmodel1 = build_ols_model(X_train)
        print(olsmodel1.summary())

        # Step 6: Evaluation metrics
        def rmse(pred, actual): return np.sqrt(((actual - pred) ** 2).mean())

        def mape(pred, actual): return np.mean(np.abs((actual - pred) / actual)) * 100

        def mae(pred, actual): return np.mean(np.abs(actual - pred))

        # Step 7: Evaluate model performance
        def model_pref(olsmodel, x_train, x_test):
            y_pred_train_log = olsmodel.predict(x_train)
            y_pred_test_log = olsmodel.predict(x_test)

            y_pred_train = np.exp(np.atleast_1d(y_pred_train_log))
            y_pred_test = np.exp(np.atleast_1d(y_pred_test_log))

            y_train_actual = y_train["Price"]
            y_test_actual = y_test["Price"]

            predictions_df = pd.DataFrame({
                "Actual_Train_Price": y_train_actual,
                "Predicted_Train_Price": y_pred_train,
                "Actual_Test_Price": pd.Series(y_test_actual),
                "Predicted_Test_Price": pd.Series(y_pred_test)
            })

            print(predictions_df.head(10))
            predictions_df.to_csv("ModelPredictions.csv", index=False)

            print(pd.DataFrame({
                "Data": ["Train", "Test"],
                "RMSE": [rmse(y_pred_train, y_train_actual), rmse(y_pred_test, y_test_actual)],
                "MAE": [mae(y_pred_train, y_train_actual), mae(y_pred_test, y_test_actual)],
                "MAPE": [mape(y_pred_train, y_train_actual), mape(y_pred_test, y_test_actual)],
            }))

        model_pref(olsmodel1, X_train, X_test)

        # Step 8: New Prediction Example
        print("\n--- Predicting for a new sample ---")
        new_sample_raw = self.CleanedDataset.drop(["Price", "Price_log"], axis=1).iloc[[0]].copy()

        # Modify the values if needed
        new_sample_raw["Kilometers_Driven"] = 50000
        new_sample_raw["Fuel_Type"] = "Petrol"
        new_sample_raw["Transmission"] = "Manual"
        new_sample_raw["Owner_Type"] = "First"
        new_sample_raw["Location"] = "Mumbai"
        new_sample_raw["Brand_Class"] = "Low"

        # Encode and align to training columns
        new_sample_encoded = pd.get_dummies(new_sample_raw)
        new_sample_encoded = new_sample_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Predict
        new_price_log = olsmodel1.predict(new_sample_encoded)[0]
        new_price = np.exp(new_price_log)

        print(f"Predicted log(Price): {new_price_log:.4f}")
        print(f"Predicted Price: ₹ {new_price:.2f}")


# Create an instance of LinearRegression_UsedCarPrediction and start the process
LinearRegression_UsedCarPredictionObj = LinearRegression_UsedCarPrediction(config)
