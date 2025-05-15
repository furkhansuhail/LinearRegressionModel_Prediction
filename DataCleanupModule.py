from ConfigurationSetup import *


class DataCleanupModule:
    def __init__(self, dataset):
        # print(dataset.info())
        self.Dataset = dataset

    def DataPreprocessing(self):
        # removing kmpl units
        self.Dataset["Mileage"] = self.Dataset["Mileage"].str.rstrip(" kmpl")
        self.Dataset["Mileage"] = self.Dataset["Mileage"].str.rstrip(" km/g")

        # remove CC units
        self.Dataset["Engine"] = self.Dataset["Engine"].str.rstrip(" CC")

        # remove bhp and replace null with nan
        self.Dataset["Power"] = self.Dataset["Power"].str.rstrip(" bhp")
        self.Dataset["Power"] = self.Dataset["Power"].replace(regex="null", value=np.nan)

        # verify the data
        num = ['Engine', 'Power', 'Mileage']
        # print(self.Dataset[num].sample(20))

        """
        I had seen some values in Power and Mileage as 0.0 so verifying data for Engine, Power, Mileage.
        Will check once again after converting datatype
        """
        self.Dataset.query("Power == '0.0'")['Power'].count()
        self.Dataset.query("Mileage == '0.0'")['Mileage'].count()

        # Converting 0.0 to Nans
        self.Dataset.loc[self.Dataset["Mileage"] == '0.0', 'Mileage'] = np.nan
        self.Dataset.loc[self.Dataset["Engine"] == '0.0', 'Engine'] = np.nan

        # print(self.Dataset[num].nunique())
        # print(self.Dataset[num].isnull().sum())

    def ProcessingSeats(self):

        # print(self.Dataset.query("Seats == 0.0")['Seats'])
        # There is one empty value at 3999 row
        self.Dataset.loc[3999, 'Seats'] = np.nan

        # Processing New Price
        new_price_num = []

        # Regex for numeric + " " + "Lakh"  format
        regex_power = "^\d+(\.\d+)? Lakh$"


        for observation in self.Dataset["New_Price"]:
            if isinstance(observation, str):
                if re.match(regex_power, observation):
                    new_price_num.append(float(observation.split(" ")[0]))
                else:
                    # To detect if there are any observations in the column that do not follow [numeric + " " + "Lakh"]  format
                    # that we see in the sample output
                    pass
                    print(
                        "The data needs furthur processing.mismatch ",
                        observation,
                    )
            else:
                # If there are any missing values in the New_Price column, we add missing values to the new column
                new_price_num.append(np.nan)

        new_price_num = []

        for observation in self.Dataset["New_Price"]:
            if isinstance(observation, str):
                if re.match(regex_power, observation):
                    new_price_num.append(float(observation.split(" ")[0]))
                else:
                    # Converting values in Crore to lakhs
                    new_price_num.append(float(observation.split(" ")[0]) * 100)
            else:
                # If there are any missing values in the New_Price column, we add missing values to the new column
                new_price_num.append(np.nan)

        # Add the new column to the data
        self.Dataset["new_price_num"] = new_price_num

        # Checking the new dataframe
        # print(self.Dataset.head(5))  # Looks ok

        # converting object data type to category data type
        self.Dataset["Fuel_Type"] = self.Dataset["Fuel_Type"].astype("category")
        self.Dataset["Transmission"] = self.Dataset["Transmission"].astype("category")
        self.Dataset["Owner_Type"] = self.Dataset["Owner_Type"].astype("category")
        # converting datatype
        self.Dataset["Mileage"] = self.Dataset["Mileage"].astype(float)
        self.Dataset["Power"] = self.Dataset["Power"].astype(float)
        self.Dataset["Engine"] = self.Dataset["Engine"].astype(float)

        # print(self.Dataset.describe())

        # Calculating Age of the car
        self.Dataset['Current_year'] = 2021
        self.Dataset['Ageofcar'] = self.Dataset['Current_year'] - self.Dataset['Year']
        self.Dataset.drop('Current_year', axis=1, inplace=True)
        # print(self.Dataset.head())

        # dropping rows with name as null
        self.Dataset = self.Dataset.dropna(subset=['Name'])

        # As mentioned in dataset car name has Brand and model so extracting it ,
        # This can help to fill missing values of price column as brand
        self.Dataset['Brand'] = self.Dataset['Name'].str.split(' ').str[0]  # Separating Brand name from the Name
        self.Dataset['Model'] = self.Dataset['Name'].str.split(' ').str[1] + self.Dataset['Name'].str.split(' ').str[2]

        # print(self.Dataset.head(10))
        # print(self.Dataset.Brand.unique())

        # Updating Brand Names-brand name cleanup
        # changing brandnames
        self.Dataset.loc[self.Dataset.Brand == 'ISUZU', 'Brand'] = 'Isuzu'
        self.Dataset.loc[self.Dataset.Brand == 'Mini', 'Brand'] = 'Mini Cooper'
        self.Dataset.loc[self.Dataset.Brand == 'Land', 'Brand'] = 'Land Rover'
        # self.Dataset['Brand']=self.Dataset["Brand"].astype("category")

        # print(self.Dataset.groupby(self.Dataset.Brand).size().sort_values(ascending=False))

        # drop row with no model
        before = self.Dataset.shape[0]
        self.Dataset.dropna(subset=['Model'], axis=0, inplace=True)
        after = self.Dataset.shape[0]
        # print(f"Dropped {before - after} rows with null 'Name'")
        #
        # print(self.Dataset.groupby('Model')['Model'].size().nlargest(30))

    def EDA(self):
        # print(self.Dataset.describe())

        self.Dataset = self.Dataset.drop(columns=['S.No.'])
        # print(self.Dataset.head(100))

        plt.style.use('ggplot')
        # select all quantitative columns for checking the spread
        numeric_columns = self.Dataset.select_dtypes(include=np.number).columns.tolist()
        plt.figure(figsize=(20, 25))

        for i, variable in enumerate(numeric_columns):
            plt.subplot(10, 3, i + 1)

            sns.distplot(self.Dataset[variable], kde=False, color='blue')
            plt.tight_layout()
            plt.title(variable)

        cat_columns = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type',
                       'Brand']  # cars.select_dtypes(exclude=np.number).columns.tolist()

        plt.figure(figsize=(15, 21))

        for i, variable in enumerate(cat_columns):
            plt.subplot(4, 2, i + 1)
            order = self.Dataset[variable].value_counts(ascending=False).index
            ax = sns.countplot(x=self.Dataset[variable], data=self.Dataset, order=order, palette='viridis')
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_height() / len(self.Dataset[variable]))
                x = p.get_x() + p.get_width() / 2 - 0.05
                y = p.get_y() + p.get_height()
                plt.annotate(percentage, (x, y), ha='center')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.title(variable)

        numeric_columns = numeric_columns = self.Dataset.select_dtypes(include=np.number).columns.tolist()
        plt.figure(figsize=(13, 17))

        for i, variable in enumerate(numeric_columns):
            plt.subplot(5, 2, i + 1)
            sns.scatterplot(x=self.Dataset[variable], y=self.Dataset['Price']).set(title='Price vs ' + variable)
            # plt.xticks(rotation=90)
            plt.tight_layout()
        plt.show()

    def HandlingMissingValues(self):
        # using business knowledge to create class
        Low = ['Maruti', 'Hyundai', 'Ambassdor', 'Hindustan', 'Force', 'Chevrolet', 'Fiat', 'Tata', 'Smart', 'Renault',
               'Datsun', 'Mahindra', 'Skoda', 'Ford', 'Toyota', 'Isuzu', 'Mitsubishi', 'Honda']

        High = ['Audi', 'Mini Cooper', 'Bentley', 'Mercedes-Benz', 'Lamborghini', 'Volkswagen', 'Porsche', 'Land Rover',
                'Nissan', 'Volvo', 'Jeep', 'Jaguar', 'BMW']  # more than 30lakh

        # Check for null values
        # print("Handling missing values")
        # print(self.Dataset.isnull().sum())

        # counting the number of missing Value per row
        num_missing = self.Dataset.isnull().sum(axis=1)
        # print(num_missing.value_counts())

        # Investigating how many missing values per row are there for each variable
        # for n in num_missing.value_counts().sort_index().index:
        #     if n > 0:
        #         print("*" * 30, f'\nFor the rows with exactly {n} missing values, NAs are found in:')
        #         n_miss_per_col = self.Dataset[num_missing == n].isnull().sum()
        #         print(n_miss_per_col[n_miss_per_col > 0])
        #         print('\n\n')

        # col = ['Engine', 'Power', 'Mileage']
        # print(self.Dataset[col].isnull().sum())

        # We need to start filling missing values by grouping name and year and fill in missing values with median
        # print(self.Dataset.info())
        # print(self.Dataset.groupby(['Name','Year'])['Engine'].median().head(30))
        #
        # self.Dataset['Engine'] = self.Dataset.groupby(['Name', 'Year'])['Engine'].apply(lambda x: x.fillna(x.median()))
        # self.Dataset['Power'] = self.Dataset.groupby(['Name', 'Year'])['Power'].apply(lambda x: x.fillna(x.median()))
        # self.Dataset['Mileage'] = self.Dataset.groupby(['Name', 'Year'])['Mileage'].apply(lambda x: x.fillna(x.median()))
        # print(self.Dataset.head(10))

        self.Dataset['Engine'] = self.Dataset.groupby(['Name', 'Year'])['Engine'].transform(lambda x: x.fillna(x.median()))
        self.Dataset['Power'] = self.Dataset.groupby(['Name', 'Year'])['Power'].transform(lambda x: x.fillna(x.median()))
        self.Dataset['Mileage'] = self.Dataset.groupby(['Name', 'Year'])['Mileage'].transform(lambda x: x.fillna(x.median()))

        # col = ['Engine', 'Power', 'Mileage']
        # print(self.Dataset[col].isnull().sum())
        # print(self.Dataset.head(10))

        # chosing Median to fill the the missing value as there are many outliers,
        # grouping by model and year to get  more granularity and more accurate Engine and then fillig with median
        self.Dataset['Engine'] = self.Dataset.groupby(['Brand', 'Model'])['Engine'].transform(lambda x: x.fillna(x.median()))
        self.Dataset['Power'] = self.Dataset.groupby(['Brand', 'Model'])['Power'].transform(lambda x: x.fillna(x.median()))
        # chosing Median to fill the the missing value as there are many outliers,
        # grouping by model to get more granularity and more accurate Engine
        self.Dataset['Mileage'] = self.Dataset.groupby(['Brand', 'Model'])['Mileage'].transform(lambda x: x.fillna(x.median()))
        # print(self.Dataset.head(10))

        """
         There are still missing values , analyzing further .Grouping by only Model for Engine and then 
         filling missing values with median. For  Power and Mileage Engine values for a Brand can be used 
         to get more accurate value.
        """
        # print(self.Dataset.groupby(['Model', 'Year'])['Engine'].agg(['median', 'mean', 'max']).
        #  sort_values(by='Model', ascending=True).head(10))
        # print(self.Dataset.groupby(['Brand','Engine'])['Power'].agg({'mean','median','max'}).head(10))
        # print(self.Dataset['Seats'].isnull().sum())

        # Grouping with Name should give me more granularity, and near to accurate Seat values.
        self.Dataset['Seats'] = self.Dataset.groupby(['Name'])['Seats'].transform(lambda x: x.fillna(x.median()))
        # print(self.Dataset['Seats'].isnull().sum())

        # Grouping with Model should give me more granularity, and near to accurate Seat values.
        self.Dataset['Seats'] = self.Dataset.groupby(['Model'])['Seats'].transform(lambda x: x.fillna(x.median()))
        # print(self.Dataset['Seats'].isnull().sum())

        # Checking which car types have missing values
        # print(self.Dataset[self.Dataset['Seats'].isnull() == True].head(10))

        # most of cars are 5 seater so fillrest of 23 by 5
        self.Dataset['Seats'] = self.Dataset['Seats'].fillna(5)
        # print(self.Dataset['Seats'].isnull().sum())

        # Need to analyse along with price if seats plays any role in price
        # The .astype("category") method tells pandas to treat the column as a categorical
        # variable instead of a plain object (string) or numeric type.

        self.Dataset["Location"] = self.Dataset["Location"].astype("category")
        self.Dataset['Brand'] = self.Dataset['Brand'].astype("category")
        # print(self.Dataset.head(10))

        # Processing New Price
        # For better granualarity grouping has there would be same car model present so filling with a median value brings it more near to real value
        self.Dataset['new_price_num'] = self.Dataset.groupby(['Name', 'Year'])['new_price_num'].transform(
            lambda x: x.fillna(x.median()))
        # print(self.Dataset.new_price_num.isnull().sum())

        self.Dataset['new_price_num'] = self.Dataset.groupby(['Name'])['new_price_num'].transform(lambda x: x.fillna(x.median()))
        # print(self.Dataset.new_price_num.isnull().sum())

        self.Dataset['new_price_num'] = self.Dataset.groupby(['Brand', 'Model'])['new_price_num'].transform(
            lambda x: x.fillna(x.median()))
        # print(self.Dataset.new_price_num.isnull().sum())

        self.Dataset['new_price_num'] = self.Dataset.groupby(['Brand'])['new_price_num'].transform(lambda x: x.fillna(x.median()))
        self.Dataset.drop(['New_Price'], axis=1, inplace=True)

        # print(self.Dataset.new_price_num.isnull().sum())

        # print(self.Dataset.groupby(['Brand'])['new_price_num'].median().sort_values(ascending=False))

        cols1 = ["Power", "Mileage", "Engine"]

        for ii in cols1:
            self.Dataset[ii] = self.Dataset[ii].fillna(self.Dataset[ii].median())

        # dropping remaining rows
        # cannot further fill this rows so dropping them
        self.Dataset.dropna(inplace=True, axis=0)
        # print(self.Dataset.isnull().sum())
        # print(self.Dataset.head())

        # Data has been cleaned
        print(self.Dataset.isnull().sum())

        print(self.Dataset.groupby(['Brand'])['Price'].agg({'median', 'mean', 'max'}))

        def classrange(x):
            if x in Low:
                return "Low"
            elif x in High:
                return "High"
            else:
                return x

        self.Dataset['Brand_Class'] = self.Dataset['Brand'].apply(lambda x: classrange(x))
        # print(self.Dataset['Brand_Class'].unique())

        # Changing Datatype of the columns
        self.Dataset['Engine'] = self.Dataset['Engine'].astype(int)
        self.Dataset['Brand_Class'] = self.Dataset["Brand_Class"].astype('category')

    def Bivariate_Multivariate_Analysis(self):
        """
        Bivariate analysis examines the relationship between two variables,
        while multivariate analysis explores the relationships and interactions among three or more variables.
        """
        # Heat Map
        # Only include numeric columns
        numeric_cars = self.Dataset.select_dtypes(include=['number'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_cars.corr(), annot=True, cmap="YlGnBu")
        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()

        """
        OBSERVATIONS

        - Engine has strong positive correlation to Power [0.86]. 
        - Price has positive correlation to Engine[0.66] as well Power [0.77].
        - Mileage is negative correlated to Engine,Power,Price.,Ageofcar
        - Price has negative  correlation to age of car.
        - Kilometer driven doesnt impact Price
        """

        sns.pairplot(data=self.Dataset, corner=True)

        """
        OBSERVATIONS 

        - Same observation  about correlation as seen in heatmap.
        - Kilometer driven  doesnot have impact on  Price . 
        - As power increase mileage decrease.
        - Car with recent make sell at higher prices.
        - Engine and Power increase , price of the car seems to increase.
        """

        # Variables that are correlated with price variable
        # understand relation ship of Engine vs Price and Transmimssion

        # Engine Vs price
        plt.figure(figsize=(10, 7))

        plt.title("Price VS Engine based on Transmission")
        sns.scatterplot(y='Engine', x='Price', hue='Transmission', data=self.Dataset)

        # Power Vs Price
        # understand relationship betweem Price and Power
        plt.figure(figsize=(10, 7))
        plt.title("Price vs Power based on Transmission")
        sns.scatterplot(y='Power', x='Price', hue='Transmission', data=self.Dataset)

        # Price Vs Mileage Vs Transmission
        # Understand the relationships  between mileage and Price
        sns.scatterplot(y='Mileage', x='Price', hue='Transmission', data=self.Dataset)
        # plt.show()

        # Price Vs Year Vs Transmission
        # Impact of years on price
        plt.figure(figsize=(10, 7))
        plt.title("Price based on manufacturing Year of model")
        sns.lineplot(x='Year', y='Price', hue='Transmission',
                     data=self.Dataset)

        # Impact of years on price Vs FuelType
        plt.figure(figsize=(10, 7))
        plt.title("Price Vs Year VS FuelType")
        sns.lineplot(x='Year', y='Price', hue='Fuel_Type',
                     data=self.Dataset)

        # Impact of years on price Vs Owner_Type
        plt.figure(figsize=(10, 7))
        plt.title("Price Vs Year VS Owner_Type")
        sns.lineplot(x='Year', y='Price', hue='Owner_Type',
                     data=self.Dataset)

        # plt.show()

        # print(self.Dataset[(self.Dataset["Owner_Type"]=='Third') & (self.Dataset["Year"].isin([2010]))].sort_values(by='Price',ascending =False))
        #
        # print(self.Dataset.describe())

        # Understand relationships  between price and mileage
        plt.figure(figsize=(10, 7))
        plt.title("Price Vs Mileage")
        sns.scatterplot(y='Price', x='Mileage', hue='Fuel_Type', data=self.Dataset)

        # Price and seats
        plt.figure(figsize=(20, 15))
        sns.set(font_scale=2)
        sns.barplot(x='Seats', y='Price', data=self.Dataset)
        plt.grid()

        # Price and LOcation
        plt.figure(figsize=(20, 15))
        sns.set(font_scale=2)
        sns.barplot(x='Location', y='Price', data=self.Dataset)
        plt.grid()

        # Price and band
        plt.figure(figsize=(20, 15))
        sns.set(font_scale=2)
        sns.boxplot(x='Price', y='Brand', data=self.Dataset)
        plt.grid()

        sns.relplot(data=self.Dataset, y='Price', x='Mileage', hue='Transmission', aspect=1, height=5)

        sns.relplot(data=self.Dataset, y='Price', x='Year', col='Owner_Type', hue='Transmission', aspect=1, height=5)

        sns.relplot(data=self.Dataset, y='Price', x='Engine', col='Transmission', aspect=1, height=6, hue="Fuel_Type")
        # plt.show()

        """
        # OBSERVATIONS

        - Expensive cars are in Coimbatore and Banglore.
        - 2 Seater cars are more expensive.
        - Deisel Fuel type car are more expensive compared to other fuel type.
        - As expected, Older model are sold cheaper compared to latest model
        - Automatic transmission vehicle have a higher price than manual transmission vehicles.
        - Vehicles with more engine capacity have higher prices. 
        - Price decreases as number of owner increases.
        - Automatic transmission require high engine and power.
        - Prices for Cars with fuel type as Deisel has increased with recent models 
        - Engine,Power, how old the car his, Mileage,Fuel type,location,Transmission effect the price.
        """

        # check distrubution if skewed. If distrubution is skewed , it is advice to use log transform
        cols_to_log = self.Dataset.select_dtypes(include=np.number).columns.tolist()
        for colname in cols_to_log:
            sns.distplot(self.Dataset[colname], kde=True)
            plt.show()

        def Perform_log_transform(df, col_log):
            """#Perform Log Transformation of dataframe , and list of columns """
            for colname in col_log:
                df[colname + '_log'] = np.log(df[colname])
            # df.drop(col_log, axis=1, inplace=True)
            df.info()

        # This needs to be done before the data is split
        Perform_log_transform(self.Dataset, ['Kilometers_Driven', 'Price'])

        self.Dataset.drop(['Name', 'Model', 'Year', 'Brand', 'new_price_num'], axis=1, inplace=True)

        print(self.Dataset.info())
        return self.Dataset