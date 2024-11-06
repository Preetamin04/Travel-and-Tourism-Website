from flask import Flask, render_template, request, Response, url_for, redirect
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

app = Flask(__name__, template_folder='templates')


# TravelCostPredictor Class
class TravelCostPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = pd.DataFrame(columns=['Destination', 'Duration', 'No_of_traveller', 'No_of_Male', 'No_of_Female', 'No_of_child', 'Accommodation_type', 'Transportation_type'])
        self.numeric_features = ['Duration', 'No_of_traveller', 'No_of_Male', 'No_of_Female', 'No_of_child']

    def load_data(self):
        # Ensure the data file exists before attempting to load it
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
        else:
            raise FileNotFoundError(f"The file {self.data_path} does not exist.")

    def clean_data(self):
        self.data_cleaned = self.data.dropna()

    def prepare_features(self):
        self.X = self.data_cleaned.drop("Total_cost", axis=1)
        self.y = self.data_cleaned["Total_cost"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.numeric_features = self.X.select_dtypes(include=["int64", "float64"]).columns
        self.categorical_features = self.X.select_dtypes(include=["object"]).columns
        self.numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        self.categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        self.preprocessor = ColumnTransformer(transformers=[("num", self.numeric_transformer, self.numeric_features),
                                                           ("cat", self.categorical_transformer, self.categorical_features),])

    def build_model(self):
        self.model = Pipeline(steps=[("preprocessor", self.preprocessor), ("regressor", LinearRegression())])
        self.model.fit(self.X_train, self.y_train)

    def predict(self, new_data):
        new_prediction = self.model.predict(new_data)
        return round(new_prediction[0], 2)


# Flask routes

@app.route("/")
def index():
    return render_template('index.html')

# Adding /index route which renders the same index page as "/"
@app.route("/index")
def index_route():
    return render_template('index.html')

@app.route("/package")
def package():
    return render_template('package.html')

@app.route("/blogs")
def blogs():
    return render_template('blogs.html')

@app.route("/about_us")
def about_us():
    return render_template('about_us.html')

@app.route("/Andaman")
def Andaman():
    return render_template('Andaman.html')

@app.route("/Reef")
def Reef():
    return render_template('Reef.html')

@app.route("/Caves")
def Caves():
    return render_template('Caves.html')

@app.route("/antelope")
def antelope():
    return render_template('antelope.html')

@app.route("/fuji")
def fuji():
    return render_template('fuji.html')

@app.route("/himalaya")
def himalaya():
    return render_template('himalaya.html')

@app.route("/azores")
def azores():
    return render_template('azores.html')

@app.route("/iceland")
def iceland():
    return render_template('iceland.html')

@app.route("/amazon")
def amazon():
    return render_template('amazon.html')

@app.route("/payment")
def payment():
    return render_template('payment.html')

@app.route("/View-Data")
def viewdata():    
    with open('Travel_details_dataset.csv', 'r') as file:
        csv_reader = csv.reader(file)
        studentdata = []
        for row in csv_reader:
            studentdata.append(row)
    n = len(studentdata) - 1
    return render_template('View-Data.html', studentdata=studentdata, n=n)

@app.route('/Student_Info')
def student_info():
    file_path = 'static/Student_Info.txt'
    with open(file_path, 'r') as file:
        content = file.read()
    return Response(content, mimetype='text/plain')

@app.route("/predict", methods=["GET", "POST"])  
def predict():
    # Specify the correct path to your dataset
    data_path = os.path.join(os.path.dirname(__file__), "Travel_details_dataset.csv")
    predictor = TravelCostPredictor(data_path)

    try:
        predictor.load_data()
        predictor.clean_data()
        predictor.prepare_features()
        predictor.build_model()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return str(e), 500

    if request.method == "POST":
        form_data = {k: v for k, v in request.form.items()}
        
        print("Received form data:", form_data)  # Print the received form data
        
        input_data = pd.DataFrame([form_data])
        
        # Explicitly map column names to expected names in predictor.X
        expected_columns = predictor.X.columns.tolist()
        input_data.columns = [col for col in expected_columns if col in form_data]  # Adjust columns based on expected names
        
        print("Input data after column adjustment:", input_data)  # Print adjusted input data

        # Convert all fields to numeric features
        for column in input_data.columns:
            input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
        
        print("Input data after converting all features to numeric:", input_data)  # Print after numeric conversion
        
        input_data.dropna(inplace=True)
        
        if input_data.empty:
            print("Input data is empty after dropping NA values.")  # Print if no valid input data is available
            return "Error: No valid input data available for prediction.", 400

        try:
            prediction = predictor.predict(input_data)
            print("Prediction result:", prediction)  # Print the prediction result
            return render_template('result.html', prediction=prediction)
        except KeyError as e:
            print(f"KeyError: Missing expected input field - {e}")  # Print missing field error
            return f"Error: Missing expected input field - {e}", 400
        except Exception as e:
            print(f"An error occurred: {str(e)}")  # Print generic error message
            return f"An error occurred: {str(e)}", 500

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
