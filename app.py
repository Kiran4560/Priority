from flask import Flask, request, render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

# Load model and data
df = pd.read_csv("iit-and-nit-colleges-admission-criteria-version-2.csv")
del df["Unnamed: 0"]

# Specify the desired number of samples
num_samples = 1000  # Adjust this number as needed
# Randomly sample a subset of the DataFrame
sampled_df = df.sample(n=num_samples, random_state=42)  # Set a random_state for reproducibility

Pred_College = sampled_df.copy()

le = LabelEncoder()
Pred_College['quota'] = le.fit_transform(Pred_College['quota'])
Pred_College['category'] = le.fit_transform(Pred_College['category'])
Pred_College['institute_type'] = [0 if x == 'IIT' else 1 for x in Pred_College['institute_type']]
Pred_College['pool'] = [0 if x == 'Gender-Neutral' else 1 for x in Pred_College['pool']]

X = Pred_College[['institute_type', 'round_no', 'quota', 'pool', 'category', 'opening_rank', 'closing_rank']]
y = Pred_College[['institute_short', 'program_name', 'degree_short']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Load the dataset
try:
    data = pd.read_csv('rankwisecolleges.csv')
except FileNotFoundError:
    print("Error: 'rankwisecolleges.csv' file not found in the current directory:", os.getcwd())

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

sample_data = {
    'branch': ['Computer Science and Engineering', 'Polymer Science and Engineering', 'Mechanical Engineering'],
    'degree': ['B.Tech','BSc','B.Tech + M.Tech (IDD)','Btech + M.Tech (IDD)']
}

@app.route('/filter')
def filter_session():
    return render_template('filter.html', data=sample_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html',user=user)
    
    return redirect('/login')

@app.route('/colleges', methods=['POST'])
def search_colleges():
    # Get the input values from the form
    quota = request.form['quota']
    pool = request.form['pool']
    category = request.form['Category']
    rank = int(request.form['rank'])

    # Filter the data based on inputs
    filtered_data = data[(data['quota'] == quota) & 
                         (data['pool'] == pool) & 
                         (data['category'] == category) &
                         (data['closing_rank'] >= rank)]

    if filtered_data.empty:
            return "Sorry, there is no college available for you, keep trying!!!"
    else:
            # Save the filtered data in the session variable
            session['key694'] = filtered_data.to_dict(orient='records')

            # Convert the filtered data to HTML table for display
            html_table = filtered_data.to_html(index=False)

            # Add CSS styling to the HTML table
            styled_table = f"<style>table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 10px; }}</style> \
                        <style>th, td {{ text-align: left; }}</style> \
                        <style>table {{ width: 100%; }}</style> \
                        <style>th {{ background-color: blue; color: white; }}</style> \
                        {html_table}"

            return styled_table

@app.route('/model', methods=['POST'])
def model_predict():
    if request.method == 'POST':
        # Get form data
        ins_t = int(request.form['institute_type'])
        qt = int(request.form['quota'])
        pl = int(request.form['pool'])
        ct = int(request.form['Category'])
        rank = int(request.form['rank'])
        if rank<=30:
            opr = rank
        else: opr = rank - 30
        clr = rank + 30

        # Make prediction
        prediction =  pd.DataFrame(model.predict([[ins_t, 6, qt, pl, ct, opr, clr]]), columns = [['College', 'Branch', 'Degree']])
        
        # If no colleges meet the conditions, return a message
        if prediction.empty:
            return "Sorry, there is no college available for you, keep trying!!!"
        else:
            # Convert the filtered data to HTML table for display
            html_table = prediction.to_html(index=False)

            # Add CSS styling to the HTML table
            styled_table = f"<style>table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 10px; }}</style> \
                        <style>th, td {{ text-align: left; }}</style> \
                        <style>table {{ width: 100%; }}</style> \
                        <style>th {{ background-color: blue; color: white; }}</style> \
                        {html_table}"

            return styled_table

@app.route('/preferences', methods=['POST'])
def filter_function():
    branch = request.form['branch']
    degree = request.form['degree']
    pd = request.form['program_duration']

    # Access the session variable
    filtered_data = session.get('key694')

    # Check if the session variable exists
    if filtered_data:
        # Apply filters based on user selections
        filtered_result = []

        for row in filtered_data:
            # Check branch filter
            if branch == "N/A" or row['branch'] == branch:
                # Check degree filter
                if degree == "N/A" or row['degree'] == degree:
                    # Check program duration filter
                    if pd == "N/A" or row['program_duration'] == pd:
                        filtered_result.append(row)

        # Output filtered result
        # If no colleges meet the conditions, return a message
        if filtered_result.empty:
            return "Sorry, there is no college available for you from this session"
        else:
            # Convert the filtered data to HTML table for display
            html_table = filtered_result.to_html(index=False)

            # Add CSS styling to the HTML table
            styled_table = f"<style>table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 10px; }}</style> \
                        <style>th, td {{ text-align: left; }}</style> \
                        <style>table {{ width: 100%; }}</style> \
                        <style>th {{ background-color: blue; color: white; }}</style> \
                        {html_table}"

            return styled_table

        # You can return filtered_result or do further processing here
    else:
        return "No filtered data available in the session."

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)