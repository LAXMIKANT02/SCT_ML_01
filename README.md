# 🏠 House Price Predictor

A sleek and interactive machine learning web application built with **Django** that predicts house prices based on input features like square footage, number of bedrooms and bathrooms. Designed with a modern UI and persistent history storage.

---

## 🔍 Features

- 📊 **Accurate Predictions** using Linear Regression
- 💾 **Prediction History** with timestamped records
- 🧼 **Delete Individual Record** dynamically
- 🎨 **Modern Glassmorphism UI** with animated effects
- 🔐 **CSRF-Protected AJAX Requests**

---

## 🛠️ Tech Stack

| Category         | Technologies Used                          |
|------------------|---------------------------------------------|
| **Backend**      | Django, Python                              |
| **Frontend**     | HTML5, CSS3, JavaScript (Vanilla)           |
| **ML Algorithm** | Linear Regression (Scikit-learn)            |
| **Database**     | SQLite (default Django DB)                  |
| **Design**       | Custom CSS with Glassmorphism, Gradient UI  |

---

## 🧠 ML Model Info

- Trained on sample housing dataset.
- Features:
  - `Square Footage`
  - `Bedrooms`
  - `Bathrooms`
- Target:
  - `Predicted Price`
- Model: `LinearRegression()` from `sklearn.linear_model`

---

## 📂 Project Structure

```

SCR\_ML\_01/
├── predictor/
│   ├── static/
│   │   ├── css/
│   │   │   ├── index.css
|   |       ├── result.css
│   │   │   └── history.css
|   |   ├──img/
│   │   └── js/
│   │       ├──result.js
|   |       ├── index.js
|   |       └── history.js
│   ├── templates/
│   │   └── predictor/
│   │       ├── index.html
|   |       ├── result.html
│   │       └── history.html
│   ├── migrations/
│   ├── **init**.py
│   ├── models.py
|   ├── apps.py
|   ├── forms.py
|   ├── tests.py
│   ├── views.py
│   ├── urls.py
│   └── admin.py
├── data/
|   ├── scalar.pkl
|   ├── linear_model.pkl
│   └── train.csv
├── test.csv
├── SCT_ML_1/
|   ├── asgi.py
|   ├── settings.py
|   ├── urls.py
|   └── wsgi.py
├── db.sqlite3
├── manage.py
├──venv/
└── README.md

````

---

## 💻 UI Preview

### 🏠  Home Page
- Input form for prediction
- Glass card layout
- Golden call-to-action button

### 📜 History Page
- Table of previous predictions
- AJAX-based Delete Single & Delete All
- Responsive & animated design

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/LAXMIKANT02/SCT_ML_01.git
   cd SCT_ML_01
   ````

2. **Create & activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Run the server**

   ```bash
   python manage.py runserver
   ```

6. **Visit**

   ```
   http://127.0.0.1:8000/
   ```

---

## 📦 Requirements
1. asgiref==3.9.1
2. Django==5.2.4
3. joblib==1.5.1
4. numpy==2.3.2
5. pandas==2.3.1
6. python-dateutil==2.9.0.post0
7. pytz==2025.2
8. scikit-learn==1.7.1
9. scipy==1.16.1
10. six==1.17.0
11. sqlparse==0.5.3
12. threadpoolctl==3.6.0
13. tzdata==2025.2

---

## 📝 Future Enhancements

* 📉 Model tuning for higher accuracy
* 📈 Add graph visualization of predictions
* 👥 User authentication with personal history
* ☁️ Cloud deployment (e.g., Heroku, Render)
* 🧠 Switch to deep learning for complex patterns

---

## 👨‍💻 Developer

**Laxmikant S**

🧑‍💼 [LinkedIn](www.linkedin.com/in/laxmikant-dadagi-b559b332a)

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

### 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, feel free to fork the repository and submit a pull request.

#### 🛠️ How to Contribute

1. **Fork** this repository
2. **Clone** your forked repo locally:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
3. **Create a new branch** for your feature or fix:

   ```bash
   git checkout -b feature-name
   ```
4. **Make your changes** and commit:

   ```bash
   git add .
   git commit -m "Add your message here"
   ```
5. **Push** to your fork:

   ```bash
   git push origin feature-name
   ```
6. **Create a Pull Request** and describe your changes

#### 📌 Guidelines

* Follow the existing coding style.
* Write clear and concise commit messages.
* Test thoroughly before submitting.
* Document any new features or configurations.
---
