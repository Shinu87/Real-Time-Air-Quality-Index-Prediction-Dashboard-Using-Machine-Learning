
# 🌫️ Air Quality Prediction Dashboard

An interactive, real-time air quality monitoring and AQI prediction dashboard built using **Streamlit**, **Open-Meteo API**, and a **Machine Learning pipeline**.

---

## 📌 Features

- 🔍 **Manual Input** or 📡 **Real-Time Data** from Maharashtra districts
- 📈 **AQI prediction** using trained ML model (`ExtraTreesRegressor`)
- 🌍 **Map integration** with location marker using Folium
- 📊 **Interactive visualizations** using Plotly
- ✅ Health message & precaution display based on AQI range
- 🕒 Real-time pollutant data fetched via **Open-Meteo Air Quality API**

---

## 📊 Dashboard Preview

![Dashboard Preview](https://user-images.githubusercontent.com/137891303/310615805-2ef8feae-49b0-4fa5-8bd9-94fba902c8a9.png)

---

## ⚙️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Model**: ExtraTreesRegressor pipeline (with preprocessing)
- **Data Source**: [Open-Meteo Air Quality API](https://open-meteo.com/)
- **Map & Charts**: Folium, Plotly
- **Other Libraries**: `joblib`, `pandas`, `numpy`, `requests`, `datetime`

---

## 📂 Project Structure

```bash
├── best_model_pipeline.pkl     # Trained ML model
├── app.py                      # Main Streamlit app
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
````

---

## 🔧 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/air-quality-dashboard.git
   cd air-quality-dashboard
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**

   ```bash
   streamlit run app.py
   ```

> Ensure `best_model_pipeline.pkl` (trained model) is present in the root directory.

---

## 📈 How it Works

* Users can **manually input pollutant concentrations** or select a **Maharashtra district** to fetch **real-time data**.
* The input is fed into a **pre-trained Extra Trees Regressor** model to predict AQI.
* Results are visualized using **dynamic charts** and **geo-mapping**.
* The dashboard also provides **health insights** based on the AQI level.

---

## 📊 AQI Ranges & Health Impact

| AQI Range | Air Quality Level              | Health Impact                       |
| --------- | ------------------------------ | ----------------------------------- |
| 0–50      | Good                           | Minimal impact                      |
| 51–100    | Moderate                       | Some risk for sensitive individuals |
| 101–150   | Unhealthy for Sensitive Groups | May affect sensitive people         |
| 151–200   | Unhealthy                      | Risk for everyone                   |
| 201–300   | Very Unhealthy                 | Health alert, emergency conditions  |
| 301+      | Hazardous                      | Serious health effects              |

---

## 📬 Contact

**Shrinivas More**
📧 [shrinivasmore51@gmail.com]


---

## 📜 License

This project is open source 
