import streamlit as st
from datetime import date

from src.predict_live import predict_live

st.set_page_config(page_title="EcoSentinel-AI", layout="centered")

st.title("EcoSentinel-AI")
st.subheader("Live Environmental Anomaly Detection")

city = st.text_input("Enter city", placeholder="Hyderabad")
selected_date = st.date_input("Select date", value=date.today())

if st.button("Check Anomaly"):
    if not city.strip():
        st.warning("Please enter a city.")
    else:
        try:
            result = predict_live(city.strip(), selected_date.isoformat())

            st.success("Live data fetched successfully.")

            col1, col2 = st.columns(2)
            col1.metric("Anomaly Detected", result["anomaly_detected"])
            col2.metric("Severity", result["severity"])

            st.markdown(f"**Alert:** {result['alert']}")
            st.markdown(f"**Possible cause:** {result['possible_cause']}")

            st.markdown("### Live Environmental Values")
            st.write({
                "City": result["city"],
                "Date": result["date"],
                "PM2.5": round(result["PM2.5"], 2),
                "PM10": round(result["PM10"], 2),
                "NO2": round(result["NO2"], 2),
                "CO": round(result["CO"], 2),
                "SO2": round(result["SO2"], 2),
                "Temperature (2m)": round(result["temperature_2m"], 2),
                "Humidity": round(result["Humidity"], 2),
                "Anomaly Score": round(result["anomaly_score"], 4),
                
            })

            if "recommendations" in result:
                  st.subheader("⚠️ Recommendations")
    
                  for rec in result["recommendations"].split("|"):
                        st.write("• " + rec.strip())

        except Exception as e:
            st.error(f"Error: {e}")