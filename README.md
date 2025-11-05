# Wine Classification CI/CD Project (Simple Version)

This project trains a machine learning model on the **Wine dataset** and serves predictions through a **Flask API**.  
The pipeline includes:
- Training the model
- Comparing new accuracy with previous version
- Deploying the model if performance improves
- Serving predictions via REST API
- Basic monitoring with Prometheus & Grafana